import os
import timeit
import inspect
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
from pyClimExtremes.indices.base_index import QuantileIndex, QuantileThresholdIndex
from pyClimExtremes.indices.registry import (
    resolve_indices,
    resolve_frequencies,
    input_var_str_normalize,
    INPUT_VAR_ALIASES,
)
from pyClimExtremes.indices.units_utils import (
    validate_input_units,
    unit_str_normalize,
    convert_units,
)
from pyClimExtremes.scripts.compute_thresholds import compute_threshold_array
from pyClimExtremes.io.data_wrapping import (
    prepare_inputs_and_meta,
    prepare_time_groupings,
)
from pyClimExtremes.io.netcdf_write import (
    build_filename, write_index_netcdf
)
from pyClimExtremes.io.save_utils import check_filepath
from pyClimExtremes.logging.setup_logging import get_logger

logger = get_logger(__name__)

_UNKNOWN_META_STRINGS = {"", "unknown", "none", "null", "nan", "na"}


def _normalize_quantile_threshold_files(
    quantile_threshold_files,
) -> dict[str, Path]:
    """Normalize saved quantile-threshold files to a mapping by threshold id."""
    if quantile_threshold_files is None:
        return {}

    normalized: dict[str, Path] = {}
    if isinstance(quantile_threshold_files, dict):
        items = quantile_threshold_files.items()
    else:
        items = []
        for file_path in quantile_threshold_files:
            path = Path(file_path)
            stem = path.stem
            if "_threshold_" not in stem:
                err_msg = (
                    "Quantile threshold file names must contain '_threshold_' "
                    f"to infer the source threshold id: {path}"
                )
                logger.error(err_msg, stack_info=True)
                raise ValueError(err_msg)
            items.append((stem.split("_threshold_", 1)[0], path))

    for threshold_id, file_path in items:
        path = Path(file_path)
        if not path.exists():
            err_msg = f"Quantile threshold file not found: {path}"
            logger.error(err_msg, stack_info=True)
            raise FileNotFoundError(err_msg)
        normalized[str(threshold_id)] = path

    return normalized


def _load_quantile_threshold_array(path: Path) -> np.ndarray:
    """Load the threshold variable from a saved quantile-threshold NetCDF."""
    with Dataset(path) as ds:
        if "threshold" not in ds.variables:
            err_msg = f"No 'threshold' variable found in quantile threshold file: {path}"
            logger.error(err_msg, stack_info=True)
            raise KeyError(err_msg)
        return np.asarray(ds.variables["threshold"][:])


def _resolve_quantile_index_class(threshold_index_id: str) -> type[QuantileIndex]:
    """Resolve and validate the QuantileIndex class backing a threshold id."""
    resolved = resolve_indices([threshold_index_id])
    if len(resolved) != 1:
        err_msg = (
            "Expected exactly one quantile threshold source index for "
            f"'{threshold_index_id}', got {len(resolved)}."
        )
        logger.error(err_msg, stack_info=True)
        raise ValueError(err_msg)

    threshold_index_class = resolved[0]
    if not issubclass(threshold_index_class, QuantileIndex):
        err_msg = (
            f"Threshold source index '{threshold_index_id}' must resolve to a "
            "QuantileIndex subclass."
        )
        logger.error(err_msg, stack_info=True)
        raise TypeError(err_msg)

    return threshold_index_class


def _get_or_compute_quantile_threshold_array(
    *,
    threshold_index_id: str,
    quantile_threshold_files: dict[str, Path],
    quantile_threshold_cache: dict[str | Path, np.ndarray],
    compute_backend: str,
    backend_kwargs: dict,
    wrappers: dict,
    meta: dict,
    time_groupings: dict[str, dict[str, np.ndarray]],
    window_size: int,
    bootstrap_samples: int,
    random_seed: int | None,
) -> np.ndarray:
    """Return a quantile-threshold array from cache, file, or in-memory generation."""
    threshold_file = quantile_threshold_files.get(threshold_index_id)
    cache_key: str | Path = threshold_file if threshold_file is not None else threshold_index_id

    if cache_key in quantile_threshold_cache:
        return quantile_threshold_cache[cache_key]

    if threshold_file is not None:
        threshold_array = _load_quantile_threshold_array(threshold_file)
    else:
        threshold_index_class = _resolve_quantile_index_class(threshold_index_id)
        threshold_index_obj = threshold_index_class(compute_backend, **backend_kwargs)
        threshold_arrays = {
            var: wrappers[var].load_ndarray(var)
            for var in threshold_index_class.required_vars
        }
        threshold_units = {
            var: wrappers[var].get_units(var)
            for var in threshold_index_class.required_vars
        }

        for var, unit in threshold_units.items():
            if unit is None:
                err_msg = (
                    f"Input variable '{var}' is missing units for "
                    f"threshold source index '{threshold_index_id}'."
                )
                logger.error(err_msg, stack_info=True)
                raise ValueError(err_msg)
            if not validate_input_units(var, unit):
                err_msg = (
                    f"Invalid input units '{unit}' for variable '{var}' "
                    f"used by threshold source index '{threshold_index_id}'."
                )
                logger.error(err_msg, stack_info=True)
                raise ValueError(err_msg)

        logger.info(
            "Generating quantile threshold array for '%s' in memory.",
            threshold_index_id,
        )
        threshold_array = compute_threshold_array(
            index_class=threshold_index_class,
            index_obj=threshold_index_obj,
            arrays=threshold_arrays,
            units=threshold_units,
            meta=meta,
            time_groupings=time_groupings,
            window_size=window_size,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed,
        )

    quantile_threshold_cache[cache_key] = np.asarray(threshold_array)
    return quantile_threshold_cache[cache_key]


def _is_missing_or_unknown_meta(value) -> bool:
    """Return True for metadata values that are missing or placeholder-ish."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _UNKNOWN_META_STRINGS
    return False


def _validate_meta_value(meta_key: str, value) -> None:
    """Validate metadata values before forwarding them into backend calls."""
    if _is_missing_or_unknown_meta(value):
        err_msg = (
            f"Metadata key '{meta_key}' is missing or unknown (value={value!r})."
        )
        logger.error(err_msg, stack_info=True)
        raise ValueError(err_msg)

    if meta_key in {"time", "lat"}:
        arr = np.asarray(value)
        if arr.size == 0:
            err_msg = f"Metadata key '{meta_key}' is empty."
            logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)

# TODO:
# output_file_template:
# * should be computable based on standard styles
# (e.g., 'cmip6', 'obs4mips', etc.).
# * should support passing metadata dict directly for custom naming.
def compute_indices(
    indices:                str | list[str],
    compute_fq:             str | list[str],
    compute_backend:        str,
    output_dir:             Path,
    output_file_template:   str | list[str] = "cmip6",
    tasmax:                 Path | None = None,
    tasmin:                 Path | None = None,
    tas:                   Path | None = None,
    pr:                     Path | None = None,
    overwrite:              bool = False,
    **kwargs
) -> tuple[list[Path] | None, ...]:
    """_summary_

    Parameters
    ----------
    indices : str | list[str]
        Which indices to request. Options:
        * 'all' for all creatable indices,
        * 'temperature' for all temperature indices,
        * 'precipitation' for all precipitation indices,
        else provide a single index ID or a list of index IDs.
    compute_fq : str | list[str]
        Which frequencies to request. 'all' for all supported frequencies,
        else provide a single frequency or a list of frequencies.
    compute_backend : str
        The name of the compute backend to use for calculations.
        For now only 'python' is supported.
    output_dir : Path
        Directory where output files will be saved.
    output_file_template : str | list[str], optional
        _description_, by default "cmip6"
    tasmax : Path | list[Path] | None, optional
        _description_, by default None
    tasmin : Path | list[Path] | None, optional
        _description_, by default None
    tas : Path | list[Path] | None, optional
        _description_, by default None
    pr : Path | list[Path] | None, optional
        _description_, by default None
    overwrite : bool, optional
        Whether to overwrite existing output files, by default False
    **kwargs : dict
        Additional keyword arguments. Supported keys:
        - 'threshold': dict mapping index IDs to threshold values.
          Example: {'SU': [25, 30] , 'rnnmm': [10, 15]} computes each index
          for each of its specified thresholds separately. Note:
          * The threshold values should be provided in the same units as the
          input data when applicable. The 'default_threshold' attribute is
          will attempt to use the correct units based on the 'units' attribute
          of the input data variable.
          * If an index is not threshold-based, the value(s) are ignored.
          * If an index is threshold-based but not specified here,
          its default threshold (if any) will be used. If no default threshold
          is set, an error is raised unless the index was requested through
          'all', 'temperature' or 'precipitation', in which case it is skipped
          with a warning.
          * Usage of index aliases as keys is also supported, e.g., 'SU', 'su'
          or 'suETCCDI' is acceptable for the SUINDEX.
        - 'spells_can_span_groups': dict mapping spell-duration index IDs to
            boolean values.
            Example: {'cdd': False, 'cwdETCCDI': True} overrides the per-index
            class defaults for spell handling.
            * If an index is not a spell-duration index, the value is ignored.
            * Usage of index aliases as keys is also supported.
        - 'backend_kwargs': dict of additional kwargs to pass to the compute
            backend when initializing index classes.
        - 'quantile_threshold_files': saved quantile-threshold NetCDF files
            used by quantile-threshold precipitation indices. This can be a
            dict like {'q95pr': path1, 'q99pr': path2} or a list of files whose
            names follow '<threshold_id>_threshold_*.nc'. If omitted, the
            required threshold array is generated in memory and cached for the
            duration of the compute_indices call.
        - 'window_size', 'bootstrap_samples', 'random_seed': optional quantile
            threshold generation settings used when in-memory threshold
            generation is needed for quantile-threshold indices.
    """

    # ------------------------------------ #
    # --- Check all inputs and prepare --- #
    # ------------------------------------ #

    start_time_checks = timeit.default_timer()
    # resolve and check indices and frequencies
    index_list = resolve_indices(indices)

    # Extract threshold, spell options, and backend kwargs from kwargs
    threshold_dict = kwargs.get('threshold', {})
    spells_can_span_groups_dict = kwargs.get('spells_can_span_groups', {})
    backend_kwargs = kwargs.get('backend_kwargs', {})
    window_size = kwargs.get('window_size', 5)
    bootstrap_samples = kwargs.get('bootstrap_samples', 1000)
    random_seed = kwargs.get('random_seed', None)
    quantile_threshold_files = _normalize_quantile_threshold_files(
        kwargs.get('quantile_threshold_files', None)
    )
    quantile_threshold_cache: dict[str | Path, np.ndarray] = {}

    # Filter out ThresholdIndex with default_threshold = None
    # unless explicitly requested in threshold_dict
    # if umbrella requests ('all', 'temperature', 'precipitation'),
    # skip with warning instead of error.
    filtered_index_list = []
    for index_class in index_list:
        # Check if this is a ThresholdIndex with no default threshold
        if hasattr(index_class, 'default_threshold') and index_class.default_threshold is None:
            # Check if it's explicitly provided in threshold_dict
            threshold_provided = False
            if index_class.index_id in threshold_dict:
                threshold_provided = True
            elif hasattr(index_class, 'index_aliases'):
                for alias in index_class.index_aliases:
                    if alias in threshold_dict:
                        threshold_provided = True
                        break

            if not threshold_provided:
                if indices in ['all', 'temperature', 'precipitation']:
                    warn_msg = (
                        f"Skipping index '{index_class.index_id}' because it "
                        "requires an explicit threshold value "
                        "(no default threshold set). Provide threshold in "
                        f"kwargs['threshold']['{index_class.index_id}']."
                    )
                    logger.warning(warn_msg)
                    continue
                else:
                    err_msg = (
                        f"Index '{index_class.index_id}' requires an "
                        "explicit threshold value (no default threshold set), "
                        "but none was provided. Provide threshold in "
                        f"kwargs['threshold']['{index_class.index_id}']."
                    )
                    logger.error(err_msg, stack_info=True)
                    raise ValueError(err_msg)

        filtered_index_list.append(index_class)
    index_list = filtered_index_list

    # resolve frequencies
    fq_list = resolve_frequencies(compute_fq)

    # check required input paths, prepare loading and metadata
    required_inputs = []
    for index_class in index_list:
        required_inputs.extend(index_class.required_vars)
    required_inputs  = list(set(required_inputs))
    logger.debug(
        "Required input variables for requested indices: %s",
        required_inputs
    )
    input_args = {
        "tasmax": tasmax,
        "tasmin": tasmin,
        "tas": tas,
        "pr": pr,
    }
    for var in required_inputs:
        path = input_args.get(var)
        if path is None:
            err_msg = (
                f"Input variable '{var}' is required for the requested "
                f"indices but was not provided."
            )
            logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        # check file existence if path provided
        if not isinstance(path, Path):
            path = Path(path)
        if not path.exists() and path.suffix == ".nc":
            err_msg = (
                f"Valid input file for variable '{var}' not found at: {path}"
            )
            logger.error(err_msg, stack_info=True)
            raise FileNotFoundError(err_msg)
    input_files_to_use: dict[str, Path] = {
        var: input_args[var] for var in required_inputs
    }

    # create data wrappers and extract metadata
    wrappers, meta = prepare_inputs_and_meta(
        **input_files_to_use
    )

    time_groupings = prepare_time_groupings(
        fq_list=fq_list,
        compute_backend=compute_backend,
        metadata=meta,
    )
    # ensure output directory exists
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # check if we have write permissions in output directory
    if not os.access(output_dir, os.W_OK):
        err_msg = f"No write permissions in output directory: {output_dir}"
        logger.error(err_msg, stack_info=True)
        raise PermissionError(err_msg)
    # check overwrite flag is boolean
    if not isinstance(overwrite, bool):
        err_msg = f"overwrite argument must be boolean, got: {type(overwrite)}"
        logger.error(err_msg, stack_info=True)
        raise ValueError(err_msg)

    checks_time = timeit.default_timer() - start_time_checks

    # --- Compute indices ---
    start_time_indices_comp = timeit.default_timer()
    new_files = []
    files_created_previously = []
    skipped_unsupported = []
    index_timing_map = {}
    init_index_elapse_time = []
    get_arrays_elapsed_time = []
    get_units_elapsed_time = []
    get_time_grouper_elapsed_time = []
    units_and_thresholds_elapsed_time = []
    compute_elapsed_time = []

    for index_class in index_list:
        init_index_timer = timeit.default_timer()
        inited_index_class = index_class(compute_backend, **backend_kwargs)
        init_index_elapse_time.append(timeit.default_timer() - init_index_timer)

        for fq in fq_list:
            start_time_index_fq = timeit.default_timer()

            # Inspect backend method signature once per index/frequency so we
            # only validate/pass metadata args that are actually consumed.
            backend_method = getattr(
                inited_index_class.compute_backend,
                index_class.backend_callable_name,
            )
            backend_sig = inspect.signature(inspect.unwrap(backend_method))
            backend_params = set(backend_sig.parameters.keys())

            # Validate frequency is supported by this index
            if fq not in index_class.frequencies:
                warn_msg = (
                    f"Frequency '{fq}' not supported by index '{index_class.index_id}'. "
                    f"Supported frequencies: {index_class.frequencies}. Skipping."
                )
                logger.warning(warn_msg)
                skipped_unsupported.append((index_class.index_id, fq))
                continue

            get_arrays_timer = timeit.default_timer()
            # get required input from data wrappers
            arrays = {
                var: wrappers[var].load_ndarray(var)
                for var in index_class.required_vars
            }
            get_arrays_elapsed_time.append(timeit.default_timer() - get_arrays_timer)
            get_units_timer = timeit.default_timer()
            units = {
                var: wrappers[var].get_units(var)
                for var in index_class.required_vars
            }
            get_units_elapsed_time.append(timeit.default_timer() - get_units_timer)
            get_time_grouper_timer = timeit.default_timer()
            time_grouper = time_groupings.get(fq)

            if time_grouper is None:
                err_msg = (
                    f"No time grouping available for frequency '{fq}'. "
                    "Verify that prepare_time_groupings returned entries "
                    "for all requested frequencies."
                )
                logger.error(err_msg, stack_info=True)
                raise ValueError(err_msg)
            get_time_grouper_elapsed_time.append(timeit.default_timer() - get_time_grouper_timer)

            units_and_thresholds_timer = timeit.default_timer()
            # Validate units and normalize for threshold matching
            normalized_units = {}
            for var, unit in units.items():
                if unit is None:
                    err_msg = (
                        f"Input variable '{var}' is missing units; "
                        f"cannot determine thresholds for index '{index_class.index_id}'."
                    )
                    logger.error(err_msg, stack_info=True)
                    raise ValueError(err_msg)
                if not validate_input_units(var, unit):
                    err_msg = (
                        f"Input unit '{unit}' for variable '{var}' is invalid "
                        f"for index '{index_class.index_id}'."
                    )
                    logger.error(err_msg, stack_info=True)
                    raise ValueError(err_msg)
                normalized_units[var] = unit_str_normalize(unit)

            # Use the first required var as the reference unit for thresholds
            primary_var = index_class.required_vars[0]
            primary_unit = units[primary_var]
            primary_unit_norm = normalized_units[primary_var]

            def _threshold_for_unit(threshold_map: dict[str, float]) -> float | None:
                for unit_key, threshold_val in threshold_map.items():
                    if unit_str_normalize(unit_key) == primary_unit_norm:
                        return threshold_val
                return None

            # Resolve fixed thresholds for BaseIndex subclasses
            fixed_threshold_value = None
            if getattr(index_class, "fixed_threshold", None) is not None:
                fixed_threshold_value = _threshold_for_unit(index_class.fixed_threshold)
                if fixed_threshold_value is None:
                    err_msg = (
                        f"Could not resolve fixed_threshold for index '{index_class.index_id}' "
                        f"with input unit '{primary_unit}'."
                    )
                    logger.error(err_msg, stack_info=True)
                    raise ValueError(err_msg)

            # Threshold handling - there is a difference between BaseIndex
            # and ThresholdIndex index_classes.
            # * BaseIndex: thresholds are provided by index_class under
            #   'fixed_threshold' attribute specified by data units.
            # * ThresholdIndex: thresholds can be provided by user
            #   (in kwargs['threshold']) or else the index_class's
            #   'default_threshold' attribute is used.

            is_threshold_index = hasattr(index_class, 'default_threshold')
            is_quantile_threshold_index = issubclass(
                index_class,
                QuantileThresholdIndex,
            )

            # Look for threshold values for this index in the threshold dict
            # Check both index_id and any aliases
            index_thresholds = threshold_dict.get(index_class.index_id, None)
            if index_thresholds is None and hasattr(index_class, 'index_aliases'):
                for alias in index_class.index_aliases:
                    index_thresholds = threshold_dict.get(alias, None)
                    if index_thresholds is not None:
                        break

            user_thresholds: list[float] = []
            if index_thresholds is not None:
                if isinstance(index_thresholds, list):
                    user_thresholds = [thr for thr in index_thresholds if thr is not None]
                else:
                    user_thresholds = [index_thresholds]

            thresholds_to_run: list[float | None] = []
            if is_threshold_index:
                if user_thresholds:
                    thresholds_to_run = list(user_thresholds)
                else:
                    default_threshold = getattr(index_class, 'default_threshold', None)
                    if default_threshold is None:
                        err_msg = (
                            f"Index '{index_class.index_id}' requires a threshold but "
                            "neither user-provided nor default thresholds were found."
                        )
                        logger.error(err_msg, stack_info=True)
                        raise ValueError(err_msg)
                    if isinstance(default_threshold, dict):
                        resolved_default = _threshold_for_unit(default_threshold)
                        if resolved_default is None:
                            err_msg = (
                                f"Could not resolve default_threshold for index '{index_class.index_id}' "
                                f"with input unit '{primary_unit}'."
                            )
                            logger.error(err_msg, stack_info=True)
                            raise ValueError(err_msg)
                        thresholds_to_run = [resolved_default]
                    else:
                        thresholds_to_run = [default_threshold]
            else:
                thresholds_to_run = [None]
            units_and_thresholds_elapsed_time.append(timeit.default_timer() - units_and_thresholds_timer)

            # Spell handling - there is a difference between BaseIndex
            # and SpellDurationIndex index_classes.
            # * SpellDurationIndex: has attribute 'spells_can_span_groups'
            # that controls whether to allow spells to span group boundaries
            # (e.g., years).
            is_spell_duration_index = hasattr(index_class, 'spells_can_span_groups')

            spell_span_groups_value = None
            if is_spell_duration_index:
                spell_span_groups_value = spells_can_span_groups_dict.get(
                    index_class.index_id,
                    None,
                )
                if (
                    spell_span_groups_value is None and
                    hasattr(index_class, 'index_aliases')
                ):
                    for alias in index_class.index_aliases:
                        spell_span_groups_value = spells_can_span_groups_dict.get(alias, None)
                        if spell_span_groups_value is not None:
                            break

                if spell_span_groups_value is None:
                    spell_span_groups_value = getattr(
                        index_class,
                        'spells_can_span_groups',
                    )

                if not isinstance(spell_span_groups_value, bool):
                    err_msg = (
                        "spells_can_span_groups must resolve to a boolean "
                        f"for index '{index_class.index_id}', got "
                        f"{type(spell_span_groups_value)}."
                    )
                    logger.error(err_msg, stack_info=True)
                    raise ValueError(err_msg)
                logger.debug(
                    "Resolved spells_can_span_groups for '%s': %s "
                    "(source: %s)",
                    index_class.index_id,
                    spell_span_groups_value,
                    "user_override" if spell_span_groups_value != getattr(index_class, 'spells_can_span_groups')
                    else "class_default"
                )

            # Compute index for each threshold value
            for threshold_value in thresholds_to_run:
                compute_timer = timeit.default_timer()

                compute_meta_kwargs = {}
                meta_to_kwarg = {
                    "time": "time_array",
                    "time_units": "time_units",
                    "calendar": "calendar",
                    "lat": "lat",
                }
                for meta_key, kwarg_name in meta_to_kwarg.items():
                    if kwarg_name not in backend_params:
                        continue

                    meta_val = meta.get(meta_key)
                    _validate_meta_value(meta_key, meta_val)
                    compute_meta_kwargs[kwarg_name] = meta_val

                # start assemble compute kwargs per threshold
                compute_kwargs = {
                    "compute_fq": fq,
                    "data_array": arrays,
                    "group_index": time_grouper["group_index"],
                }
                compute_kwargs.update(compute_meta_kwargs)

                comp_info = (
                        f"Computing index '{index_class.index_id}' "
                        f"at frequency '{fq}'. "
                        f"Data units: '{primary_unit}' and "
                    )
                if fixed_threshold_value is not None:
                    compute_kwargs["fixed_threshold"] = fixed_threshold_value
                    comp_info += f"fixed_threshold = {fixed_threshold_value}."

                if is_threshold_index:
                    compute_kwargs["threshold"] = threshold_value
                    comp_info += f"threshold = {threshold_value}."

                if is_spell_duration_index:
                    compute_kwargs["spells_can_span_groups"] = spell_span_groups_value
                    comp_info += (
                        f"spells_can_span_groups = {spell_span_groups_value}."
                    )

                if is_quantile_threshold_index:
                    threshold_index_id = getattr(
                        index_class,
                        'quantile_threshold_index_id',
                        None,
                    )
                    if threshold_index_id is None:
                        err_msg = (
                            f"Quantile-threshold index '{index_class.index_id}' is missing "
                            "the 'quantile_threshold_index_id' attribute."
                        )
                        logger.error(err_msg, stack_info=True)
                        raise ValueError(err_msg)

                    threshold_file = quantile_threshold_files.get(threshold_index_id)
                    compute_kwargs["quantile_thresholds"] = _get_or_compute_quantile_threshold_array(
                        threshold_index_id=threshold_index_id,
                        quantile_threshold_files=quantile_threshold_files,
                        quantile_threshold_cache=quantile_threshold_cache,
                        compute_backend=compute_backend,
                        backend_kwargs=backend_kwargs,
                        wrappers=wrappers,
                        meta=meta,
                        time_groupings=time_groupings,
                        window_size=window_size,
                        bootstrap_samples=bootstrap_samples,
                        random_seed=random_seed,
                    )
                    if threshold_file is not None:
                        comp_info += f"quantile_threshold_file = {threshold_file}."
                    else:
                        comp_info += (
                            f"quantile_threshold_generated = {threshold_index_id}."
                        )

                try:
                    logger.info(comp_info)
                    index_values = inited_index_class.compute(**compute_kwargs)
                    meta["time_out"] = time_grouper["time_out"]
                    meta["time_bnds"] = time_grouper["time_bnds_out"]
                except Exception as e:
                    err_msg = (
                        f"Error computing index '{index_class.index_id}' "
                        f"at frequency '{fq}': {e}"
                    )
                    logger.error(err_msg, stack_info=True)
                    raise RuntimeError(err_msg) from e

                # check if we need to convert units after computation
                unit_after_compute = index_class.unit_after_aggregation.get(
                    primary_unit
                )

                if unit_after_compute != index_class.index_units:
                    logger.info(
                        "Converting computed index units from '%s' to '%s'.",
                        unit_after_compute,
                        index_class.index_units
                    )
                    try:
                        index_values = convert_units(
                            index_values,
                            from_unit=unit_after_compute,
                            to_unit=index_class.index_units,
                        )
                    except ValueError as e:
                        err_msg = (
                            "Unit conversion failed for index "
                            f"'{index_class.index_id}': {e}"
                        )
                        logger.error(err_msg, stack_info=True)
                        raise RuntimeError(err_msg) from e


                # Add threshold and index_class to metadata for filename
                meta_for_file = meta.copy()
                if is_threshold_index:
                    meta_for_file["threshold"] = threshold_value
                    meta_for_file["index_class"] = index_class

                output_filename = build_filename(
                    index_id=index_class.index_id,
                    compute_fq=fq,
                    template=output_file_template,
                    **meta_for_file,
                )

                output_path = output_dir.joinpath(output_filename)

                if not check_filepath(output_path, overwrite):
                    files_created_previously.append(output_path)
                    continue

                write_index_netcdf(
                    index_values,
                    index=index_class,
                    metadata=meta_for_file,
                    output_path=output_path,
                )
                new_files.append(output_path)
                compute_elapsed_time.append(timeit.default_timer() - compute_timer)

            index_fq_time = timeit.default_timer() - start_time_index_fq
            index_timing_map[(index_class.index_id, fq)] = index_fq_time

    indices_comp_time = timeit.default_timer() - start_time_indices_comp
    total_time = timeit.default_timer() - start_time_checks

    # --- Log summary ---
    # average times for the various steps
    avg_init_index_time = sum(init_index_elapse_time) / len(init_index_elapse_time) if init_index_elapse_time else 0.0
    avg_get_arrays_time = sum(get_arrays_elapsed_time) / len(get_arrays_elapsed_time) if get_arrays_elapsed_time else 0.0
    avg_get_units_time = sum(get_units_elapsed_time) / len(get_units_elapsed_time) if get_units_elapsed_time else 0.0
    avg_get_time_grouper_time = sum(get_time_grouper_elapsed_time) / len(get_time_grouper_elapsed_time) if get_time_grouper_elapsed_time else 0.0
    avg_units_and_thresholds_time = sum(units_and_thresholds_elapsed_time) / len(units_and_thresholds_elapsed_time) if units_and_thresholds_elapsed_time else 0.0
    avg_compute_time = sum(compute_elapsed_time) / len(compute_elapsed_time) if compute_elapsed_time else 0.0

    timing_summary = (
        f"Time taken for input checks and loading: {checks_time:.2f} secs.\n"
        f"Time taken for index computations: {indices_comp_time:.2f} secs.\n"
        f" - Average index class initialization time: {avg_init_index_time:.2f} secs.\n"
        f" - Average get_arrays time: {avg_get_arrays_time:.2f} secs.\n"
        f" - Average get_units time: {avg_get_units_time:.2f} secs.\n"
        f" - Average get_time_grouper time: {avg_get_time_grouper_time:.2f} secs.\n"
        f" - Average units and thresholds resolution time: {avg_units_and_thresholds_time:.2f} secs.\n"
        f" - Average compute time: {avg_compute_time:.2f} secs.\n"
        f"Total time taken: {total_time:.2f} secs.\n"
        "Detailed timing per index and frequency:\n"
    )
    for (index_name, fq), duration in index_timing_map.items():
        timing_summary += (
            f" - Index '{index_name}' at frequency '{fq}':"
            f"{duration:.2f} secs.\n"
    )

    summary_msg = (
        "Index computation completed. "
        f"Total indices computed: {len(index_list)}"
        f" with timing details:\n{timing_summary}"
    )
    summary_msg += (
        f"New files created: {len(new_files)}." if new_files else
        f"No new files created."
    )
    summary_msg += (
        f"Files skipped (already existed): {len(files_created_previously)}."
        if files_created_previously else ""
    )
    summary_msg += (
        f"\nSkipped unsupported index-frequency combinations: {len(skipped_unsupported)}."
        if skipped_unsupported else ""
    )
    if skipped_unsupported:
        summary_msg += "\nUnsupported combinations:\n"
        for idx_id, freq in skipped_unsupported:
            summary_msg += f"  - {idx_id} at {freq}\n"
    logger.info(summary_msg)

    return new_files, files_created_previously

