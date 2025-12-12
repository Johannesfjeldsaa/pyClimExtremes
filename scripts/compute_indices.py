import os
import timeit
from pathlib import Path
from reversclim.utils.preprocessing.variables.extremes.indices.registry import (
    resolve_indices,
    resolve_frequencies,
    register_index
)
from reversclim.utils.preprocessing.variables.extremes.io.data_wrapping import (
    prepare_inputs_and_meta,
    prepare_time_groupings,
)
from reversclim.utils.preprocessing.variables.extremes.io.netcdf_write import (
    build_filename, write_index_netcdf
)
from general_backend.utils.save_utils import check_filepath
from general_backend.logging.setup_logging import get_logger

logger = get_logger(__name__)

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
        Which indices to request. 'all' for all creatable indices, else
        provide a single index ID or a list of index IDs.
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
        Additional keyword arguments for future extensions.
    """

    # --- Check all inputs and prepare ---
    start_time_checks = timeit.default_timer()
    # resolve and check indices and frequencies
    index_list = resolve_indices(indices)
    required_inputs = []
    for index_class in index_list:
        required_inputs.extend(index_class.required_vars)
    required_inputs  = list(set(required_inputs))
    logger.debug(
        f"Required input variables for requested indices: {required_inputs}"
    )
    fq_list = resolve_frequencies(compute_fq)

    # check required input paths and prepare loading
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
    input_files_to_use: dict[str, Path] = {var: input_args[var] for var in required_inputs}

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
    for index_class in index_list:
        inited_index_class = index_class(compute_backend, **kwargs)
        for fq in fq_list:
            # Validate frequency is supported by this index
            if fq not in index_class.frequencies:
                warn_msg = (
                    f"Frequency '{fq}' not supported by index '{index_class.index_id}'. "
                    f"Supported frequencies: {index_class.frequencies}. Skipping."
                )
                logger.warning(warn_msg)
                skipped_unsupported.append((index_class.index_id, fq))
                continue
            
            start_time_index_fq = timeit.default_timer()

            arrays = {
                var: wrappers[var].load_ndarray(var)
                for var in index_class.required_vars
            }

            grouping = time_groupings.get(fq)
            
            # Extract threshold if provided in kwargs
            threshold = kwargs.get('threshold', None)
            
            index_values = inited_index_class.compute(
                compute_fq=fq,
                data_array=arrays,
                threshold=threshold,
                group_index=grouping["group_index"],
            )
            meta["time_out"] = grouping["time_out"]
            meta["time_bnds"] = grouping["time_bnds_out"]

            # 4d. Build output file name
            output_filename = build_filename(
                index_id=index_class.index_id,
                compute_fq=fq,
                template=output_file_template,
                **meta,
            )

            output_path = output_dir.joinpath(output_filename)

            if not check_filepath(output_path, overwrite):
                files_created_previously.append(output_path)
                continue

            write_index_netcdf(
                index_values,
                index=index_class,
                metadata=meta,
                output_path=output_path,
            )
            new_files.append(output_path)

            index_fq_time = timeit.default_timer() - start_time_index_fq
            index_timing_map[(index_class.index_id, fq)] = index_fq_time

    indices_comp_time = timeit.default_timer() - start_time_indices_comp
    total_time = timeit.default_timer() - start_time_checks

    timing_summary = (
        f"Time taken for input checks and loading: {checks_time:.2f} seconds.\n"
        f"Time taken for index computations: {indices_comp_time:.2f} seconds.\n"
        f"Total time taken: {total_time:.2f} seconds.\n"
        "Detailed timing per index and frequency:\n"
    )
    for (index_name, fq), duration in index_timing_map.items():
        timing_summary += (
            f" - Index '{index_name}' at frequency '{fq}':"
            f"{duration:.2f} seconds.\n"
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

