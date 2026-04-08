from pathlib import Path
from typing import Any
from netCDF4 import Dataset
import numpy as np

from reversclim.utils.preprocessing.variables.extremes.indices.registry import (
    input_var_str_normalize
)
from pyClimExtremes.logging.setup_logging import get_logger
from reversclim.utils.preprocessing.variables.extremes.indices.units_utils import (
    validate_input_units,
    INPUT_VAR_ALLOWED_INPUT_UNITS,
)

logger = get_logger(__name__)


class DataWrapper:
    """Lightweight wrapper around NetCDF files for index inputs.

    Provides a consistent API `load_ndarray(var)` returning a numpy array
    for the requested variable name. Keeps a lazy handle to the file and
    exposes basic metadata when needed.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._ds = Dataset(self.path)

        # Store original variables dict
        self._raw_variables = self._ds.variables

        # Create a mapping with canonical names as keys
        # This allows access via canonical names regardless of file naming
        self.variables = {}
        self.units = {}
        for var in self._raw_variables:
            canonical = input_var_str_normalize(var)
            # Store under canonical name, but keep reference to actual NC var
            self.variables[canonical] = self._raw_variables[var]
            raw_unit = getattr(self._raw_variables[var], "units", None)
            if (
                raw_unit is not None and
                canonical in INPUT_VAR_ALLOWED_INPUT_UNITS
            ):
                if not validate_input_units(input_var=canonical, input_unit=raw_unit):
                    err_msg = (
                        f"Invalid unit '{raw_unit}' for variable '{canonical}' "
                        f"in file {self.path}"
                    )
                    logger.error(err_msg, stack_info=True)
                    raise ValueError(err_msg)
            self.units[canonical] = raw_unit

        self.global_attrs = {
            attr: getattr(self._ds, attr) for attr in self._ds.ncattrs()
        }
        self.var_attrs = {var: {
                attr: getattr(self.variables[var], attr)
                for attr in self.variables[var].ncattrs()
            } for var in self.variables
        }
        logger.debug(f"Opened DataWrapper for {self.path}")

    def load_ndarray(self, varname: str):
        """Load array for variable using canonical name."""
        canonical_var = input_var_str_normalize(varname)
        if canonical_var not in self.variables:
            err_msg = (
                f"Variable '{varname}' (canonical: '{canonical_var}') "
                f"not found in {self.path}"
            )
            logger.error(err_msg)
            raise KeyError(err_msg)
        return self.variables[canonical_var][:]

    def get_units(self, varname: str) -> str | None:
        """Get units for variable using canonical name."""
        canonical_var = input_var_str_normalize(varname)
        if canonical_var not in self.units:
            err_msg = (
                f"Units for variable '{varname}' (canonical: "
                f"'{canonical_var}') not found in {self.path}"
            )
            logger.error(err_msg)
            raise KeyError(err_msg)
        return self.units[canonical_var]

    def close(self):
        try:
            self._ds.close()
        except Exception:
            pass

def load_input_wrappers(**input_files: dict[str, Path]) -> dict[str, DataWrapper]:
    """Create `DataWrapper`s for provided variable file paths.

    Parameters
    ----------
    **input_files : dict[str, Path]
        Mapping of variable names to NetCDF file paths. Example:
        {"tasmax": Path("/path/to/tasmax.nc"), "tasmin": Path("/path/to/tasmin.nc")}

    Returns
    -------
    dict[str, DataWrapper]
        Mapping of variable names to `DataWrapper` instances.

    Notes
    -----
    This is designed to be used like:
        inputs = load_input_wrappers(**input_files_to_use)
        arrays = {var: inputs[var].load_ndarray(var) for var in required_vars}
    """
    wrappers: dict[str, DataWrapper] = {}
    for var, path in input_files.items():
        wrappers[var] = DataWrapper(Path(path))
    return wrappers


def gather_metadata(
    *,
    wrapper: DataWrapper | None = None,
    close_wrapper: bool = True,
    **input_files: dict[str, Path],
) -> dict[str, Any]:
    """Gather metadata for both filename building and NetCDF writing.

    You may pass a pre-made `DataWrapper` to avoid reopening files. If no
    wrapper is provided, one is constructed from the supplied input files.

    Returns a dict with two sections:
    - String metadata for filename building: source_id, experiment_id,
      variant_label, YYYYMM_start, YYYYMM_end
    - Array/dict metadata for NetCDF writing: lat, lon, time (input time array),
      time_units, calendar, parent_global_attrs, etc.

    Parameters
    ----------
    wrapper : DataWrapper | None, optional
        Existing wrapper to use. If None, one is built from input_files.
    close_wrapper : bool, optional
        Close the wrapper if it was created inside this function. Set False
        when passing an external wrapper you want to keep open.
    **input_files : dict[str, Path]
        Mapping of variable names to NetCDF file paths.

    Returns
    -------
    dict[str, Any]
        Metadata dict containing both string fields (for filename) and
        arrays/dicts (for NetCDF writing).
    """
    created_here = False
    if wrapper is None:
        # Prefer tasmax/tasmin/tas as metadata source, else first item
        preferred_order = ("tasmax", "tasmin", "tas")
        chosen_path = None
        for key in preferred_order:
            if key in input_files:
                chosen_path = Path(input_files[key])
                break
        if chosen_path is None:
            if len(input_files) == 0:
                raise ValueError("No input files provided for metadata gathering")
            first_key = next(iter(input_files))
            chosen_path = Path(input_files[first_key])
        wrapper = DataWrapper(chosen_path)
        created_here = True

    meta: dict[str, Any] = {}

    # Extract CF/CMIP-like attrs if present (for filename and global attrs)
    parent_global_attrs = {}
    for key in ("source_id", "experiment_id", "variant_label", "institution",
                "source", "activity_id", "table_id", "grid_label"):
        value = wrapper.global_attrs.get(key)
        if value is not None:
            parent_global_attrs[key] = value
            # Also store commonly used ones at top level for filename
            if key in ("source_id", "experiment_id", "variant_label"):
                meta[key] = str(value)

    meta["parent_global_attrs"] = parent_global_attrs
    meta["child_global_attrs"] = {}  # Can be populated by caller
    meta["parent_var_attrs"] = {}    # Can be populated by caller
    meta["child_var_attrs"] = {}     # Can be populated by caller

    # Extract coordinate arrays (lat, lon, time)
    try:
        # Latitude
        for cand in ("lat", "latitude"):
            if cand in wrapper.variables:
                meta["lat"] = wrapper.variables[cand][:]
                break
        else:
            meta["lat"] = None

        # Latitude bounds
        for cand in ("lat_bnds", "latitude_bnds"):
            if cand in wrapper.variables:
                meta["lat_bnds"] = wrapper.variables[cand][:]
                break
        else:
            meta["lat_bnds"] = None

        # Longitude
        for cand in ("lon", "longitude"):
            if cand in wrapper.variables:
                meta["lon"] = wrapper.variables[cand][:]
                break
        else:
            meta["lon"] = None

        # Longitude bounds
        for cand in ("lon_bnds", "longitude_bnds"):
            if cand in wrapper.variables:
                meta["lon_bnds"] = wrapper.variables[cand][:]
                break
        else:
            meta["lon_bnds"] = None

        # Time coordinate and metadata
        time_var = None
        for cand in ("time", "Time"):
            if cand in wrapper.variables:
                time_var = wrapper.variables[cand]
                break

        if time_var is not None:
            from netCDF4 import num2date
            units = getattr(time_var, "units", "days since 1850-01-01")
            calendar = getattr(time_var, "calendar", "standard")
            times_num = time_var[:]

            meta["time"] = times_num  # Store numeric time for later processing
            meta["time_units"] = units
            meta["calendar"] = calendar

            # Extract time_bnds if available
            if "time_bnds" in wrapper.variables:
                meta["time_bnds_in"] = wrapper.variables["time_bnds"][:]
            else:
                meta["time_bnds_in"] = None

            if times_num.size > 0:
                times = num2date(times_num, units=units, calendar=calendar)
                start_dt = times[0]
                end_dt = times[-1]
                # Monthly format (YYYYMM)
                meta["YYYYMM_start"] = f"{start_dt.year:04d}{start_dt.month:02d}"
                meta["YYYYMM_end"] = f"{end_dt.year:04d}{end_dt.month:02d}"
                # Annual format (YYYY)
                meta["YYYY_start"] = f"{start_dt.year:04d}"
                meta["YYYY_end"] = f"{end_dt.year:04d}"
            else:
                meta.setdefault("YYYYMM_start", "unknown")
                meta.setdefault("YYYYMM_end", "unknown")
                meta.setdefault("YYYY_start", "unknown")
                meta.setdefault("YYYY_end", "unknown")
        else:
            meta.setdefault("time", None)
            meta.setdefault("time_units", "unknown")
            meta.setdefault("calendar", "standard")
            meta.setdefault("YYYYMM_start", "unknown")
            meta.setdefault("YYYYMM_end", "unknown")
            meta.setdefault("YYYY_start", "unknown")
            meta.setdefault("YYYY_end", "unknown")

    except Exception:
        meta.setdefault("lat", None)
        meta.setdefault("lon", None)
        meta.setdefault("time", None)
        meta.setdefault("time_units", "unknown")
        meta.setdefault("calendar", "standard")
        meta.setdefault("YYYYMM_start", "unknown")
        meta.setdefault("YYYYMM_end", "unknown")
        meta.setdefault("YYYY_start", "unknown")
        meta.setdefault("YYYY_end", "unknown")

    if created_here and close_wrapper:
        wrapper.close()
    return meta


def prepare_time_groupings(
    *,
    fq_list: list[str],
    compute_backend: str,
    metadata: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Pre-compute time groupings and output timestamps for requested frequencies.

    Uses metadata from :func:`gather_metadata` to avoid reopening files. Returns
    a mapping from frequency (e.g., ``"mon"``) to ``{"group_index": inv,
    "time_out": time_out}`` where ``group_index`` is the inverse index from
    ``np.unique`` and ``time_out`` are mean-of-period time coordinates.
    """

    from reversclim.utils.preprocessing.variables.extremes.compute_backend.backend_registry import (
        get_compute_backend,
    )

    backend = get_compute_backend(compute_backend)

    time_array = metadata.get("time")
    time_bnds_in = metadata.get("time_bnds_in")
    time_units = metadata.get("time_units")
    calendar = metadata.get("calendar")

    time_info: dict[str, dict[str, Any]] = {}
    for fq in fq_list:
        _, inv = backend.group_indices(
            compute_fq=fq,
            time_array=time_array,
            time_units=time_units,
            calendar=calendar,
        )

        # If input has time_bnds, use month/year boundaries for output
        # Otherwise use mean of time_array
        if time_bnds_in is not None:
            # Each row of time_bnds_in is [lower, upper] for that timestep
            n_groups = int(inv.max()) + 1 if inv.size else 0
            time_out = np.empty(n_groups, dtype=float)
            time_bnds_out = np.empty((n_groups, 2), dtype=float)

            for i in range(n_groups):
                mask = inv == i
                # For this group, get indices of timesteps
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    # Month bounds: lower of first day and of last day
                    lower_bound = float(time_bnds_in[indices[0], 0])
                    upper_bound = float(time_bnds_in[indices[-1], 0])

                    # Output time: mean of bounds
                    time_out[i] = (lower_bound + upper_bound) / 2.0
                    time_bnds_out[i, 0] = lower_bound
                    time_bnds_out[i, 1] = upper_bound
        else:
            # Fall back to time_array mean
            time_out = backend.get_time_out(
                compute_fq=fq,
                time_array=time_array,
                time_units=time_units,
                calendar=calendar,
                group_index=inv,
            )
            # Compute time bounds (min, max of each group)
            n_groups = int(inv.max()) + 1 if inv.size else 0
            time_bnds_out = np.empty((n_groups, 2), dtype=float)
            for i in range(n_groups):
                group_times = time_array[inv == i]
                time_bnds_out[i, 0] = float(group_times.min())
                time_bnds_out[i, 1] = float(group_times.max())

        time_info[fq] = {
            "group_index": inv,
            "time_out": time_out,
            "time_bnds_out": time_bnds_out,
        }

    return time_info


def prepare_inputs_and_meta(
    *,
    wrappers: dict[str, DataWrapper] | None = None,
    close_meta_wrapper: bool = False,
    **input_files: dict[str, Path],
) -> tuple[dict[str, DataWrapper], dict[str, Any]]:
    """Create input wrappers and metadata with shared file handles to reduce IO.

    Accepts pre-made wrappers to avoid reopening files. Metadata extraction
    reuses one of the wrappers and closes it only if it was created here and
    `close_meta_wrapper` is True.

    Returns
    -------
    tuple[dict[str, DataWrapper], dict[str, Any]]
        - dictionary mapping variable names to DataWrapper instances
        - Metadata dict containing both filename strings and NetCDF arrays/attrs
    """
    created_wrappers = False
    if wrappers is None:
        wrappers = load_input_wrappers(**input_files)
        created_wrappers = True

    # Choose a wrapper as in gather_metadata's preference order
    preferred_order = ("tasmax", "tasmin", "tas")
    chosen_wrapper = None
    for key in preferred_order:
        if key in wrappers:
            chosen_wrapper = wrappers[key]
            break
    if chosen_wrapper is None:
        if len(wrappers) == 0:
            raise ValueError("No input files provided")
        chosen_wrapper = next(iter(wrappers.values()))

    meta = gather_metadata(
        wrapper=chosen_wrapper,
        close_wrapper=(created_wrappers and close_meta_wrapper)
    )

    return wrappers, meta
