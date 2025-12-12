# io/netcdf_writer.py
from pathlib import Path
from typing import Any
from netCDF4 import Dataset
import numpy as np

from reversclim.utils.preprocessing.variables.extremes.indices.base_index import (
    BaseIndex,
)

from general_backend.logging.setup_logging import get_logger

logger = get_logger(__name__)


def build_filename(
    index_id:   str,
    compute_fq: str,
    template:   str | list[str],
    **meta:     dict[str, str]
) -> str:
    """Builds a filename based on the provided template and metadata
    collected from the input datasets.

    Parameters
    ----------
    index_id : str
        The index identifier (e.g., 'TXx').
    compute_fq : str
        The computation frequency (e.g., 'ann').
    template : str | list[str]
        The filename template, either as a predefined style or a list of
        metadata keys.
    meta : dict[str, str]
        Additional metadata for filename construction. Should include all keys
        referenced in the template apart from 'index' and 'compute_fq'.

    Returns
    -------
    str
        The constructed filename.
    """
    if isinstance(template, str):
        if template == "cmip6":
            template_keys = [
                "index", "compute_fq", "source_id", "experiment_id",
                "variant_label", "YYYYMM_start", "YYYYMM_end"
            ]
        else:
            raise ValueError(f"Unknown filename template: {template}")
    else:
        template_keys = template

    filename_parts = []
    for key in template_keys:
        if key == "index":
            filename_parts.append(index_id)
        elif key == "compute_fq":
            filename_parts.append(compute_fq)
        else:
            value = meta.get(key, "unknown")
            logger.debug(f"Adding '{key}': '{value}' to filename")
            if value == "unknown":

                logger.warning(
                    f"Metadata key '{key}' is missing or unknown.",
                    stack_info=True
                )
            filename_parts.append(str(value))
    filename = "_".join(filename_parts) + ".nc"
    return filename

def write_index_netcdf(
    values:         np.ndarray,
    index:          type[BaseIndex],
    metadata:       dict[str, Any],
    output_path:    Path,
):
    """
    Write ETCCDI index results to a CF-compliant NetCDF file.

    Parameters
    ----------
    values :
        ndarray of shape (time, lat, lon) or (n_years, lat, lon)
    index :
        Subclass of BaseIndex, provides index_id, index_long_name, index_units, etc.
    metadata :
        Dict containing:
            - lat
            - lon
            - time_out (the output time axis for this frequency)
            - time_bnds (optional)
            - parent_global_attrs
            - child_global_attrs
            - parent_var_attrs
            - child_var_attrs
    output_path :
        Path object where file is written
    """

    # Unpack needed metadata
    lat = metadata["lat"]
    lon = metadata["lon"]
    time_out = metadata["time_out"]

    time_bnds = metadata.get("time_bnds")
    lat_bnds = metadata.get("lat_bnds")
    lon_bnds = metadata.get("lon_bnds")

    parent_global = metadata.get("parent_global_attrs", {})
    child_global = metadata.get("child_global_attrs", {})
    parent_var = metadata.get("parent_var_attrs", {})
    child_var = metadata.get("child_var_attrs", {})

    index_id = index.index_id
    long_name = index.index_long_name
    units = index.index_units

    # ------------------------------------------------------------------
    # Create dataset
    # ------------------------------------------------------------------
    with Dataset(output_path, "w") as ds:
        # Dimensions
        ds.createDimension("time", len(time_out))
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        if any(bnds is not None for bnds in [time_bnds, lat_bnds, lon_bnds]):
            ds.createDimension("bnds", 2)

        # ------------------------------------------------------------------
        # Coordinates
        # ------------------------------------------------------------------
        t = ds.createVariable("time", "f8", ("time",))
        t[:] = time_out
        if "time_units" in metadata:
            t.units = metadata["time_units"]
        if "calendar" in metadata:
            t.calendar = metadata["calendar"]

        if time_bnds is not None:
            tb = ds.createVariable("time_bnds", "f8", ("time", "bnds"))
            tb[:] = time_bnds
            t.bounds = "time_bnds"

        latv = ds.createVariable("lat", "f4", ("lat",))
        latv[:] = lat
        latv.standard_name = "latitude"
        latv.units = "degrees_north"

        # lat_bnds if available
        if lat_bnds is not None:
            latb = ds.createVariable("lat_bnds", "f8", ("lat", "bnds"))
            latb[:] = lat_bnds
            latv.bounds = "lat_bnds"

        lonv = ds.createVariable("lon", "f4", ("lon",))
        lonv[:] = lon
        lonv.standard_name = "longitude"
        lonv.units = "degrees_east"

        # lon_bnds if available
        if lon_bnds is not None:
            lonb = ds.createVariable("lon_bnds", "f8", ("lon", "bnds"))
            lonb[:] = lon_bnds
            lonv.bounds = "lon_bnds"

        # ------------------------------------------------------------------
        # Main variable
        # ------------------------------------------------------------------
        var = ds.createVariable(
            varname=index_id,
            datatype="f4",
            dimensions=("time", "lat", "lon"),
            zlib=True
        )
        var[:] = values
        var.long_name = long_name
        var.units = units

        # ------------------------------------------------------------------
        # Global attributes (parent first, then child overrides)
        # ------------------------------------------------------------------
        ncattrs = {}
        ncattrs.update(parent_global)
        ncattrs.update(child_global)  # ensure child attrs appear last
        for attr_name, attr_value in ncattrs.items():
            setattr(ds, attr_name, attr_value)

        # ------------------------------------------------------------------
        # Variable attributes (parent first, then child overrides)
        # ------------------------------------------------------------------
        varattrs = {}
        varattrs.update(parent_var)
        varattrs.update(child_var)
        for attr_name, attr_value in varattrs.items():
            setattr(var, attr_name, attr_value)
