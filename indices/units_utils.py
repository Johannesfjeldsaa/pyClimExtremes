import numpy as np

from general_backend.logging.setup_logging import get_logger
from .registry import input_var_str_normalize

logger = get_logger(__name__)


def unit_str_normalize(unit: str) -> str:
    """Normalize unit strings for comparison.

    Converts to lowercase, removes spaces and underscores for consistent
    comparison. Preserves hyphens.

    Parameters
    ----------
    unit : str
        Input unit string (e.g., 'K', 'deg_C', 'kg m-2 s-1', 'mm day-1').
    Returns
    -------
    str
        Normalized unit string (e.g., 'k', 'degc', 'kgm-2s-1', 'mmday-1').
    """
    return unit.strip().lower().replace("_", "").replace(" ", "")

temp_unit_aliases = ["K", "degC", "degreesC", "C"]
tua_normalized = [unit_str_normalize(u) for u in temp_unit_aliases]
precip_unit_aliases = ["kg m-2 s-1", "mm day-1", "mm/day", "mm d-1"]
pua_normalized = [unit_str_normalize(u) for u in precip_unit_aliases]
INPUT_VAR_ALLOWED_INPUT_UNITS = {
    "tasmax": tua_normalized,
    "tasmin": tua_normalized,
    "tas": tua_normalized,
    "pr": pua_normalized
}

def validate_input_units(
    input_var: str,
    input_unit: str
) -> bool:
    """Validate if the input unit for a variable is acceptable. Inteded usage
    is to check if the input data array's unit is valid before performing
    index computations.

    Parameters
    ----------
    input_var : str
        Input variable name (e.g., 'tasmax', 'pr'). Aliases like 'tx' or
        'tavg' are accepted and normalized to canonical variable names.
    input_unit : str
        Input unit string to validate.

    Returns
    -------
    bool
        True if the input unit is valid for the variable, False otherwise.
    """
    canonical_var = input_var_str_normalize(input_var)
    if canonical_var not in INPUT_VAR_ALLOWED_INPUT_UNITS:
        err_msg = (
            f"Variable '{input_var}' (normalized to '{canonical_var}') "
            "is not recognized for unit validation."
        )
        logger.error(err_msg, stack_info=True)
        raise ValueError(err_msg)

    normalized_input_unit = unit_str_normalize(input_unit)
    allowed_units = INPUT_VAR_ALLOWED_INPUT_UNITS[canonical_var]

    if normalized_input_unit in allowed_units:
        return True
    else:
        logger.error(
            "Input unit '%s' for variable '%s' is not among the allowed "
            "units: %s",
            input_unit, input_var, allowed_units
        )
        return False



def _kgm2s1_to_mmday1(precip_kgm2s1: float | np.ndarray) -> float | np.ndarray:
    return precip_kgm2s1 * 86400.0

def _mmday1_to_kgm2s1(precip_mmday1: float | np.ndarray) -> float | np.ndarray:
    return precip_mmday1 * (1./86400.0)

def _k_to_degc(temp_k: float | np.ndarray) -> float | np.ndarray:
    """Convert temperature from Kelvin to Celsius."""
    return temp_k - 273.15


def _degc_to_k(temp_degc: float | np.ndarray) -> float | np.ndarray:
    """Convert temperature from Celsius to Kelvin."""
    return temp_degc + 273.15


def convert_units(
    values: np.ndarray,
    from_unit: str,
    to_unit: str,
) -> np.ndarray | float:
    """Convert array values from one unit to another.

    Supports temperature (K, deg_C) and precipitation (mm d-1, kg m-2 s-1)
    conversions.

    Parameters
    ----------
    values : np.ndarray
        Array of values to convert.
    from_unit : str
        Source unit string (e.g., 'K', 'deg_C', 'mm d-1', 'kg m-2 s-1').
    to_unit : str
        Target unit string.

    Returns
    -------
    np.ndarray | float
        Converted array or scalar.

    Raises
    ------
    ValueError
        If conversion between from_unit and to_unit is not supported.
    """
    from_norm = unit_str_normalize(from_unit)
    to_norm = unit_str_normalize(to_unit)

    # If already the same, no conversion needed
    if from_norm == to_norm:
        return values

    # Temperature conversions
    if from_norm == "k" and to_norm == "degc":
        return _k_to_degc(values)
    elif from_norm == "degc" and to_norm == "k":
        return _degc_to_k(values)

    # Precipitation conversions
    elif from_norm == "kgm-2s-1" and to_norm == "mmday-1":
        return _kgm2s1_to_mmday1(values)
    elif from_norm == "mmday-1" and to_norm == "kgm-2s-1":
        return _mmday1_to_kgm2s1(values)
    # Accumulated precipitation conversions (no flux conversions due to lack of time unit awareness)
    elif from_norm == "kgm-2" and to_norm == "mm":
        return _kgm2s1_to_mmday1(values)
    elif from_norm == "mm" and to_norm == "kgm-2":
        return _mmday1_to_kgm2s1(values)

    else:
        err_msg = (
            f"Conversion from '{from_unit}' to '{to_unit}' is not supported."
        )
        logger.error(err_msg, stack_info=True)
        raise ValueError(err_msg)
