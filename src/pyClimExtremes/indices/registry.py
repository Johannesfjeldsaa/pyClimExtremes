import numpy as np

from pyClimExtremes.logging.setup_logging import get_logger

logger = get_logger(__name__)


INPUT_VAR_ALIASES = {
    "tasmax": ["tasmax", "tx"],
    "tasmin": ["tasmin", "tn"],
    "pr": ["pr", "precip", "prcp", 'prect'],
    "tas": ["tas", "tavg"],
}

def input_var_str_normalize(name: str) -> str:
    """Normalize input variable names to their canonical IDs.

    Maps aliases (e.g., 'tx' -> 'tasmax') using INPUT_VAR_ALIASES.
    Returns the canonical variable name if found; otherwise returns
    the original name unchanged.
    """
    if not isinstance(name, str):
        return name
    lower = name.strip().lower()
    for canonical, aliases in INPUT_VAR_ALIASES.items():
        if lower in [alias.lower() for alias in aliases]:
            return canonical
    return name

INDEX_REGISTRY = {}
TEMPERATURE_INDEX_REGISTRY, PRECIPITATION_INDEX_REGISTRY = {}, {}
QUANTILE_REGISTRY = {}
TEMPERATURE_QUANTILE_REGISTRY, PRECIPITATION_QUANTILE_REGISTRY = {}, {}

def register_index(cls):
    """Developers' decorator to add the index class to the global registry.
    Should be applied to all index classes that should be computable
    for users.
    """
    if cls.index_id is None:
        raise ValueError(f"{cls.__name__} must define index_id.")

    if cls.index_type.endswith("quantile"):
        is_generic_template = getattr(cls, "is_generic_template", False)
        has_valid_quantile = (
            hasattr(cls, 'quantile') and
            cls.quantile is not None and
            not getattr(np, "isnan", lambda x: False)(cls.quantile)
        )
        if not has_valid_quantile and not is_generic_template:
            err_msg = (
                f"{cls.__name__} has index_type '{cls.index_type}' but does not define a valid 'quantile' attribute. "
                "Quantile indices must have a 'quantile' attribute defined as a float between 0 and 1."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        QUANTILE_REGISTRY[cls.index_id] = cls
        register_msg = f"Registered quantile '{cls.index_id}'"
    else:
        INDEX_REGISTRY[cls.index_id] = cls
        register_msg = f"Registered impact index '{cls.index_id}'"

    unrecognized_type = False
    if cls.index_type == "temperature":
        TEMPERATURE_INDEX_REGISTRY[cls.index_id] = cls
    elif cls.index_type == "precipitation":
        PRECIPITATION_INDEX_REGISTRY[cls.index_id] = cls
    elif cls.index_type == "temperature_quantile":
        TEMPERATURE_QUANTILE_REGISTRY[cls.index_id] = cls
    elif cls.index_type == "precipitation_quantile":
        PRECIPITATION_QUANTILE_REGISTRY[cls.index_id] = cls
    else:
        unrecognized_type = True
        logger.warning(
            "Index '%s' has unrecognized index_type '%s'. "
            "It will not be added to specific type registries.",
            cls.index_id, cls.index_type
        )
    if not unrecognized_type:
        logger.debug(
            register_msg + " of type '%s' with ID '%s'.",
            cls.index_type, cls.index_id
        )
    return cls


def get_creatable_indices(
    subset:     str = "all",
    print_msg:  bool = False,
    log_msg:    bool = False
) -> dict:
    """Return a dictionary of creatable index IDs to their long names.

    Parameters
    ----------
    subset : str, optional
        If 'all', return all creatable indices.
        If 'temperature', return only temperature indices.
        If 'precipitation', return only precipitation indices.
        by default "all"
    print_msg : bool, optional
        If True, print the available indices, by default False.
    log_msg : bool, optional
        If True, log the available indices, by default False.
    """
    if subset == "temperature":
        registry = TEMPERATURE_INDEX_REGISTRY
    elif subset == "precipitation":
        registry = PRECIPITATION_INDEX_REGISTRY
    else:
        registry = INDEX_REGISTRY
    if print_msg or log_msg:
        logg_msg = (
            "Available creatable indices:\n" if subset == "all" else
            f"Available creatable {subset} indices:\n"
        )

        # Collect indices that require custom threshold
        requires_threshold = []

        for idx_id, cls in registry.items():
            long_name = cls.index_long_name
            # Check if this index requires a custom threshold
            if (
                hasattr(cls, 'default_threshold') and
                cls.default_threshold is None
            ):
                logg_msg += f" - {idx_id}: {long_name} *,\n"
                requires_threshold.append(idx_id)
            else:
                logg_msg += f" - {idx_id}: {long_name},\n"

        # Add note if any indices require threshold
        if requires_threshold:
            logg_msg += "\n* custom threshold X must be provided. Provide through kwargs = {'threshold': {index_ID: X}}.\n"

        if print_msg:
            print(logg_msg)
        if log_msg:
            logger.info(logg_msg)

    return {
        idx_id: cls.index_long_name for idx_id, cls in registry.items()
    }

def get_creatable_quantiles(
    subset:     str = "all",
    print_msg:  bool = False,
    log_msg:    bool = False
) -> dict:
    """Return a dictionary of creatable quantile index IDs to their long names.

    Parameters
    ----------
    subset : str, optional
        If 'all', return all creatable quantiles.
        If 'temperature', return only temperature quantiles.
        If 'precipitation', return only precipitation quantiles.
        by default "all"
    print_msg : bool, optional
        If True, print the available quantiles, by default False.
    log_msg : bool, optional
        If True, log the available quantiles, by default False.
    """
    if subset == "temperature":
        registry = TEMPERATURE_QUANTILE_REGISTRY
    elif subset == "precipitation":
        registry = PRECIPITATION_QUANTILE_REGISTRY
    else:
        registry = QUANTILE_REGISTRY
    creatable_registry = {
        idx_id: cls for idx_id, cls in registry.items()
        if getattr(cls, "is_generic_template", False)
    }
    if not creatable_registry:
        creatable_registry = registry
    if print_msg or log_msg:
        logg_msg = (
            "Available creatable quantiles:\n" if subset == "all" else
            f"Available creatable {subset} quantiles:\n"
        )

        requires_quantile = []

        for idx_id, cls in creatable_registry.items():
            long_name = cls.index_long_name
            if getattr(cls, "is_generic_template", False):
                logg_msg += f" - {idx_id}: {long_name} *,\n"
                requires_quantile.append(idx_id)
            else:
                logg_msg += f" - {idx_id}: {long_name},\n"

        if requires_quantile:
            logg_msg += (
                "\n* custom quantile values must be provided through "
                "compute_thresholds(quantiles={index_id: [q1, q2, ...]}).\n"
            )

        if print_msg:
            print(logg_msg)
        if log_msg:
            logger.info(logg_msg)

    return {
        idx_id: cls.index_long_name for idx_id, cls in creatable_registry.items()
    }

def resolve_indices(
    indices: str | list[str],
    impact_or_quantile: str = "impact"
):
    """Check and resolve the list of indices to compute.

    Parameters
    ----------
    indices : str | list[str]
        Which indices to request. 'all' for all creatable indices, else
        provide a single index ID or a list of index IDs.

    Raises
    ------
    ValueError
        If any requested index ID is not creatable / defined yet in the
        registry.
    """
    if impact_or_quantile not in ["impact", "quantile"]:
        err_msg = (
            f"Invalid value for impact_or_quantile: '{impact_or_quantile}'. "
            "Expected 'impact' or 'quantile'."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    if impact_or_quantile == "impact":
        registry = INDEX_REGISTRY
        temp_registry = TEMPERATURE_INDEX_REGISTRY
        precip_registry = PRECIPITATION_INDEX_REGISTRY
    else:
        registry = QUANTILE_REGISTRY
        temp_registry = TEMPERATURE_QUANTILE_REGISTRY
        precip_registry = PRECIPITATION_QUANTILE_REGISTRY

    creatable_indices = sorted(list(registry.keys()))
    if indices == "all":
        index_list = list(registry.values())
    elif indices == "temperature":
        index_list = list(temp_registry.values())
    elif indices == "precipitation":
        index_list = list(precip_registry.values())
    else:
        index_list = []
        if not isinstance(indices, list):
            indices = [indices]
        for index_id in indices:
            if index_id not in creatable_indices:
                available_list = [f"{idx_id}" for idx_id in creatable_indices]
                err_msg = (
                    f"Index '{index_id}' is not among the creatable indices.\n"
                    f"Available index IDs are:\n"
                    + "\n".join([f"  • {item}" for item in available_list])
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
            index_list.append(registry[index_id])
    logger.debug(f"Resolved indices to compute: {index_list}")
    return index_list


def resolve_frequencies(
    frequencies: str | list[str],
):
    """Check and resolve the list of frequencies to compute.

    Parameters
    ----------
    frequencies : str | list[str]
        Which frequencies to request. 'all' for all supported frequencies,
        else provide a single frequency or a list of frequencies.

    Raises
    ------
    ValueError
        If any requested frequency is not supported.
    """
    supported_frequencies = ["mon", "yr"]
    if frequencies == "all":
        frequency_list = supported_frequencies
    else:
        frequency_list = []
        if not isinstance(frequencies, list):
            frequencies = [frequencies]
        for fq in frequencies:
            if fq not in supported_frequencies:
                err_msg = (
                    f"Frequency '{fq}' is not supported."
                    f" Supported frequencies are: {supported_frequencies}"
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
            frequency_list.append(fq)
    logger.debug(f"Resolved frequencies to compute: {frequency_list}")
    return frequency_list