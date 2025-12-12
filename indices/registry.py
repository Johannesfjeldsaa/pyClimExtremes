from general_backend.logging.setup_logging import get_logger

logger = get_logger(__name__)

INPUT_VAR_ALIASES = {
    "tasmax": ["tasmax", "tx"],
    "tasmin": ["tasmin", "tn"],
    "pr": ["pr", "precip", "prcp", 'prect'],
    "tas": ["tas", "tavg"],
}
INDEX_REGISTRY = {}
INDEX_REGISTRY_ALIASES = {}
INDEX_ALIAS_MAP = {}

def register_index(cls):
    """Developers' decorator to add the index class to the global registry.
    Should be applied to all index classes that should be computable
    for users.
    """
    if cls.index_id is None:
        raise ValueError(f"{cls.__name__} must define index_id.")
    INDEX_REGISTRY[cls.index_id] = cls

    index_aliases = (
        cls.index_aliases if hasattr(cls, "index_aliases") else
        [cls.index_id]
    )
    INDEX_ALIAS_MAP[cls.index_id] = index_aliases
    for alias in index_aliases:
        INDEX_REGISTRY_ALIASES[alias] = cls
    logger.debug(
        "Registered index '%s' with aliases %s",
        cls.index_id, index_aliases
    )
    return cls


def get_creatable_indices(
    print_msg:  bool = False,
    log_msg:    bool = False
) -> dict:
    """Return a dictionary of creatable index IDs to their long names.

    Parameters
    ----------
    print_msg : bool, optional
        If True, print the available indices, by default False.
    log_msg : bool, optional
        If True, log the available indices, by default False.
    """
    if print_msg or log_msg:
        logg_msg = "Available creatable indices:\n"

        for idx_id, cls in INDEX_REGISTRY.items():
            logg_msg += f" - {idx_id}: {cls.index_long_name}, \n"
        if print_msg:
            print(logg_msg)
        if log_msg:
            logger.info(logg_msg)

    return {
        idx_id: cls.index_long_name for idx_id, cls in INDEX_REGISTRY.items()
    }

def resolve_indices(
    indices: str | list[str],
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
    creatable_indices = list(INDEX_REGISTRY.keys())
    creatable_indices.sort()
    creatable_indices_aliases = list(INDEX_REGISTRY_ALIASES.keys())
    creatable_indices_aliases.sort()
    if indices == "all":
        index_list = list(INDEX_REGISTRY.values())
    else:
        index_list = []
        if not isinstance(indices, list):
            indices = [indices]
        for index_id in indices:
            if index_id not in creatable_indices_aliases:
                # Build formatted list of available indices and aliases
                available_list = []
                for idx_id, aliases in INDEX_ALIAS_MAP.items():
                    available_list.append(f"{idx_id}: {', '.join(aliases)}")
                
                err_msg = (
                    f"Index '{index_id}' is not among the creatable indices.\n"
                    f"Available indices and accepted aliases are:\n"
                    + "\n".join([f"  • {item}" for item in available_list])
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
            # Use alias-aware lookup
            index_list.append(INDEX_REGISTRY_ALIASES[index_id])
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
    supported_frequencies = ["mon", "ann"]
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