"""Utility functions for saving figures, datasets, and files safely.

This module provides utilities for saving figures, xarray datasets, and
information to files in a safe and structured way. It includes functions
to handle overwriting filepaths and variables in xarray datasets,
ensuring that existing files or variables are not unintentionally
overwritten unless explicitly allowed.

The module provides functionality to save figures and datasets in
specified formats (e.g., NetCDF for xarray datasets, YAML for
information).

Author: Johannes Fjeldså
"""
import os
import traceback
from pathlib import Path
from typing import Any, Union

import matplotlib.pyplot as plt
import xarray as xr

from general_backend.logging.setup_logging import get_logger

logger = get_logger(__name__)


def validate_file(
    file_path: str | Path,
    expected_suffix: str | list[str],
    description: str,
    new_file: bool,
) -> None:
    """Validate if a file has: 1) expected suffixes, 2) if it exists
    when new_file is False.

    Parameters
    ----------
    file_path : str | Path
        path to the file to validate.
    expected_suffix : str | list[str]
        expected file suffix or list of suffixes.
    description : str
        description of the file type for error messages.
    new_file : bool
        whether the file is expected to be new (True) or existing (False).

    Raises
    ------
    SystemExit
        Raised if the file suffix is not as expected.
    SystemExit
        Raised if the file does not exist when new_file is False.
    """
    if isinstance(expected_suffix, str):
        expected_suffix = [expected_suffix]
    if not isinstance(file_path, Path):
        file_path = Path(file_path).resolve()

    if file_path.suffix not in expected_suffix:
        raise SystemExit(f"ERROR: {file_path} is not a valid {description}")
    else:
        if not new_file and not file_path.exists():
            traceback.print_stack()
            raise SystemExit(
                f"ERROR: {file_path} does not exist. "
                "Please provide a valid file path."
            )


def check_filepath(
    file_path: Union[str, Path],
    overwrite: bool | str,
) -> bool:
    """Check how to handle the file path for saving content.

    Parameters
    ----------
    file_path : Union[str, Path]
        The path where the file will be saved.
    overwrite : bool or "prompt"
        - True  → always overwrite
        - False → never overwrite
        - "prompt" → ask the user interactively

    Returns
    -------
    bool
        True if the file can be saved (either it doesn't exist or
        overwrite is allowed), False if the file already exists
        and overwrite is not allowed.
    """

    file_path = Path(file_path).resolve()

    # default write to True, if the file exists, check if we can overwrite it
    write = True
    if file_path.exists():
        prompt_msg = (
            f"File {file_path.name} already exists at {file_path.parent}."
        )
        positive_msg = (
            f"File {file_path.name} will be overwritten at {file_path.parent}."
        )
        negative_msg = (
            f"File {file_path.name} already exists at {file_path.parent}. "
            "To overwrite change the file name or set overwrite to True."
        )
        write = overwrite_handler(
            overwrite,
            prompt_msg=prompt_msg,
            positive_msg=positive_msg,
            negative_msg=negative_msg,
        )
        if write:
            logger.debug(f"Overwriting existing file: {file_path}")
            os.remove(file_path)
    if write and not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Created directory: %s to save %s",
            file_path.parent,
            file_path.name,
        )

    return write


def check_variable_overwrite(
    ds: xr.Dataset, var_name: str, overwrite: bool | str
) -> bool:
    """Check if a variable can be saved in an xarray dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray dataset where the variable will be saved.
    var_name : str
        The name of the variable to be saved.
    overwrite : bool or "prompt"
        - True  → always overwrite
        - False → never overwrite
        - "prompt" → ask the user interactively

    Returns
    -------
    bool
        True if the variable can be saved (either it doesn't exist or
        overwrite is allowed), False if the variable already exists
        and overwrite is not allowed.
    """

    # default write to True, if the variable exists, check if we can overwrite
    write = True
    if var_name in ds.data_vars:
        prompt_msg = f"Variable {var_name} already exists in the dataset."
        positive_msg = (
            f"Variable {var_name} will be overwritten in the dataset."
        )
        negative_msg = (
            f"Variable {var_name} already exists in the dataset. "
            f"If you want to overwrite set overwrite to True."
        )
        write = overwrite_handler(
            overwrite,
            prompt_msg=prompt_msg,
            positive_msg=positive_msg,
            negative_msg=negative_msg,
        )
    return write


def overwrite_handler(
    overwrite: bool | str,
    prompt_msg: str,
    positive_msg: str,
    negative_msg: str,
) -> bool:
    """Decide whether to overwrite based on input value or prompt.

    Parameters
    ----------
    overwrite : bool or "prompt"
        - True  → always overwrite
        - False → never overwrite
        - "prompt" → ask the user interactively
    prompt_msg : str
        Context message before prompting (if overwrite="prompt").
    positive_msg : str
        Message to log if overwriting is allowed.
    negative_msg : str
        Message to log if overwriting is not allowed.
    force_msg : str
        Message to log if the user chooses 'force' in prompt mode.
    skip_msg : str
        Message to log if the user chooses 'skip' in prompt mode.

    Returns
    -------
    bool
        True if overwrite is allowed, False otherwise.
    """
    if overwrite is True:
        logger.debug(positive_msg)
        return True
    elif overwrite is False:
        logger.debug(negative_msg)
        return False
    elif overwrite == "prompt":
        response = input(f"{prompt_msg}\nOverwrite? (y/n): ").strip().lower()

        if response == "y":
            logger.info(positive_msg)
            return True
        else:
            logger.info(negative_msg)
            return False
    else:
        err_msg = (
            f"Invalid value for overwrite: {overwrite}.\n"
            "Expected True, False, or 'prompt'."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)


def save_figure(
    fig: Any, file_path: Union[str, Path], overwrite: bool | str
) -> bool:
    """Save a matplotlib figure to a specified file path.

    Parameters
    ----------
    fig : Any
        The matplotlib figure to save.
    file_path : Union[str, Path]
        The path where the figure will be saved.
    general_settings : dict
        General settings from config.general_settings.

    Returns
    -------
    bool
        True if the figure was saved, False otherwise.
    """
    if check_filepath(file_path, overwrite):
        fig.savefig(file_path, dpi=300)
        logger.info("Figure saved to %s", file_path)
        plt.close(fig)
        return True
    return False


def save_xarray_to_netcdf(
    dataset: xr.Dataset, file_path: Union[str, Path], overwrite: bool | str
) -> bool:
    """Save a xarray dataset to NetCDF at a specified file path.

    Parameters
    ----------
    dataset : xr.Dataset
        The xarray dataset to save.
    file_path : Union[str, Path]
        The path where the dataset will be saved.
    overwrite : bool or "prompt"
        - True  → always overwrite
        - False → never overwrite
        - "prompt" → ask the user interactively

    Returns
    -------
    bool
        True if the dataset was saved, False otherwise.
    """
    if check_filepath(file_path, overwrite):
        dataset.to_netcdf(file_path)
        logger.info("NetCDF dataset saved to %s", file_path)
        return True
    return False
