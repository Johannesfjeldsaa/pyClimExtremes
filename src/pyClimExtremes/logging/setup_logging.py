"""Logging configuration utilities for the pyClimExtremes package.

This module provides logging utilities designed for use in dependency packages.
The package respects the main project's logging configuration by default,
allowing logs to propagate to the application's root logger.

Functions
---------
get_logger : function
    Get a package-scoped logger that respects parent application logging.
    Usage: logger = get_logger(__name__)

set_logger_level_for_dependency : function
    Set logging level for a specific dependency package.
    Usage: set_logger_level_for_dependency(
        'some_dependency', logging.ERROR
    )

configure_standalone_logging : function
    Configure logging when using the package standalone (not as dependency).
    Usage: configure_standalone_logging(
        pckg_level=logging.INFO, root_level=logging.WARNING
    )

configure_package_logger : function
    Configure only the package logger without affecting root logger.
    Usage: configure_package_logger(level=logging.INFO)

Notes
-----
By default, the package uses a dependency-friendly approach:
- Logs propagate to parent loggers (respects main project configuration)
- No automatic root logger configuration
- Minimal console output from the package itself

For standalone use, call configure_standalone_logging() to set up
full logging configuration.

Authors
Johannes Fjeldså
"""

import logging

from pyClimExtremes import PACKAGE_LOGGER_NAME


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a package-scoped logger.
    Example: get_logger('module') -> 'pyClimExtremes.module'

    Parameters
    ----------
    name : str | None, optional
        name for logger object.

    Returns
    -------
    logging.Logger : logging.Logger
        Logger object with the name.
    """
    name = name or ""
    if name.startswith(PACKAGE_LOGGER_NAME):
        logger_name = name
    else:
        logger_name = PACKAGE_LOGGER_NAME + f".{name}"

    return logging.getLogger(logger_name)


def set_logger_level_for_dependency(dependency_name: str, level: int) -> None:
    """Set the logging level for a specific dependency package.

    Parameters
    ----------
    dependency_name : str
        Name of the dependency package (e.g., 'some_dependency').
    level : int
        Logging level to set for the dependency package (e.g., logging.ERROR).
    """
    logger = logging.getLogger(dependency_name)
    logger.setLevel(level)


def configure_package_logger(
    level: int = logging.INFO,
    propagate: bool = True,
    add_handler: bool = False,
    fmt: str | None = None,
) -> None:
    """Configure the package logger without affecting the root logger.

    This function is dependency-friendly - it only configures the package's
    own logger and respects the main application's logging setup.

    Parameters
    ----------
    level : int, optional
        Logging level for the package logger, by default logging.INFO
    propagate : bool, optional
        Whether package logs should propagate to parent loggers,
        by default True (dependency-friendly)
    add_handler : bool, optional
        Whether to add a handler to the package logger,
        by default False (lets parent handle output)
    fmt : str | None, optional
        Logging format string. Only used if add_handler=True.
        If None, uses default format.
    """
    pkg_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    pkg_logger.setLevel(level)
    pkg_logger.propagate = propagate

    if add_handler:
        # Remove existing StreamHandlers to avoid duplicates
        for h in list(pkg_logger.handlers):
            if isinstance(h, logging.StreamHandler):
                pkg_logger.removeHandler(h)

        handler = logging.StreamHandler()
        fmt_to_use = (
            fmt
            if fmt is not None
            else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(logging.Formatter(fmt_to_use))
        handler.setLevel(level)
        pkg_logger.addHandler(handler)


STANDALONE_LOG_CONFIG_MSG = """
Configuring standalone logging for pyClimExtremes package:
* Package logger level: {pckg_level}
* Root logger level: {root_level}
To change the logging level for a specific dependency, use
set_logger_level_for_dependency importable from
pyClimExtremes.logging.setup_logging.
"""


def configure_standalone_logging(
    pckg_level: int = logging.INFO,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    root_level: int = logging.WARNING,
    suppress_log_config_msg: bool = False,
) -> None:
    """Configure logging for standalone use (not as a dependency).

    This function configures both the root logger and package logger.
    Use this when the package is the main application, not when it's
    used as a dependency.

    Parameters
    ----------
    pckg_level : int, optional
        Logging level for the pyClimExtremes package logger,
        by default logging.INFO
    fmt : str, optional
        Logging format string,
        by default '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    root_level : int, optional
        Logging level for the root logger, by default logging.WARNING
    suppress_log_config_msg: bool, optional
        Whether to suppress the logging configuration message, by default False
    """
    # Configure the root logger centrally. Using basicConfig is the
    # simplest way to ensure the root handler/level are set consistently.
    logging.basicConfig(level=root_level, format=fmt)

    # Ensure the root logger's level is explicitly set (basicConfig might
    # not change it if handlers already existed in some environments).
    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)

    # Configure the package logger with a handler and no propagation
    # (since we want independent control in standalone mode)
    configure_package_logger(
        level=pckg_level,
        propagate=False,  # Don't propagate in standalone mode
        add_handler=True,  # Add our own handler
        fmt=fmt,
    )

    if not suppress_log_config_msg:
        print(
            STANDALONE_LOG_CONFIG_MSG.format(
                pckg_level=pckg_level, root_level=root_level
            )
        )
