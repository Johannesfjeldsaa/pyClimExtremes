# Package-level logger setup:
# we just provide a logger and a NullHandler
import logging

# Use a constant package name so logs are consistently tagged
PACKAGE_LOGGER_NAME = "pyClimExtremes"
logger = logging.getLogger(PACKAGE_LOGGER_NAME)

# Defensive logging setup - add a NullHandler to:
# * avoid "No handler found" warnings if the application using the package
#   has not configured logging.
# * allow the package logger to propagate messages to the root logger
#   if the application has configured logging.
logger.addHandler(logging.NullHandler())

# Export only the essentials for external use
__all__ = ["logger", "PACKAGE_LOGGER_NAME"]
