"""ETCCDI index implementations.

Importing this module ensures all index classes are registered in the
INDEX_REGISTRY for discovery and computation.
"""

# Import all index modules to trigger @register_index decorators
from .python_backend import *

# Expose registry utilities
from .backend_registry import (
    BACKEND_REGISTRY,
    get_compute_backend,
)

__all__ = [
    "BACKEND_REGISTRY",
    "get_compute_backend",
]
