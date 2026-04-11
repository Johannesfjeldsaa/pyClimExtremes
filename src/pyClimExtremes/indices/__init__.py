"""ETCCDI index implementations.

Importing this module ensures all index classes are registered in the
INDEX_REGISTRY for discovery and computation.
"""

# Import all index modules to trigger @register_index decorators
from .temperature_indices import *
from .precipitation_indices import *
from .quantile_indices import *

# Expose only core workflow functions
from .registry import (
    resolve_indices,
    resolve_frequencies,
    get_creatable_indices,
    get_creatable_quantiles
)

__all__ = [
    "resolve_indices",
    "resolve_frequencies",
    "get_creatable_indices",
    "get_creatable_quantiles"
]
