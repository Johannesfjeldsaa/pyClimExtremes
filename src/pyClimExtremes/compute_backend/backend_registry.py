"""Registry for compute backends, allowing selection of calculation modules.
Each backend should implement the necessary ETCCDI index calculations.
"""

from typing import Callable
BACKEND_REGISTRY = {}


def register_backend(name: str) -> Callable:
    def wrapper(instance: object) -> object:
        BACKEND_REGISTRY[name] = instance
        return instance

    return wrapper


def get_compute_backend(name: str, **kwargs) -> object:
    backend_cls = BACKEND_REGISTRY[name]
    return backend_cls(**kwargs)
