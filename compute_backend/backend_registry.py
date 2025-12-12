"""Registry for compute backends, allowing selection of calculation modules.
Each backend should implement the necessary ETCCDI index calculations.
"""

BACKEND_REGISTRY = {}

def register_backend(name):
    def wrapper(instance):
        BACKEND_REGISTRY[name] = instance
        return instance

    return wrapper

def get_compute_backend(name):
    backend_cls = BACKEND_REGISTRY[name]
    return backend_cls()
