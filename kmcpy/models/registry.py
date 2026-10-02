"""Model types for model files and resolution of serialized model classes.

Built-in model types are listed in ``MODEL_CLASS_REGISTRY``. Register your
own model class with :func:`register_model` so that model files, YAML inputs,
and ``BaseModel.load`` can refer to it by name::

    @register_model("my_model")
    class MyModel(BaseModel):
        ...
"""

from __future__ import annotations

import importlib
from typing import Any

# Maps ``model_type`` names to model classes or fully-qualified class paths
# (paths are imported on first use, which avoids import cycles).
MODEL_CLASS_REGISTRY: dict[str, type | str] = {
    "composite_lce": "kmcpy.models.composite_lce_model.CompositeLCEModel",
    "lce": "kmcpy.models.local_cluster_expansion.LocalClusterExpansion",
    "local_cluster_expansion": "kmcpy.models.local_cluster_expansion.LocalClusterExpansion",
    "local_barrier": "kmcpy.models.local_barrier_model.LocalBarrierModel",
    "site_energy": "kmcpy.models.site_energy.SiteEnergyModel",
}


def register_model(model_type: str, *, replace: bool = False):
    """Class decorator that registers a model class under ``model_type``.

    Registering a different class under an existing name raises
    ``ValueError`` unless ``replace=True``.
    """

    def decorator(model_class: type) -> type:
        existing = MODEL_CLASS_REGISTRY.get(model_type)
        if existing is not None and not replace and _class_path(existing) != _class_path(model_class):
            raise ValueError(
                f"Model type '{model_type}' is already registered to "
                f"{_class_path(existing)}; pass replace=True to override it."
            )
        MODEL_CLASS_REGISTRY[model_type] = model_class
        return model_class

    return decorator


def _class_path(entry: type | str) -> str:
    if isinstance(entry, str):
        return entry
    return f"{entry.__module__}.{entry.__qualname__}"


def model_class_for_type(model_type: str) -> type:
    """Return the model class registered for ``model_type``."""
    if model_type not in MODEL_CLASS_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available types: {list(MODEL_CLASS_REGISTRY)}"
        )
    class_path = MODEL_CLASS_REGISTRY[model_type]
    if not isinstance(class_path, str):
        return class_path
    module_path, class_name = class_path.rsplit(".", 1)
    try:
        return getattr(importlib.import_module(module_path), class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot import model class '{class_path}': {exc}")


def model_class_for_payload(payload: dict[str, Any]) -> type:
    """Return the class named by a payload's ``@module`` and ``@class``.

    If that import fails (for example, the module moved between kMCpy
    versions), a registered model class with the same class name is used.
    """
    module_path = payload["@module"]
    class_name = payload["@class"]
    try:
        return getattr(importlib.import_module(module_path), class_name)
    except (ImportError, AttributeError) as exc:
        for model_type, entry in MODEL_CLASS_REGISTRY.items():
            if _class_path(entry).rsplit(".", 1)[1] == class_name:
                return model_class_for_type(model_type)
        raise ValueError(f"Cannot import model class '{module_path}.{class_name}': {exc}")
