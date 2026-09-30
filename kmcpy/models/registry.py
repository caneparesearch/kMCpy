"""Model types for model files and resolution of serialized model classes."""

from __future__ import annotations

import importlib
from typing import Any

# Maps ``model_type`` names to fully-qualified model class paths.
MODEL_CLASS_REGISTRY = {
    "composite_lce": "kmcpy.models.composite_lce_model.CompositeLCEModel",
    "lce": "kmcpy.models.local_cluster_expansion.LocalClusterExpansion",
    "local_cluster_expansion": "kmcpy.models.local_cluster_expansion.LocalClusterExpansion",
    "local_barrier": "kmcpy.models.local_barrier_model.LocalBarrierModel",
    "site_energy": "kmcpy.models.site_energy.SiteEnergyModel",
}


def model_class_for_type(model_type: str) -> type:
    """Return the model class registered for ``model_type``."""
    if model_type not in MODEL_CLASS_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available types: {list(MODEL_CLASS_REGISTRY)}"
        )
    class_path = MODEL_CLASS_REGISTRY[model_type]
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
        for model_type, class_path in MODEL_CLASS_REGISTRY.items():
            if class_path.rsplit(".", 1)[1] == class_name:
                return model_class_for_type(model_type)
        raise ValueError(f"Cannot import model class '{module_path}.{class_name}': {exc}")
