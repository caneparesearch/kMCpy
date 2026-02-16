"""Registry utilities for mapping model classes to fitter classes."""

from __future__ import annotations

from typing import Type

_FITTER_REGISTRY: dict[type, type] = {}


def register_fitter(model_class: Type, fitter_class: Type) -> None:
    """Register a fitter implementation for a model class."""
    if model_class in _FITTER_REGISTRY:
        existing = _FITTER_REGISTRY[model_class]
        if existing is not fitter_class:
            raise ValueError(
                f"Model {model_class.__name__} already has fitter "
                f"{existing.__name__} registered."
            )
    _FITTER_REGISTRY[model_class] = fitter_class


def get_fitter_for_model(model_class: Type) -> Type | None:
    """Look up a fitter for a model class using MRO fallback."""
    for cls in model_class.__mro__:
        fitter_class = _FITTER_REGISTRY.get(cls)
        if fitter_class is not None:
            return fitter_class
    return None
