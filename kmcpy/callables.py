"""Helpers for user-supplied callables: import references and keyword support.

KMC property callbacks and ``SiteEnergyModel`` compute/apply functions can be
given as ``"package.module:function"`` strings and may accept only a subset of
the keyword arguments kMCpy offers. The model hooks use the same keyword checks
to stay compatible with older hook signatures.
"""

from __future__ import annotations

import functools
import importlib
import inspect
from typing import Any, Callable, Mapping


def resolve_callable_reference(callable_ref: str) -> Callable[..., Any]:
    """Resolve ``package.module:function`` or ``package.module.function``."""
    if ":" in callable_ref:
        module_path, attr_path = callable_ref.split(":", 1)
    else:
        module_path, _, attr_path = callable_ref.rpartition(".")
    if not module_path or not attr_path:
        raise ValueError(
            f"Invalid callable reference '{callable_ref}'. Use "
            "'package.module:function' or 'package.module.function'."
        )
    obj: Any = importlib.import_module(module_path)
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    if not callable(obj):
        raise TypeError(f"Resolved object '{callable_ref}' is not callable")
    return obj


def accepts_keyword(func: Callable[..., Any], keyword: str) -> bool:
    """Return whether ``func`` can be called with ``keyword=...``.

    Returns ``False`` when the signature cannot be inspected.
    """
    parameters = _signature_parameters(func)
    if parameters is None:
        return False
    return keyword in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def supported_keyword_names(func: Callable[..., Any]) -> frozenset[str] | None:
    """Return the keyword names ``func`` accepts, or ``None`` if it accepts any.

    ``None`` is also returned when the signature cannot be inspected, so callers
    fall back to passing every keyword. Results are cached per callable because
    this runs for every event-rate evaluation.
    """
    try:
        return _cached_supported_keyword_names(func)
    except TypeError:
        # Unhashable callable objects cannot be cached.
        return _supported_keyword_names(func)


def call_with_supported_keywords(
    func: Callable[..., Any], kwargs: Mapping[str, Any]
) -> Any:
    """Call ``func`` with only the entries of ``kwargs`` it accepts."""
    accepted = supported_keyword_names(func)
    if accepted is None:
        return func(**kwargs)
    return func(**{key: value for key, value in kwargs.items() if key in accepted})


def _signature_parameters(func):
    try:
        return inspect.signature(func).parameters
    except (TypeError, ValueError):
        return None


def _supported_keyword_names(func) -> frozenset[str] | None:
    parameters = _signature_parameters(func)
    if parameters is None or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return None
    return frozenset(
        name
        for name, parameter in parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )


_cached_supported_keyword_names = functools.lru_cache(maxsize=256)(
    _supported_keyword_names
)
