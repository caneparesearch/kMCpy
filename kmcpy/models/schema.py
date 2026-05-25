"""Shared constants for serialized model payloads."""

from __future__ import annotations

from typing import Any

MODEL_FILETYPE = "kmcpy.model_file"
SUPPORTED_MODEL_FILETYPES = frozenset({MODEL_FILETYPE})


def require_model_file_payload(payload: Any) -> dict[str, Any]:
    """Validate and return a model-file payload dictionary."""
    if not isinstance(payload, dict):
        raise ValueError("Model file must be a JSON object")

    if payload.get("filetype") not in SUPPORTED_MODEL_FILETYPES:
        raise ValueError(
            f"Unsupported model filetype. Expected '{MODEL_FILETYPE}'."
        )

    return payload


def require_model_type(payload: Any, model_type: str) -> dict[str, Any]:
    """Validate that a model-file payload has the expected model type."""
    data = require_model_file_payload(payload)
    observed = data.get("model_type")
    if observed != model_type:
        raise ValueError(f"Expected model_type '{model_type}', got '{observed}'")
    return data
