"""Shared constants for serialized model payloads."""

from __future__ import annotations

from typing import Any

MODEL_FILE_FORMAT = "kmcpy.model_file"
SUPPORTED_MODEL_FILE_FORMATS = frozenset({MODEL_FILE_FORMAT})


def require_model_file_payload(payload: Any) -> dict[str, Any]:
    """Validate and return a model-file payload dictionary."""
    if not isinstance(payload, dict):
        raise ValueError("Model file must be a JSON object")

    if payload.get("format") not in SUPPORTED_MODEL_FILE_FORMATS:
        raise ValueError(
            f"Unsupported model file format. Expected '{MODEL_FILE_FORMAT}'."
        )

    return payload


def require_model_type(payload: Any, model_type: str) -> dict[str, Any]:
    """Validate that a model-file payload has the expected model type."""
    data = require_model_file_payload(payload)
    observed = data.get("model_type")
    if observed != model_type:
        raise ValueError(f"Expected model_type '{model_type}', got '{observed}'")
    return data
