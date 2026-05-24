"""Helpers for loading, validating, and building serialized model files."""

from __future__ import annotations

from typing import Any

from kmcpy.io.files import load_json, save_json

MODEL_FILE_FORMAT = "kmcpy.model_file"
LEGACY_MODEL_FILE_FORMATS = {"kmcpy.model_bundle.v1"}


def _validate_model_component(name: str, component: dict[str, Any]) -> None:
    if not isinstance(component, dict):
        raise ValueError(f"Model component '{name}' must be an object")

    if "lce" not in component or not isinstance(component["lce"], dict):
        raise ValueError(f"Model component '{name}' must contain object key 'lce'")

    parameters = component.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(
            f"Model component '{name}' must contain object key 'parameters'"
        )
    if "keci" not in parameters or "empty_cluster" not in parameters:
        raise ValueError(
            f"Model component '{name}.parameters' must contain keys "
            "'keci' and 'empty_cluster'"
        )


def _validate_composite_lce_model_file(model_data: dict[str, Any]) -> None:
    if "kra" not in model_data:
        raise ValueError("Composite model file must contain required key 'kra'")

    _validate_model_component("kra", model_data["kra"])
    if "site" in model_data and model_data["site"] is not None:
        _validate_model_component("site", model_data["site"])


def _validate_tabulated_model_payload(component: dict[str, Any]) -> None:
    if not isinstance(component, dict):
        raise ValueError("Tabulated model payload must be an object")

    entries = component.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "Tabulated model payload must contain non-empty list key 'entries'"
        )

    from kmcpy.models.tabulated_model import TabulatedModel

    TabulatedModel.from_dict(component)


def _validate_tabulated_model_file(model_data: dict[str, Any]) -> None:
    if "tabulated" not in model_data:
        raise ValueError("Tabulated model file must contain required key 'tabulated'")
    _validate_tabulated_model_payload(model_data["tabulated"])


def validate_model_file(model_data: dict[str, Any]) -> None:
    """Validate a serialized model-file dictionary."""
    if not isinstance(model_data, dict):
        raise ValueError("Model file must be a JSON object")

    model_format = model_data.get("format")
    if model_format != MODEL_FILE_FORMAT and model_format not in LEGACY_MODEL_FILE_FORMATS:
        raise ValueError(
            f"Unsupported model file format. Expected '{MODEL_FILE_FORMAT}'."
        )

    model_type = model_data.get("model_type")
    if model_type == "composite_lce":
        _validate_composite_lce_model_file(model_data)
        return
    if model_type == "tabulated":
        _validate_tabulated_model_file(model_data)
        return

    raise ValueError(
        "Model file 'model_type' must be one of: ['composite_lce', 'tabulated']."
    )


def load_model_file(model_file: str) -> dict[str, Any]:
    """Load and validate a serialized model JSON file."""
    model_data = load_json(model_file)
    validate_model_file(model_data)
    return model_data


def save_model_file(
    model_data: dict[str, Any],
    output_file: str,
    indent: int = 2,
) -> None:
    """Validate and save a serialized model JSON file."""
    validate_model_file(model_data)
    save_json(model_data, output_file, indent=indent)


def _latest_fit_record(fit_file: str) -> dict[str, Any]:
    """Load the latest fitting record from a legacy fitting-results JSON file."""
    payload = load_json(fit_file)
    rows = [row for row in payload.values() if isinstance(row, dict)]
    if not rows:
        raise ValueError(f"No fitting rows found in {fit_file}")
    rows_with_ts = [row for row in rows if row.get("time_stamp") is not None]
    return max(rows_with_ts, key=lambda row: row["time_stamp"]) if rows_with_ts else rows[-1]


def _model_component_from_legacy_files(
    lce_file: str,
    fit_file: str,
) -> dict[str, Any]:
    lce = load_json(lce_file)
    fit = _latest_fit_record(fit_file)
    if "keci" not in fit or "empty_cluster" not in fit:
        raise ValueError(
            f"Fitting results file {fit_file} must contain keys keci and empty_cluster"
        )

    return {
        "lce": lce,
        "parameters": {
            "keci": fit["keci"],
            "empty_cluster": fit["empty_cluster"],
        },
        "fit_metadata": {
            key: value
            for key, value in fit.items()
            if key not in {"keci", "empty_cluster"}
        },
    }


def build_model_file_from_legacy_files(
    kra_lce: str,
    kra_fit: str,
    site_lce: str | None = None,
    site_fit: str | None = None,
) -> dict[str, Any]:
    """Build a composite LCE model file from legacy LCE and fitting JSON files."""
    model_data: dict[str, Any] = {
        "format": MODEL_FILE_FORMAT,
        "model_type": "composite_lce",
        "kra": _model_component_from_legacy_files(kra_lce, kra_fit),
    }

    if site_lce is not None or site_fit is not None:
        if site_lce is None or site_fit is None:
            raise ValueError("Both site_lce and site_fit are required for a site model")
        model_data["site"] = _model_component_from_legacy_files(site_lce, site_fit)

    validate_model_file(model_data)
    return model_data


def build_tabulated_model_file(
    entries: list[dict[str, Any]],
    name: str = "TabulatedModel",
    lookup_key: str = "event_local_occupation",
    default_property: str = "barrier",
    probability_mode: str = "barrier_arrhenius",
    probability_property: str = "barrier",
) -> dict[str, Any]:
    """Build a tabulated model file from explicit entries."""
    from kmcpy.models.tabulated_model import TabulatedModel

    model = TabulatedModel.from_entries(
        entries=entries,
        name=name,
        lookup_key=lookup_key,
        default_property=default_property,
        probability_mode=probability_mode,
        probability_property=probability_property,
    )
    model_data = model.to_model_file_dict()
    validate_model_file(model_data)
    return model_data


def build_tabulated_model_file_from_entries_file(
    entries_file: str,
    name: str | None = None,
    lookup_key: str | None = None,
    default_property: str | None = None,
    probability_mode: str | None = None,
    probability_property: str | None = None,
) -> dict[str, Any]:
    """
    Build a tabulated model file from a JSON entries payload.

    Accepted payloads:
    - JSON list of entries
    - JSON object with key ``entries`` and optional metadata
    - Full tabulated model file with ``format/model_type/tabulated``
    """
    payload = load_json(entries_file)
    if isinstance(payload, list):
        entries = payload
        payload_metadata: dict[str, Any] = {}
    elif isinstance(payload, dict):
        if (
            payload.get("format") in {MODEL_FILE_FORMAT, *LEGACY_MODEL_FILE_FORMATS}
            and payload.get("model_type") == "tabulated"
        ):
            payload = payload.get("tabulated")
        if not isinstance(payload, dict):
            raise ValueError("Tabulated entries file has invalid payload structure")
        if "entries" not in payload:
            raise ValueError("Tabulated entries JSON object must include key 'entries'")
        entries = payload["entries"]
        payload_metadata = payload
    else:
        raise ValueError("Tabulated entries file must contain a JSON list or object")

    resolved_name = name or payload_metadata.get("name", "TabulatedModel")
    resolved_lookup_key = lookup_key or payload_metadata.get(
        "lookup_key", "event_local_occupation"
    )
    resolved_default_property = default_property or payload_metadata.get(
        "default_property", "barrier"
    )
    resolved_probability_mode = probability_mode or payload_metadata.get(
        "probability_mode", "barrier_arrhenius"
    )
    resolved_probability_property = probability_property or payload_metadata.get(
        "probability_property", "barrier"
    )

    return build_tabulated_model_file(
        entries=entries,
        name=resolved_name,
        lookup_key=resolved_lookup_key,
        default_property=resolved_default_property,
        probability_mode=resolved_probability_mode,
        probability_property=resolved_probability_property,
    )

