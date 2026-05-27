"""File output helpers for KMC tracker results."""

from __future__ import annotations

import gzip
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from monty.json import jsanitize

from kmcpy.simulator.property import PropertyRecord


SUMMARY_FIELDS = (
    "time",
    "msd",
    "jump_diffusivity",
    "tracer_diffusivity",
    "conductivity",
    "havens_ratio",
    "correlation_factor",
)


def format_tracker_summary(
    *,
    current_pass: int,
    results: Mapping[str, Sequence[float]],
    result_units: Mapping[str, str],
    attached_values: Mapping[str, Any],
) -> str:
    """Return a printable table for the latest tracker summary."""
    rows = [["pass", current_pass, ""]]
    for field in SUMMARY_FIELDS:
        rows.append([field, results[field][-1], result_units[field]])

    for name, value in attached_values.items():
        rows.append([name, value, "user-defined"])

    return "\n" + pd.DataFrame(
        rows,
        columns=["Property", "Value", "Unit"],
    ).to_string(index=False)


def write_tracker_results(
    *,
    label: str | None,
    current_pass: int,
    displacement: np.ndarray,
    hop_counter: np.ndarray,
    occupations: Sequence[int],
    results: Mapping[str, Sequence[float]],
    result_units: Mapping[str, str],
    property_records: Mapping[str, Sequence[PropertyRecord]],
) -> None:
    """Write trajectory arrays, built-in summaries, and custom-property records."""
    trajectory_prefix = (
        f"{label}_{current_pass}" if label else str(current_pass)
    )
    _write_trajectory_arrays(
        prefix=trajectory_prefix,
        displacement=displacement,
        hop_counter=hop_counter,
        occupations=occupations,
    )

    suffix = f"_{label}" if label else ""
    _write_summary_results(
        results_file=f"results{suffix}.csv.gz",
        results_units_file=f"results_units{suffix}.json.gz",
        results=results,
        result_units=result_units,
    )
    _write_property_records(
        properties_file=f"properties{suffix}.json.gz",
        property_records=property_records,
    )


def _write_trajectory_arrays(
    *,
    prefix: str,
    displacement: np.ndarray,
    hop_counter: np.ndarray,
    occupations: Sequence[int],
) -> None:
    """Write final displacement, hop counters, and occupations."""
    np.savetxt(
        f"displacement_{prefix}.csv.gz",
        displacement,
        delimiter=",",
    )
    np.savetxt(
        f"hop_counter_{prefix}.csv.gz",
        hop_counter,
        delimiter=",",
    )
    np.savetxt(
        f"current_occ_{prefix}.csv.gz",
        occupations,
        delimiter=",",
    )


def _write_summary_results(
    *,
    results_file: str,
    results_units_file: str,
    results: Mapping[str, Sequence[float]],
    result_units: Mapping[str, str],
) -> None:
    """Write built-in transport summaries and their units."""
    pd.DataFrame(results).to_csv(results_file, compression="gzip", index=False)
    with gzip.open(results_units_file, "wt", encoding="utf-8") as fhandle:
        json.dump(dict(result_units), fhandle, indent=2)


def _write_property_records(
    *,
    properties_file: str,
    property_records: Mapping[str, Sequence[PropertyRecord]],
) -> None:
    """Write user-attached property records when any were sampled."""
    payload = _stored_property_payload(property_records)
    if not payload:
        return
    with gzip.open(properties_file, "wt", encoding="utf-8") as fhandle:
        json.dump(payload, fhandle, indent=2)


def _stored_property_payload(
    property_records: Mapping[str, Sequence[PropertyRecord]],
) -> list[dict[str, Any]]:
    """Return JSON-safe records for user-attached property callbacks."""
    payload: list[dict[str, Any]] = []
    for records in property_records.values():
        for record in records:
            payload.append(
                {
                    "name": record.name,
                    "step": int(record.step),
                    "time": float(record.time),
                    "value": _to_json_safe(record.value),
                }
            )
    return payload


def _to_json_safe(value: Any) -> Any:
    """Convert callback payloads to JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]

    try:
        return jsanitize(value)
    except TypeError:
        return repr(value)
