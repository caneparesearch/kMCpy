"""Shared scheduling and property callback utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

import numpy as np

if TYPE_CHECKING:
    from kmcpy.simulator.state import State


@dataclass
class PropertySpec:
    """Property callback registration metadata."""

    name: str
    callback: Callable[["State", int, float], Any]
    interval: Optional[int] = None
    time_interval: Optional[float] = None
    store: bool = True
    max_records: Optional[int] = None
    enabled: bool = True
    on_error: Optional[Callable[[Exception, "State", int, float], bool]] = None
    last_trigger_step: int = 0
    last_trigger_time: Optional[float] = None


@dataclass
class PropertyRecord:
    """Stored callback output for one sampling point."""

    name: str
    step: int
    time: float
    value: Any



def validate_schedule(interval: Optional[int], time_interval: Optional[float]) -> None:
    """Validate callback scheduling parameters."""
    if interval is not None:
        if not isinstance(interval, int):
            raise TypeError("interval must be an integer")
        if interval <= 0:
            raise ValueError("interval must be a positive integer")

    if time_interval is not None and time_interval <= 0:
        raise ValueError("time_interval must be positive")



def validate_max_records(max_records: Optional[int]) -> None:
    """Validate storage truncation parameter."""
    if max_records is None:
        return
    if not isinstance(max_records, int):
        raise TypeError("max_records must be an integer")
    if max_records <= 0:
        raise ValueError("max_records must be a positive integer")



def should_trigger(
    step: int,
    sim_time: float,
    interval: Optional[int],
    time_interval: Optional[float],
    last_trigger_time: Optional[float],
) -> bool:
    """Return whether a scheduled property should execute."""
    step_trigger = interval is not None and step > 0 and step % interval == 0

    time_trigger = False
    if time_interval is not None:
        if last_trigger_time is None:
            time_trigger = sim_time >= time_interval
        else:
            time_trigger = (sim_time - last_trigger_time) >= time_interval

    return step_trigger or time_trigger



def append_record(records: list[PropertyRecord], spec: PropertySpec, step: int, sim_time: float, value: Any) -> None:
    """Append one callback record and enforce record limits."""
    records.append(
        PropertyRecord(name=spec.name, step=int(step), time=float(sim_time), value=value)
    )
    if spec.max_records is not None and len(records) > spec.max_records:
        del records[0 : len(records) - spec.max_records]


BUILTIN_PROPERTY_FIELDS = (
    "msd",
    "jump_diffusivity",
    "tracer_diffusivity",
    "conductivity",
    "havens_ratio",
    "correlation_factor",
)


def _is_enabled(enabled: Mapping[str, bool] | None, name: str) -> bool:
    """Return True when a metric is enabled under the provided toggle map."""
    if enabled is None:
        return True
    return bool(enabled.get(name, True))


def compute_transport_properties(
    displacement: np.ndarray,
    hop_counter: np.ndarray,
    *,
    sim_time: float,
    dimension: int,
    n_mobile_ion_specie: int,
    elementary_hop_distance: float,
    volume: float,
    mobile_ion_charge: float,
    temperature: float,
    enabled: Mapping[str, bool] | None = None,
) -> dict[str, float]:
    """Compute built-in transport properties from trajectory state."""
    nan = float("nan")

    displacement_norm_sq = np.linalg.norm(displacement, axis=1) ** 2
    msd_internal = float(np.mean(displacement_norm_sq))

    if sim_time > 0 and n_mobile_ion_specie > 0:
        displacement_vector_tot = np.linalg.norm(np.sum(displacement, axis=0))
        jump_diffusivity_internal = (
            displacement_vector_tot**2
            / (2 * dimension * sim_time * n_mobile_ion_specie)
            * 10 ** (-16)
        )
        tracer_diffusivity_internal = (
            msd_internal
            / (2 * dimension * sim_time)
            * 10 ** (-16)
        )
    else:
        jump_diffusivity_internal = nan
        tracer_diffusivity_internal = nan

    conductivity_internal = nan
    if np.isfinite(jump_diffusivity_internal):
        k = 8.617333262145 * 10 ** (-2)  # meV/K
        n_carrier = n_mobile_ion_specie / volume
        conductivity_internal = (
            jump_diffusivity_internal
            * n_carrier
            * mobile_ion_charge**2
            / (k * temperature)
            * 1.602
            * 10**11
        )

    havens_ratio_internal = nan
    if (
        np.isfinite(jump_diffusivity_internal)
        and np.isfinite(tracer_diffusivity_internal)
        and jump_diffusivity_internal != 0
    ):
        havens_ratio_internal = tracer_diffusivity_internal / jump_diffusivity_internal

    hop_counter_safe = np.where(hop_counter == 0, 1, hop_counter)
    correlation_factor_internal = displacement_norm_sq / (
        hop_counter_safe * elementary_hop_distance**2
    )
    correlation_factor_internal[hop_counter == 0] = 0
    correlation_factor_internal = float(np.mean(correlation_factor_internal))

    return {
        "msd": msd_internal if _is_enabled(enabled, "msd") else nan,
        "jump_diffusivity": (
            jump_diffusivity_internal if _is_enabled(enabled, "jump_diffusivity") else nan
        ),
        "tracer_diffusivity": (
            tracer_diffusivity_internal if _is_enabled(enabled, "tracer_diffusivity") else nan
        ),
        "conductivity": (
            conductivity_internal if _is_enabled(enabled, "conductivity") else nan
        ),
        "havens_ratio": (
            havens_ratio_internal if _is_enabled(enabled, "havens_ratio") else nan
        ),
        "correlation_factor": (
            correlation_factor_internal if _is_enabled(enabled, "correlation_factor") else nan
        ),
    }
