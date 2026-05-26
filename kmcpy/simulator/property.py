"""Shared scheduling and property callback utilities."""

from __future__ import annotations

from dataclasses import dataclass, replace
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


class PropertyPlan:
    """
    Property sampling recipe for a KMC run.

    The plan stores user intent: global cadence, enabled built-in fields, and
    custom callback registrations. Runtime sampling state and stored records
    belong to the Tracker.
    """

    def __init__(
        self,
        interval: Optional[int] = None,
        time_interval: Optional[float] = None,
        builtin_enabled: Mapping[str, bool] | None = None,
    ) -> None:
        validate_schedule(interval=interval, time_interval=time_interval)
        self._global_interval = interval
        self._global_time_interval = time_interval
        self._enabled_builtin_properties = {
            name: True for name in BUILTIN_PROPERTY_FIELDS
        }
        self._attached_properties: dict[str, PropertySpec] = {}
        if builtin_enabled:
            for name, enabled in builtin_enabled.items():
                self.set_property_enabled(name, bool(enabled))

    @property
    def global_interval(self) -> Optional[int]:
        """Return the configured global step interval."""
        return self._global_interval

    @property
    def global_time_interval(self) -> Optional[float]:
        """Return the configured global simulation-time interval."""
        return self._global_time_interval

    @property
    def builtin_enabled(self) -> dict[str, bool]:
        """Return a copy of built-in property enablement flags."""
        return self._enabled_builtin_properties.copy()

    def set_frequency(
        self,
        interval: Optional[int] = None,
        time_interval: Optional[float] = None,
    ) -> None:
        """Set the global sampling cadence for built-ins and callbacks."""
        validate_schedule(interval=interval, time_interval=time_interval)
        self._global_interval = interval
        self._global_time_interval = time_interval

    def attach(
        self,
        func: Callable[["State", int, float], Any],
        interval: Optional[int] = None,
        time_interval: Optional[float] = None,
        name: Optional[str] = None,
        store: bool = True,
        max_records: Optional[int] = None,
        on_error: Optional[Callable[[Exception, "State", int, float], bool]] = None,
        enabled: bool = True,
    ) -> str:
        """Register a custom property callback in the plan."""
        if not callable(func):
            raise TypeError("func must be callable")

        validate_schedule(interval=interval, time_interval=time_interval)
        validate_max_records(max_records=max_records)

        property_name = name or getattr(func, "__name__", "attached_property")
        if property_name in BUILTIN_PROPERTY_FIELDS:
            raise ValueError(
                f"'{property_name}' is reserved for built-in properties"
            )
        if property_name in self._attached_properties:
            raise ValueError(f"Callback '{property_name}' is already attached")

        self._attached_properties[property_name] = PropertySpec(
            name=property_name,
            callback=func,
            interval=interval,
            time_interval=time_interval,
            store=store,
            max_records=max_records,
            enabled=bool(enabled),
            on_error=on_error,
        )
        return property_name

    def detach(self, name: str) -> None:
        """Remove a custom property callback from the plan."""
        if name not in self._attached_properties:
            raise ValueError(f"Callback '{name}' is not attached")
        del self._attached_properties[name]

    def clear_attachments(self) -> None:
        """Remove all custom callbacks while preserving built-in settings."""
        self._attached_properties.clear()

    def list_attachments(self) -> list[str]:
        """Return registered custom callback names."""
        return list(self._attached_properties.keys())

    def list_property_calculations(self) -> dict[str, list[str]]:
        """Return enabled/disabled built-ins and custom callbacks."""
        built_in_enabled = [
            name
            for name in BUILTIN_PROPERTY_FIELDS
            if self._enabled_builtin_properties.get(name, True)
        ]
        built_in_disabled = [
            name
            for name in BUILTIN_PROPERTY_FIELDS
            if not self._enabled_builtin_properties.get(name, True)
        ]
        attached_enabled = [
            name
            for name, spec in self._attached_properties.items()
            if spec.enabled
        ]
        attached_disabled = [
            name
            for name, spec in self._attached_properties.items()
            if not spec.enabled
        ]
        return {
            "built_in_enabled": built_in_enabled,
            "built_in_disabled": built_in_disabled,
            "attached_enabled": attached_enabled,
            "attached_disabled": attached_disabled,
        }

    def set_property_enabled(self, name: str, enabled: bool) -> None:
        """Enable or disable one built-in field or custom callback."""
        if name in self._enabled_builtin_properties:
            self._enabled_builtin_properties[name] = bool(enabled)
            return

        if name not in self._attached_properties:
            raise ValueError(f"Unknown property '{name}'")
        self._attached_properties[name].enabled = bool(enabled)

    def fresh_attachment_specs(self) -> list[PropertySpec]:
        """
        Return callback specs with runtime sampling counters reset.

        Trackers mutate ``last_trigger_step`` and ``last_trigger_time`` while
        sampling, so each run must receive fresh spec objects.
        """
        return [
            replace(spec, last_trigger_step=0, last_trigger_time=None)
            for spec in self._attached_properties.values()
        ]


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

    if hop_counter.size > 0:
        hop_counter_safe = np.where(hop_counter == 0, 1, hop_counter)
        correlation_factor_by_ion = displacement_norm_sq / (
            hop_counter_safe * elementary_hop_distance**2
        )
        correlation_factor_by_ion = np.where(
            hop_counter == 0, 0.0, correlation_factor_by_ion
        )
        correlation_factor_internal = float(np.mean(correlation_factor_by_ion))
    else:
        correlation_factor_internal = nan

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
