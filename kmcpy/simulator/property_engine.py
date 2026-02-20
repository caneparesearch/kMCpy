"""Shared scheduling and property callback utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from kmcpy.simulator.state import SimulationState


@dataclass
class PropertySpec:
    """Property callback registration metadata."""

    name: str
    callback: Callable[["SimulationState", int, float], Any]
    interval: Optional[int] = None
    time_interval: Optional[float] = None
    store: bool = True
    max_records: Optional[int] = None
    enabled: bool = True
    on_error: Optional[Callable[[Exception, "SimulationState", int, float], bool]] = None
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
