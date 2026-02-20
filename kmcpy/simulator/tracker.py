#!/usr/bin/env python
"""Tracker for monitoring mobile ion trajectories during kMC simulations."""

from __future__ import annotations

from copy import copy
import gzip
import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import pandas as pd

from kmcpy.external.structure import StructureKMCpy
from kmcpy.io.serialization import to_json_compatible
from kmcpy.simulator.built_in_properties import (
    BUILTIN_PROPERTY_FIELDS,
    compute_transport_properties,
)
from kmcpy.simulator.property_engine import (
    PropertyRecord,
    PropertySpec,
    append_record,
    should_trigger,
    validate_max_records,
    validate_schedule,
)

if TYPE_CHECKING:
    from kmcpy.simulator.config import SimulationConfig
    from kmcpy.simulator.state import SimulationState

logger = logging.getLogger(__name__)

RESULT_FIELDS = (
    "time",
    "jump_diffusivity",
    "tracer_diffusivity",
    "conductivity",
    "correlation_factor",
    "havens_ratio",
    "msd",
)

SUMMARY_PROPERTY_NAME = "_built_in_summary"


class CallbackExecutionError(RuntimeError):
    """Raised when an attached property callback fails and cannot be recovered."""



def _create_result_store() -> dict[str, list[float]]:
    """Allocate empty storage lists for built-in summary fields."""
    return {field: [] for field in RESULT_FIELDS}



def _append_result_row(store: dict[str, list[float]], sim_time: float, metrics: dict[str, float]) -> None:
    """Append one built-in summary sample to the result table."""
    store["time"].append(sim_time)
    store["jump_diffusivity"].append(metrics["jump_diffusivity"])
    store["tracer_diffusivity"].append(metrics["tracer_diffusivity"])
    store["conductivity"].append(metrics["conductivity"])
    store["correlation_factor"].append(metrics["correlation_factor"])
    store["havens_ratio"].append(metrics["havens_ratio"])
    store["msd"].append(metrics["msd"])



def _to_json_safe(value: Any) -> Any:
    """Convert callback payloads to JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]

    try:
        return to_json_compatible(value)
    except TypeError:
        return repr(value)


class Tracker:
    """Track trajectories and evaluate attached properties for each sampling point."""

    def __init__(
        self,
        config: "SimulationConfig",
        structure: StructureKMCpy,
        initial_state: Optional["SimulationState"] = None,
    ) -> None:
        """Initialize tracker state, trajectory arrays, and built-in sampling."""
        logger.info("Initializing Tracker ...")
        if initial_state is None:
            raise ValueError("SimulationState must be provided to Tracker")

        self.config = config
        self.structure = structure
        self.state = initial_state

        self._initialize_mobile_ion_tracking(initial_state.occupations)

        self.results = _create_result_store()
        self.current_pass = 0

        self._global_interval: Optional[int] = 1
        self._global_time_interval: Optional[float] = None
        self._enabled_builtin_properties = {
            name: True for name in BUILTIN_PROPERTY_FIELDS
        }

        self._properties: dict[str, PropertySpec] = {}
        self._property_records: dict[str, list[PropertyRecord]] = {}

        self.attach(
            self._compute_built_in_summary,
            name=SUMMARY_PROPERTY_NAME,
            store=False,
        )

        logger.info("number of mobile ion specie = %d", self.n_mobile_ion_specie)
        logger.info(
            "Center of mass (%s): %s",
            self.mobile_ion_specie,
            np.mean(self.r0, axis=0),
        )

    def _initialize_mobile_ion_tracking(self, initial_occ: list[int]) -> None:
        """Initialize mobile ion tracking arrays."""
        self.n_mobile_ion_specie_site = len(
            [el.symbol for el in self.structure.species if self.mobile_ion_specie in el.symbol]
        )
        self.mobile_ion_specie_locations = np.where(
            np.array(initial_occ[0 : self.n_mobile_ion_specie_site]) == -1
        )[0]
        self.n_mobile_ion_specie = len(self.mobile_ion_specie_locations)

        logger.debug("Initial mobile ion locations = %s", self.mobile_ion_specie_locations)

        self.displacement = np.zeros((self.n_mobile_ion_specie, 3))
        self.hop_counter = np.zeros(self.n_mobile_ion_specie, dtype=np.int64)
        self.r0 = self.frac_coords[self.mobile_ion_specie_locations] @ self.latt.matrix

    @property
    def occ_initial(self) -> list:
        """Return current occupations from the shared simulation state."""
        return self.state.occupations

    @property
    def frac_coords(self) -> np.ndarray:
        """Return structure fractional coordinates."""
        return self.structure.frac_coords

    @property
    def latt(self):
        """Return structure lattice."""
        return self.structure.lattice

    @property
    def volume(self) -> float:
        """Return structure volume."""
        return self.structure.volume

    @property
    def dimension(self) -> int:
        """Return simulation dimensionality."""
        return self.config.dimension

    @property
    def q(self) -> float:
        """Return mobile ion charge."""
        return self.config.mobile_ion_charge

    @property
    def elem_hop_distance(self) -> float:
        """Return elementary hop distance."""
        return self.config.elementary_hop_distance

    @property
    def temperature(self) -> float:
        """Return simulation temperature."""
        return self.config.temperature

    @property
    def v(self) -> float:
        """Return attempt frequency."""
        return self.config.attempt_frequency

    @property
    def time(self) -> float:
        """Return current simulation time."""
        return self.state.time

    @property
    def mobile_ion_specie(self) -> str:
        """Return tracked mobile ion species label."""
        return self.config.mobile_ion_specie

    @classmethod
    def from_config(
        cls,
        config: "SimulationConfig",
        structure: StructureKMCpy,
        occ_initial: list,
    ) -> "Tracker":
        """Construct tracker from config and initial occupations."""
        from kmcpy.simulator.state import SimulationState

        initial_state = SimulationState(occupations=occ_initial)
        return cls(config=config, structure=structure, initial_state=initial_state)

    def set_global_property_frequency(
        self,
        interval: Optional[int] = None,
        time_interval: Optional[float] = None,
    ) -> None:
        """Set global sampling defaults for all attached properties."""
        validate_schedule(interval=interval, time_interval=time_interval)
        self._global_interval = interval
        self._global_time_interval = time_interval

    def attach(
        self,
        func: Callable[["SimulationState", int, float], Any],
        interval: Optional[int] = None,
        time_interval: Optional[float] = None,
        name: Optional[str] = None,
        store: bool = True,
        max_records: Optional[int] = None,
        on_error: Optional[Callable[[Exception, "SimulationState", int, float], bool]] = None,
        enabled: bool = True,
    ) -> str:
        """Attach one property callback to this tracker."""
        if not callable(func):
            raise TypeError("func must be callable")

        validate_schedule(interval=interval, time_interval=time_interval)
        validate_max_records(max_records=max_records)

        property_name = name or getattr(func, "__name__", "attached_property")
        if property_name in BUILTIN_PROPERTY_FIELDS:
            raise ValueError(
                f"'{property_name}' is reserved for built-in summary fields"
            )
        if property_name in self._properties:
            raise ValueError(f"Property '{property_name}' is already attached")

        spec = PropertySpec(
            name=property_name,
            callback=func,
            interval=interval,
            time_interval=time_interval,
            store=store,
            max_records=max_records,
            enabled=enabled,
            on_error=on_error,
        )
        self._properties[property_name] = spec
        self._property_records[property_name] = []
        return property_name

    def detach(self, name: str) -> None:
        """Detach a previously attached property callback."""
        if name == SUMMARY_PROPERTY_NAME:
            raise ValueError("Cannot detach the built-in summary property")
        if name not in self._properties:
            raise ValueError(f"Property '{name}' is not attached")
        del self._properties[name]
        self._property_records.pop(name, None)

    def clear_attachments(self) -> None:
        """Remove all user attachments while preserving built-in summary sampling."""
        self._properties = {
            SUMMARY_PROPERTY_NAME: self._properties[SUMMARY_PROPERTY_NAME]
        }
        self._property_records = {
            SUMMARY_PROPERTY_NAME: self._property_records[SUMMARY_PROPERTY_NAME]
        }

    def list_attachments(self) -> list[str]:
        """Return names of user-attached properties."""
        return [name for name in self._properties if name != SUMMARY_PROPERTY_NAME]

    def set_property_enabled(self, name: str, enabled: bool) -> None:
        """Enable or disable a built-in summary field or an attached callback."""
        if name in self._enabled_builtin_properties:
            self._enabled_builtin_properties[name] = bool(enabled)
            return

        if name not in self._properties:
            raise ValueError(f"Unknown property '{name}'")
        self._properties[name].enabled = bool(enabled)

    def _compute_built_in_summary(self, _state: "SimulationState", _step: int, _sim_time: float) -> dict[str, float]:
        """Compute the built-in transport summary from current tracker state."""
        return compute_transport_properties(
            self.displacement,
            self.hop_counter,
            sim_time=float(self.time),
            dimension=self.dimension,
            n_mobile_ion_specie=self.n_mobile_ion_specie,
            elementary_hop_distance=self.elem_hop_distance,
            volume=self.volume,
            mobile_ion_charge=self.q,
            temperature=self.temperature,
            enabled=self._enabled_builtin_properties,
        )

    def _handle_callback_error(
        self,
        spec: PropertySpec,
        exc: Exception,
        step: int,
        sim_time: float,
    ) -> None:
        """Handle callback failures using optional on_error policy."""
        if spec.on_error is None:
            raise CallbackExecutionError(
                f"Property callback '{spec.name}' failed at step={step}, time={sim_time}"
            ) from exc

        try:
            keep_running = bool(spec.on_error(exc, self.state, step, sim_time))
        except Exception as handler_exc:
            raise CallbackExecutionError(
                f"Error handler for callback '{spec.name}' failed"
            ) from handler_exc

        if not keep_running:
            raise CallbackExecutionError(
                f"Property callback '{spec.name}' failed and requested termination"
            ) from exc

    def _append_summary_result(self, sim_time: float, metrics: dict[str, float]) -> None:
        """Persist one built-in summary sample."""
        _append_result_row(self.results, copy(sim_time), metrics)

    def _latest_property_value(self, name: str) -> Any:
        """Return latest sampled value for a property name."""
        records = self._property_records.get(name, [])
        if not records:
            return float("nan")
        return records[-1].value

    def sample_properties(self, step: int, sim_time: float) -> None:
        """Evaluate schedules and execute all attached property callbacks."""
        for spec in list(self._properties.values()):
            if not spec.enabled:
                continue

            interval = spec.interval if spec.interval is not None else self._global_interval
            time_interval = (
                spec.time_interval
                if spec.time_interval is not None
                else self._global_time_interval
            )

            if not should_trigger(
                step=step,
                sim_time=sim_time,
                interval=interval,
                time_interval=time_interval,
                last_trigger_time=spec.last_trigger_time,
            ):
                continue

            try:
                value = spec.callback(self.state, step, sim_time)
            except Exception as exc:
                self._handle_callback_error(spec=spec, exc=exc, step=step, sim_time=sim_time)
                spec.last_trigger_step = step
                spec.last_trigger_time = sim_time
                continue

            if spec.name == SUMMARY_PROPERTY_NAME:
                self._append_summary_result(sim_time=sim_time, metrics=value)

            if spec.store:
                append_record(
                    records=self._property_records[spec.name],
                    spec=spec,
                    step=step,
                    sim_time=sim_time,
                    value=value,
                )

            spec.last_trigger_step = step
            spec.last_trigger_time = sim_time

    def get_property_records(
        self, name: Optional[str] = None
    ) -> dict[str, list[dict[str, Any]]] | list[dict[str, Any]]:
        """Retrieve stored callback records."""
        if name is not None:
            if name not in self._property_records:
                raise ValueError(f"Property '{name}' has no stored records")
            return [record.__dict__.copy() for record in self._property_records[name]]

        return {
            key: [record.__dict__.copy() for record in records]
            for key, records in self._property_records.items()
            if key != SUMMARY_PROPERTY_NAME
        }

    def update(self, event, current_occ, dt) -> None:
        """Update trajectory observables for a proposed kMC event."""
        _ = dt
        mobile_ion_specie_1_coord = copy(self.frac_coords[event.mobile_ion_indices[0]])
        mobile_ion_specie_2_coord = copy(self.frac_coords[event.mobile_ion_indices[1]])
        mobile_ion_specie_1_occ = current_occ[event.mobile_ion_indices[0]]
        mobile_ion_specie_2_occ = current_occ[event.mobile_ion_indices[1]]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("--------------------- Tracker Update Start ---------------------")
            logger.debug(
                "%s(1): idx=%d, coord=%s, occ=%s",
                self.mobile_ion_specie,
                event.mobile_ion_indices[0],
                np.array2string(mobile_ion_specie_1_coord, precision=4),
                mobile_ion_specie_1_occ,
            )
            logger.debug(
                "%s(2): idx=%d, coord=%s, occ=%s",
                self.mobile_ion_specie,
                event.mobile_ion_indices[1],
                np.array2string(mobile_ion_specie_2_coord, precision=4),
                mobile_ion_specie_2_occ,
            )
            logger.debug("Current simulation time: %.6f", self.time)
            logger.debug(
                "Hop counters: %s",
                np.array2string(self.hop_counter, precision=0, separator=", "),
            )
            logger.debug("Event probability: %s", getattr(event, "probability", None))
            logger.debug(
                "Mobile ion locations before update: %s",
                self.mobile_ion_specie_locations,
            )
            logger.debug(
                "Occupation before update: %s",
                np.array2string(current_occ, precision=0, separator=", "),
            )

        direction = int((mobile_ion_specie_2_occ - mobile_ion_specie_1_occ) / 2)
        displacement_frac = copy(direction * (mobile_ion_specie_2_coord - mobile_ion_specie_1_coord))
        displacement_frac -= np.array([int(round(i)) for i in displacement_frac])
        displacement_cart = copy(self.latt.get_cartesian_coords(displacement_frac))

        if direction == -1:
            logger.debug(
                "Diffuse direction: %s(2) -> %s(1)",
                self.mobile_ion_specie,
                self.mobile_ion_specie,
            )
            specie_to_diff = np.where(
                self.mobile_ion_specie_locations == event.mobile_ion_indices[1]
            )[0][0]
            self.mobile_ion_specie_locations[specie_to_diff] = event.mobile_ion_indices[0]
        elif direction == 1:
            logger.debug(
                "Diffuse direction: %s(1) -> %s(2)",
                self.mobile_ion_specie,
                self.mobile_ion_specie,
            )
            specie_to_diff = np.where(
                self.mobile_ion_specie_locations == event.mobile_ion_indices[0]
            )[0][0]
            self.mobile_ion_specie_locations[specie_to_diff] = event.mobile_ion_indices[1]
        else:
            logger.error("Proposed a wrong event! Please check the code!")
            return

        self.displacement[specie_to_diff] += copy(np.array(displacement_cart))
        self.hop_counter[specie_to_diff] += 1
        logger.debug("------------------------ Tracker Update End --------------------")

    def update_current_pass(self, current_pass: int) -> None:
        """Update current pass index used in logging/output."""
        self.current_pass = current_pass

    def show_current_info(self) -> None:
        """Log current simulation information and latest sampled summary."""
        if not self.results["time"]:
            logger.info("Pass %d has no sampled properties yet.", self.current_pass)
            return

        rows = [
            ["pass", self.current_pass],
            ["time", self.results["time"][-1]],
            ["msd", self.results["msd"][-1]],
            ["jump_diffusivity", self.results["jump_diffusivity"][-1]],
            ["tracer_diffusivity", self.results["tracer_diffusivity"][-1]],
            ["conductivity", self.results["conductivity"][-1]],
            ["havens_ratio", self.results["havens_ratio"][-1]],
            ["correlation_factor", self.results["correlation_factor"][-1]],
        ]

        for name in self.list_attachments():
            value = self._latest_property_value(name)
            rows.append([name, value])

        table = "\n" + pd.DataFrame(rows, columns=["Property", "Value"]).to_string(index=False)
        logger.info("Tracker Summary:%s", table)

    def return_current_info(self) -> tuple[float, float, float, float, float, float, float]:
        """Return latest sampled summary values for testing/reporting."""
        if not self.results["time"]:
            raise ValueError("No property samples are available. Increase sampling frequency.")

        return (
            self.results["time"][-1],
            self.results["msd"][-1],
            self.results["jump_diffusivity"][-1],
            self.results["tracer_diffusivity"][-1],
            self.results["conductivity"][-1],
            self.results["havens_ratio"][-1],
            self.results["correlation_factor"][-1],
        )

    def write_results(self, current_occupation: list, label: str | None = None) -> None:
        """Save displacement/hop arrays, summary CSV, and property records."""
        if label:
            prefix = f"{label}_{self.current_pass}"
        else:
            prefix = f"{self.current_pass}"
        np.savetxt(
            f"displacement_{prefix}.csv.gz",
            self.displacement,
            delimiter=",",
        )
        np.savetxt(
            f"hop_counter_{prefix}.csv.gz",
            self.hop_counter,
            delimiter=",",
        )
        np.savetxt(
            f"current_occ_{prefix}.csv.gz",
            current_occupation,
            delimiter=",",
        )

        if label:
            results_file = f"results_{label}.csv.gz"
            properties_file = f"properties_{label}.json.gz"
        else:
            results_file = "results.csv.gz"
            properties_file = "properties.json.gz"

        pd.DataFrame(self.results).to_csv(results_file, compression="gzip", index=False)

        flat_records: list[dict[str, Any]] = []
        for name, records in self._property_records.items():
            if name == SUMMARY_PROPERTY_NAME:
                continue
            for record in records:
                flat_records.append(
                    {
                        "name": record.name,
                        "step": int(record.step),
                        "time": float(record.time),
                        "value": _to_json_safe(record.value),
                    }
                )

        if flat_records:
            with gzip.open(properties_file, "wt", encoding="utf-8") as fhandle:
                json.dump(flat_records, fhandle, indent=2)
