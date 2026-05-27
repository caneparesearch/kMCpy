#!/usr/bin/env python
"""Tracker for monitoring mobile ion trajectories during kMC simulations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from pymatgen.core import Structure
from kmcpy.simulator.property import (
    BUILTIN_PROPERTY_FIELDS,
    BUILTIN_PROPERTY_UNITS,
    PropertyPlan,
    PropertyRecord,
    PropertySpec,
    append_record,
    compute_transport_properties,
    describe_property_calculations,
    make_property_spec,
    set_property_enabled_flag,
    should_trigger,
    validate_schedule,
)
from kmcpy.simulator.results import format_tracker_summary, write_tracker_results
from kmcpy.event import INVALID_STATE, event_direction

if TYPE_CHECKING:
    from kmcpy.simulator.config import Configuration
    from kmcpy.simulator.state import State

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

RESULT_UNITS = {
    "time": "s",
    **BUILTIN_PROPERTY_UNITS,
}


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

class Tracker:
    """Track trajectories and evaluate attached properties for each sampling point.

    Built-in result units are available through ``Tracker.result_units`` and are
    also written next to the result CSV by ``write_results``.
    """

    def __init__(
        self,
        config: "Configuration",
        structure: Structure,
        initial_state: Optional["State"] = None,
        property_plan: Optional[PropertyPlan] = None,
        default_property_interval: Optional[int] = None,
        hop_state_lookup: Any = None,
    ) -> None:
        """Initialize tracker state, trajectory arrays, and built-in sampling."""
        logger.info("Initializing Tracker ...")
        if initial_state is None:
            raise ValueError("State must be provided to Tracker")

        self.config = config
        self.structure = structure
        self.state = initial_state
        self.hop_state_lookup = hop_state_lookup

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
        self._last_summary_trigger_time: Optional[float] = None

        if property_plan is not None:
            self.apply_property_plan(
                property_plan,
                default_interval=default_property_interval,
            )

        logger.info("number of mobile ion specie = %d", self.n_mobile_ion_specie)
        logger.info(
            "Center of mass (%s): %s",
            self.mobile_ion_specie,
            np.mean(self.r0, axis=0),
        )

    def _initialize_mobile_ion_tracking(self, initial_occ: list[int]) -> None:
        """Initialize mobile ion tracking arrays."""
        if self.hop_state_lookup is None:
            self.n_mobile_ion_specie_site = len(
                [el.symbol for el in self.structure.species if self.mobile_ion_specie in el.symbol]
            )
            initial_active_occ = np.array(initial_occ[0 : self.n_mobile_ion_specie_site])
            mobile_state_mask = initial_active_occ == 0
        else:
            mobile_states = self.hop_state_lookup.mobile_state_by_site
            self.n_mobile_ion_specie_site = int(np.sum(mobile_states != INVALID_STATE))
            initial_active_occ = np.array(initial_occ[0 : len(mobile_states)])
            mobile_state_mask = (
                (mobile_states != INVALID_STATE)
                & (initial_active_occ == mobile_states)
            )
        self.mobile_ion_specie_locations = np.where(mobile_state_mask)[0]
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

    @property
    def result_units(self) -> dict[str, str]:
        """Return units for built-in result fields."""
        return dict(RESULT_UNITS)

    def set_global_property_frequency(
        self,
        interval: Optional[int] = None,
        time_interval: Optional[float] = None,
    ) -> None:
        """Set global sampling defaults for all attached properties."""
        validate_schedule(interval=interval, time_interval=time_interval)
        self._global_interval = interval
        self._global_time_interval = time_interval

    def apply_property_plan(
        self,
        property_plan: PropertyPlan,
        default_interval: Optional[int] = None,
    ) -> None:
        """Apply a property sampling recipe to this tracker."""
        interval = property_plan.global_interval
        time_interval = property_plan.global_time_interval

        if interval is None and time_interval is None and default_interval is not None:
            interval = default_interval

        self.set_global_property_frequency(
            interval=interval,
            time_interval=time_interval,
        )

        for property_name, enabled in property_plan.builtin_enabled.items():
            self.set_property_enabled(property_name, enabled)

        for spec in property_plan.fresh_attachment_specs():
            self.attach_spec(spec)

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
        """Attach one property callback to this tracker."""
        spec = make_property_spec(
            func,
            interval=interval,
            time_interval=time_interval,
            name=name,
            store=store,
            max_records=max_records,
            on_error=on_error,
            enabled=enabled,
            existing_names=set(self._properties),
        )
        self._properties[spec.name] = spec
        self._property_records[spec.name] = []
        return spec.name

    def attach_spec(self, spec: PropertySpec) -> str:
        """Attach a prevalidated property specification to this tracker."""
        return self.attach(
            spec.callback,
            interval=spec.interval,
            time_interval=spec.time_interval,
            name=spec.name,
            store=spec.store,
            max_records=spec.max_records,
            on_error=spec.on_error,
            enabled=spec.enabled,
        )

    def detach(self, name: str) -> None:
        """Detach a previously attached property callback."""
        if name not in self._properties:
            raise ValueError(f"Property '{name}' is not attached")
        del self._properties[name]
        self._property_records.pop(name, None)

    def clear_attachments(self) -> None:
        """Remove all user-attached property callbacks."""
        self._properties.clear()
        self._property_records.clear()

    def list_attachments(self) -> list[str]:
        """Return names of user-attached properties."""
        return list(self._properties)

    def list_property_calculations(self) -> dict[str, list[str]]:
        """Return enabled/disabled built-ins and currently attached callbacks."""
        return describe_property_calculations(
            builtin_enabled=self._enabled_builtin_properties,
            attached_properties=self._properties,
        )

    def set_property_enabled(self, name: str, enabled: bool) -> None:
        """Enable or disable a built-in summary field or an attached callback."""
        set_property_enabled_flag(
            builtin_enabled=self._enabled_builtin_properties,
            attached_properties=self._properties,
            name=name,
            enabled=enabled,
        )

    def _compute_transport_summary(self) -> dict[str, float]:
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

    def _sample_transport_summary(self, step: int, sim_time: float) -> None:
        """Sample built-in transport metrics when the global schedule is due."""
        if not should_trigger(
            step=step,
            sim_time=sim_time,
            interval=self._global_interval,
            time_interval=self._global_time_interval,
            last_trigger_time=self._last_summary_trigger_time,
        ):
            return

        metrics = self._compute_transport_summary()
        _append_result_row(self.results, sim_time, metrics)
        self._last_summary_trigger_time = sim_time

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

    def _latest_property_value(self, name: str) -> Any:
        """Return latest sampled value for a property name."""
        records = self._property_records.get(name, [])
        if not records:
            return float("nan")
        return records[-1].value

    def sample_properties(self, step: int, sim_time: float) -> None:
        """Evaluate schedules and execute all attached property callbacks."""
        self._sample_transport_summary(step=step, sim_time=sim_time)

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
        }

    def update(self, event, dt) -> None:
        """Update trajectory observables using the current pre-event State."""
        _ = dt
        occupations = self.state.occupations
        direction = event_direction(occupations, event)
        if direction == 0:
            logger.error("Proposed event does not match current endpoint occupations")
            return

        logger.debug(
            "Tracker update: event=%s direction=%d time=%.6f",
            event.mobile_ion_indices,
            direction,
            self.time,
        )

        mobile_ion_index = self._record_mobile_ion_hop(event, direction)
        displacement = self._wrapped_hop_displacement(event, direction)
        self.displacement[mobile_ion_index] += displacement
        self.hop_counter[mobile_ion_index] += 1

    def _record_mobile_ion_hop(self, event, direction: int) -> int:
        """Move the tracked mobile ion identity and return its row index."""
        from_site, to_site = event.mobile_ion_indices
        source_site = from_site if direction == 1 else to_site
        destination_site = to_site if direction == 1 else from_site

        matches = np.where(self.mobile_ion_specie_locations == source_site)[0]
        if len(matches) == 0:
            raise RuntimeError(
                "Tracker mobile-ion locations are inconsistent with the "
                f"accepted event source site {source_site}."
            )

        mobile_ion_index = int(matches[0])
        self.mobile_ion_specie_locations[mobile_ion_index] = destination_site
        return mobile_ion_index

    def _wrapped_hop_displacement(self, event, direction: int) -> np.ndarray:
        """Return the minimum-image Cartesian displacement for one accepted hop."""
        from_site, to_site = event.mobile_ion_indices
        displacement_frac = direction * (
            self.frac_coords[to_site] - self.frac_coords[from_site]
        )
        displacement_frac -= np.round(displacement_frac).astype(int)
        return np.array(self.latt.get_cartesian_coords(displacement_frac))

    def update_current_pass(self, current_pass: int) -> None:
        """Update current pass index used in logging/output."""
        self.current_pass = current_pass

    def show_current_info(self) -> None:
        """Log current simulation information and latest sampled summary."""
        if not self.results["time"]:
            logger.info("Pass %d has no sampled properties yet.", self.current_pass)
            return

        attached_values = {
            name: self._latest_property_value(name)
            for name in self.list_attachments()
        }
        logger.info(
            "Tracker Summary:%s",
            format_tracker_summary(
                current_pass=self.current_pass,
                results=self.results,
                result_units=self.result_units,
                attached_values=attached_values,
            ),
        )

    def return_current_info(self) -> tuple[float, float, float, float, float, float, float]:
        """Return latest sampled summary values for testing/reporting.

        Units follow ``Tracker.result_units`` and tuple order is:
        ``time``, ``msd``, ``jump_diffusivity``, ``tracer_diffusivity``,
        ``conductivity``, ``havens_ratio``, ``correlation_factor``.
        """
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

    def write_results(self, label: str | None = None) -> None:
        """Write trajectory arrays, built-in summaries, and custom-property records."""
        write_tracker_results(
            label=label,
            current_pass=self.current_pass,
            displacement=self.displacement,
            hop_counter=self.hop_counter,
            occupations=self.state.occupations,
            results=self.results,
            result_units=self.result_units,
            property_records=self._property_records,
        )
