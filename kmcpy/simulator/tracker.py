#!/usr/bin/env python
"""
This module defines a Tracker class for monitoring mobile ion species in kinetic Monte Carlo (kMC) simulations.
"""

from dataclasses import dataclass
from copy import copy
import gzip
import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import pandas as pd

from kmcpy.external.structure import StructureKMCpy
from kmcpy.io import convert
from kmcpy.io.serialization import to_json_compatible

if TYPE_CHECKING:
    from kmcpy.simulator.condition import SimulationConfig
    from kmcpy.simulator.state import SimulationState

logger = logging.getLogger(__name__)

RESULT_FIELDS = (
    "time",
    "D_J",
    "D_tracer",
    "conductivity",
    "f",
    "H_R",
    "msd",
)

BUILTIN_PROPERTY_FIELDS = (
    "msd",
    "D_J",
    "D_tracer",
    "conductivity",
    "H_R",
    "f",
)


class CallbackExecutionError(RuntimeError):
    """Raised when an attached property callback fails and cannot be recovered."""


@dataclass
class PropertyRegistration:
    """User-defined property callback registration metadata."""

    name: str
    callback: Callable[["SimulationState", int, float], Any]
    interval: Optional[int] = None
    time_interval: Optional[float] = None
    store: bool = True
    max_records: Optional[int] = None
    on_error: Optional[Callable[[Exception, "SimulationState", int, float], bool]] = None
    last_trigger_step: int = 0
    last_trigger_time: Optional[float] = None


def _create_result_store() -> dict:
    return {field: [] for field in RESULT_FIELDS}


def _append_result(store: dict, time, D_J, D_tracer, conductivity, f, H_R, msd) -> None:
    store["time"].append(time)
    store["D_J"].append(D_J)
    store["D_tracer"].append(D_tracer)
    store["conductivity"].append(conductivity)
    store["f"].append(f)
    store["H_R"].append(H_R)
    store["msd"].append(msd)


def _to_json_safe(value: Any) -> Any:
    """Convert arbitrary callback payloads to JSON-compatible data."""
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
    """
    Tracker class for monitoring mobile ion species in kinetic Monte Carlo (kMC) simulations.

    The Tracker class is responsible for tracking the positions, displacements, hop counts, and related transport properties
    of mobile ion species within a given structure during kMC simulations. It provides methods to update the tracked state
    after each kMC event, calculate diffusion coefficients, correlation factors, conductivity, and to summarize and save
    simulation results.
    """

    def __init__(
        self,
        config: "SimulationConfig",
        structure: StructureKMCpy,
        initial_state: Optional["SimulationState"] = None,
        **kwargs,
    ) -> None:
        """Initialize a Tracker object for monitoring mobile ion species.

        Args:
            config (SimulationConfig): Configuration object containing simulation parameters.
            structure (StructureKMCpy): Structure object.
            initial_state (SimulationState, optional): Initial simulation state.
            **kwargs: Additional legacy parameters for backward compatibility.

        """
        logger.info("Initializing Tracker ...")

        # Store configuration reference (no parameter duplication)
        self.config = config
        self.structure = structure

        # Initialize or use provided simulation state
        if initial_state is not None:
            self.state = initial_state
        else:
            raise ValueError("SimulationState must be provided to Tracker")

        # Initialize mobile ion tracking in Tracker (where it belongs)
        initial_occ = initial_state.occupations
        self._initialize_mobile_ion_tracking(initial_occ)

        # Results storage (owned by Tracker).
        self.results = _create_result_store()
        self.current_pass = 0

        # Property scheduling/registration state.
        self._global_interval: Optional[int] = 1
        self._global_time_interval: Optional[float] = None
        self._custom_properties: dict[str, PropertyRegistration] = {}
        self._custom_property_records: dict[str, list[dict[str, Any]]] = {}
        self._enabled_builtin_properties = {
            name: True for name in BUILTIN_PROPERTY_FIELDS
        }
        self._builtin_last_trigger_step: int = 0
        self._builtin_last_trigger_time: Optional[float] = None

        logger.info("number of mobile ion specie = %d", self.n_mobile_ion_specie)
        logger.info(
            f"""Center of mass ({self.mobile_ion_specie}): {np.mean(
            self.r0, axis=0
            )}"""
        )

    def _initialize_mobile_ion_tracking(self, initial_occ: list):
        """Initialize mobile ion tracking arrays."""
        # Find mobile ion sites (working version - mobile ion sites are first N sites)
        self.n_mobile_ion_specie_site = len(
            [el.symbol for el in self.structure.species if self.mobile_ion_specie in el.symbol]
        )
        self.mobile_ion_specie_locations = np.where(
            np.array(initial_occ[0:self.n_mobile_ion_specie_site]) == -1
        )[0]
        self.n_mobile_ion_specie = len(self.mobile_ion_specie_locations)

        logger.debug("Initial mobile ion locations = %s", self.mobile_ion_specie_locations)

        # Initialize tracking arrays
        self.displacement = np.zeros((self.n_mobile_ion_specie, 3))
        self.hop_counter = np.zeros(self.n_mobile_ion_specie, dtype=np.int64)
        self.r0 = self.frac_coords[self.mobile_ion_specie_locations] @ self.latt.matrix

    # Properties to access state information (delegation to state)
    @property
    def occ_initial(self) -> list:
        """Get initial occupation from state."""
        return self.state.occupations  # Current occupations

    @property
    def frac_coords(self) -> np.ndarray:
        """Get fractional coordinates from structure."""
        return self.structure.frac_coords

    @property
    def latt(self):
        """Get lattice from structure."""
        return self.structure.lattice

    @property
    def volume(self) -> float:
        """Get volume from structure."""
        return self.structure.volume

    # Properties to access configuration parameters (no duplication)
    @property
    def dimension(self) -> int:
        """Get dimension from configuration."""
        return self.config.dimension

    @property
    def q(self) -> float:
        """Get mobile ion charge from configuration."""
        return self.config.mobile_ion_charge

    @property
    def elem_hop_distance(self) -> float:
        """Get elementary hop distance from configuration."""
        return self.config.elementary_hop_distance

    @property
    def temperature(self) -> float:
        """Get temperature from configuration."""
        return self.config.temperature

    @property
    def v(self) -> float:
        """Get attempt frequency from configuration."""
        return self.config.attempt_frequency

    @property
    def time(self) -> float:
        """Get current simulation time from state."""
        return self.state.time

    @time.setter
    def time(self, value: float):
        """Set simulation time in state (for backward compatibility)."""
        self.state.time = value

    @property
    def mobile_ion_specie(self) -> str:
        """Get mobile ion species from configuration."""
        return self.config.mobile_ion_specie

    @classmethod
    def from_config(
        cls,
        config: "SimulationConfig",
        structure: StructureKMCpy,
        occ_initial: list,
    ) -> "Tracker":
        """
        Create a Tracker object from a SimulationConfig object.

        This is the preferred method for creating Tracker instances.

        Args:
            config (SimulationConfig): A SimulationConfig object containing simulation parameters.
            structure (StructureKMCpy): A StructureKMCpy object containing the structure information.
            occ_initial (list): Initial occupation list for the mobile ion specie.

        Returns:
            Tracker: An instance of the Tracker class.
        """
        # Create SimulationState with initial occupation
        from kmcpy.simulator.state import SimulationState

        initial_state = SimulationState(occupations=occ_initial)

        return cls(config=config, structure=structure, initial_state=initial_state)

    @staticmethod
    def _validate_schedule(
        interval: Optional[int],
        time_interval: Optional[float],
    ) -> None:
        """Validate callback scheduling parameters."""
        if interval is not None:
            if not isinstance(interval, int):
                raise TypeError("interval must be an integer")
            if interval <= 0:
                raise ValueError("interval must be a positive integer")

        if time_interval is not None:
            if time_interval <= 0:
                raise ValueError("time_interval must be positive")

    @staticmethod
    def _validate_max_records(max_records: Optional[int]) -> None:
        """Validate storage truncation parameter."""
        if max_records is None:
            return
        if not isinstance(max_records, int):
            raise TypeError("max_records must be an integer")
        if max_records <= 0:
            raise ValueError("max_records must be a positive integer")

    @staticmethod
    def _should_trigger(
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

    def set_global_property_frequency(
        self,
        interval: Optional[int] = None,
        time_interval: Optional[float] = None,
    ) -> None:
        """
        Set global scheduling defaults for all properties.

        Properties registered with `interval=None` and `time_interval=None` use these global defaults.

        Args:
            interval: Global step interval (every N production events).
            time_interval: Global simulation-time interval.
        """
        self._validate_schedule(interval=interval, time_interval=time_interval)
        self._global_interval = interval
        self._global_time_interval = time_interval

    def register_property(
        self,
        func: Callable[["SimulationState", int, float], Any],
        interval: Optional[int] = None,
        time_interval: Optional[float] = None,
        name: Optional[str] = None,
        store: bool = True,
        max_records: Optional[int] = None,
        on_error: Optional[Callable[[Exception, "SimulationState", int, float], bool]] = None,
    ) -> str:
        """
        Register a user-defined property callback.

        Args:
            func: Callback with signature `(state, step, time)`.
            interval: Callback-specific event interval. Uses global interval if None.
            time_interval: Callback-specific simulation-time interval. Uses global interval if None.
            name: Property name. Defaults to function `__name__`.
            store: Whether to persist callback return values.
            max_records: Keep only the most recent N values for this property.
            on_error: Error handler callback. Return True to continue simulation.

        Returns:
            str: The registered property name.
        """
        if not callable(func):
            raise TypeError("func must be callable")

        self._validate_schedule(interval=interval, time_interval=time_interval)
        self._validate_max_records(max_records)

        property_name = name or getattr(func, "__name__", "attached_property")

        if property_name in BUILTIN_PROPERTY_FIELDS:
            raise ValueError(
                f"'{property_name}' is reserved for built-in properties and cannot be reused"
            )

        if property_name in self._custom_properties:
            raise ValueError(f"Property '{property_name}' is already registered")

        registration = PropertyRegistration(
            name=property_name,
            callback=func,
            interval=interval,
            time_interval=time_interval,
            store=store,
            max_records=max_records,
            on_error=on_error,
        )
        self._custom_properties[property_name] = registration
        self._custom_property_records[property_name] = []
        return property_name

    def unregister_property(self, name: str) -> None:
        """Remove a registered user-defined property callback."""
        if name not in self._custom_properties:
            raise ValueError(f"Property '{name}' is not registered")
        del self._custom_properties[name]
        self._custom_property_records.pop(name, None)

    def clear_properties(self) -> None:
        """Remove all registered user-defined property callbacks."""
        self._custom_properties.clear()
        self._custom_property_records.clear()

    def list_properties(self) -> list[str]:
        """Return registered user-defined property names."""
        return list(self._custom_properties.keys())

    def list_builtin_properties(self) -> list[str]:
        """Return available built-in property names."""
        return list(BUILTIN_PROPERTY_FIELDS)

    def enable_builtin_property(self, name: str) -> None:
        """Enable one built-in property field in legacy results output."""
        if name not in self._enabled_builtin_properties:
            raise ValueError(f"Unknown built-in property '{name}'")
        self._enabled_builtin_properties[name] = True

    def disable_builtin_property(self, name: str) -> None:
        """Disable one built-in property field in legacy results output."""
        if name not in self._enabled_builtin_properties:
            raise ValueError(f"Unknown built-in property '{name}'")
        self._enabled_builtin_properties[name] = False

    def _run_property_error_handler(
        self,
        registration: PropertyRegistration,
        exc: Exception,
        step: int,
        sim_time: float,
    ) -> None:
        """Execute user-specified error handler and raise on unhandled callback failures."""
        if registration.on_error is None:
            raise CallbackExecutionError(
                f"Property callback '{registration.name}' failed at step={step}, time={sim_time}"
            ) from exc

        try:
            keep_running = bool(registration.on_error(exc, self.state, step, sim_time))
        except Exception as handler_exc:
            raise CallbackExecutionError(
                f"Error handler for callback '{registration.name}' failed"
            ) from handler_exc

        if not keep_running:
            raise CallbackExecutionError(
                f"Property callback '{registration.name}' failed and requested termination"
            ) from exc

    def _append_custom_record(self, registration: PropertyRegistration, step: int, sim_time: float, value: Any) -> None:
        """Append one custom callback result record and enforce record limit."""
        record = {
            "name": registration.name,
            "step": int(step),
            "time": float(sim_time),
            "value": value,
        }
        records = self._custom_property_records[registration.name]
        records.append(record)
        if registration.max_records is not None and len(records) > registration.max_records:
            del records[0 : len(records) - registration.max_records]

    def _compute_builtin_values(self) -> dict[str, float]:
        """Compute all built-in properties, returning NaN for disabled fields."""
        nan = float("nan")
        enabled = self._enabled_builtin_properties

        any_dj_needed = enabled["D_J"] or enabled["conductivity"] or enabled["H_R"]
        any_dtracer_needed = enabled["D_tracer"] or enabled["H_R"]

        D_J_internal = self.calc_D_J() if any_dj_needed else nan
        D_tracer_internal = self.calc_D_tracer() if any_dtracer_needed else nan

        msd = np.mean(np.linalg.norm(self.displacement, axis=1) ** 2) if enabled["msd"] else nan
        D_J = D_J_internal if enabled["D_J"] else nan
        D_tracer = D_tracer_internal if enabled["D_tracer"] else nan

        conductivity = nan
        if enabled["conductivity"] and np.isfinite(D_J_internal):
            conductivity = self.calc_conductivity(D_J=D_J_internal)

        H_R = nan
        if enabled["H_R"] and np.isfinite(D_J_internal) and D_J_internal != 0:
            H_R = D_tracer_internal / D_J_internal

        f = self.calc_corr_factor() if enabled["f"] else nan

        return {
            "msd": msd,
            "D_J": D_J,
            "D_tracer": D_tracer,
            "conductivity": conductivity,
            "H_R": H_R,
            "f": f,
        }

    def _append_builtin_results(self, step: int, sim_time: float) -> None:
        """Compute and append one built-in result row."""
        builtins = self._compute_builtin_values()
        _append_result(
            self.results,
            copy(sim_time),
            builtins["D_J"],
            builtins["D_tracer"],
            builtins["conductivity"],
            builtins["f"],
            builtins["H_R"],
            builtins["msd"],
        )
        self._builtin_last_trigger_step = step
        self._builtin_last_trigger_time = sim_time

        if logger.isEnabledFor(logging.DEBUG):
            summary_data = [
                ["Time elapsed", sim_time],
                ["Current pass", self.current_pass],
                ["Temperature (K)", self.temperature],
                ["Attempt frequency (v)", self.v],
                [
                    f"{self.mobile_ion_specie} ratio ({self.mobile_ion_specie}/({self.mobile_ion_specie}+Va))",
                    self.n_mobile_ion_specie / self.n_mobile_ion_specie_site,
                ],
                ["Haven's ratio H_R", builtins["H_R"]],
            ]
            table_str = "\n" + pd.DataFrame(summary_data, columns=["Property", "Value"]).to_string(index=False)
            logger.debug("Tracker Summary:%s", table_str)

    def maybe_compute_properties(self, step: int, sim_time: float) -> None:
        """
        Evaluate property schedules and execute built-in/user-defined property callbacks.

        Args:
            step: Current production step index.
            sim_time: Current simulation time.
        """
        builtin_interval = self._global_interval
        builtin_time_interval = self._global_time_interval

        if self._should_trigger(
            step=step,
            sim_time=sim_time,
            interval=builtin_interval,
            time_interval=builtin_time_interval,
            last_trigger_time=self._builtin_last_trigger_time,
        ):
            self._append_builtin_results(step=step, sim_time=sim_time)

        for registration in list(self._custom_properties.values()):
            interval = registration.interval if registration.interval is not None else self._global_interval
            time_interval = (
                registration.time_interval
                if registration.time_interval is not None
                else self._global_time_interval
            )

            if not self._should_trigger(
                step=step,
                sim_time=sim_time,
                interval=interval,
                time_interval=time_interval,
                last_trigger_time=registration.last_trigger_time,
            ):
                continue

            try:
                value = registration.callback(self.state, step, sim_time)
            except Exception as exc:
                self._run_property_error_handler(
                    registration=registration,
                    exc=exc,
                    step=step,
                    sim_time=sim_time,
                )
                registration.last_trigger_step = step
                registration.last_trigger_time = sim_time
                continue

            if registration.store:
                self._append_custom_record(
                    registration=registration,
                    step=step,
                    sim_time=sim_time,
                    value=value,
                )

            registration.last_trigger_step = step
            registration.last_trigger_time = sim_time

    def get_custom_results(self, name: Optional[str] = None) -> dict[str, list[dict[str, Any]]] | list[dict[str, Any]]:
        """
        Retrieve stored user-defined property records.

        Args:
            name: Optional callback name. If omitted, return all callbacks.

        Returns:
            Stored callback records for one callback or all callbacks.
        """
        if name is not None:
            if name not in self._custom_property_records:
                raise ValueError(f"Property '{name}' has no stored records")
            return list(self._custom_property_records[name])

        return {
            key: list(records)
            for key, records in self._custom_property_records.items()
        }

    def update(self, event, current_occ, dt) -> None:
        """
        Update tracker observables for a proposed kMC event.

        This method should be called with the pre-event occupation snapshot.
        Tracker updates trajectory observables (positions, displacements, hops),
        while simulation time/step are owned by KMC via SimulationState.

        Args:
            event: An object representing the KMC event, containing indices and properties of the mobile ions involved.
            current_occ (np.ndarray): The current occupation array indicating the occupation state of each site.
            dt (float): Time increment for this event (kept for API compatibility).

        Side Effects:
            - Updates the internal state of the tracker, including:
                - `mobile_ion_specie_locations`: The indices of the mobile ions after the event.
                - `displacement`: The cumulative displacement of each mobile ion.
                - `hop_counter`: The number of hops performed by each mobile ion.
            - Logs detailed debug information if the logger is set to DEBUG level.

        Raises:
            Logs an error if the event direction cannot be determined (i.e., if the event is invalid).
        """
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
            logger.debug("Mobile ion locations before update: %s", self.mobile_ion_specie_locations)
            logger.debug(
                "Occupation before update: %s",
                np.array2string(current_occ, precision=0, separator=", "),
            )
        direction = int((mobile_ion_specie_2_occ - mobile_ion_specie_1_occ) / 2)
        displacement_frac = copy(direction * (mobile_ion_specie_2_coord - mobile_ion_specie_1_coord))
        displacement_frac -= np.array([int(round(i)) for i in displacement_frac])
        displacement_cart = copy(self.latt.get_cartesian_coords(displacement_frac))
        if direction == -1:  # Na(2) -> Na(1)
            logger.debug(
                "Diffuse direction: %s(2) -> %s(1)",
                self.mobile_ion_specie,
                self.mobile_ion_specie,
            )
            specie_to_diff = np.where(
                self.mobile_ion_specie_locations == event.mobile_ion_indices[1]
            )[0][0]
            self.mobile_ion_specie_locations[specie_to_diff] = event.mobile_ion_indices[0]
        elif direction == 1:  # Na(1) -> Na(2)
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
        """
        Update the current pass number for the tracker.

        Args:
            current_pass (int): The new current pass number.
        """
        self.current_pass = current_pass

    def calc_D_J(self) -> float:
        """
        Calculate the jump diffusivity (D_J) based on the total displacement vector.

        Returns:
            float: The calculated jump diffusivity (D_J) in cm^2/s.
        """
        displacement_vector_tot = np.linalg.norm(np.sum(self.displacement, axis=0))

        D_J = (
            displacement_vector_tot**2
            / (2 * self.dimension * self.time * self.n_mobile_ion_specie)
            * 10 ** (-16)
        )  # to cm^2/s

        return D_J

    def calc_D_tracer(self) -> float:
        """
        Calculate the tracer diffusivity (D_tracer).

        Returns:
            float: The calculated tracer diffusivity.
        """
        D_tracer = (
            np.mean(np.linalg.norm(self.displacement, axis=1) ** 2)
            / (2 * self.dimension * self.time)
            * 10 ** (-16)
        )

        return D_tracer

    def calc_corr_factor(self) -> float:  # a is the hopping distance in Angstrom
        """
        Calculate the correlation factor for the tracked hops.

        Returns:
            float: The mean correlation factor for the tracked hops.
        """
        hop_counter_safe = np.where(self.hop_counter == 0, 1, self.hop_counter)
        corr_factor = np.linalg.norm(self.displacement, axis=1) ** 2 / (
            hop_counter_safe * self.elem_hop_distance**2
        )
        corr_factor[self.hop_counter == 0] = 0

        return np.mean(corr_factor)

    def calc_conductivity(self, D_J) -> float:
        """
        Calculate the ionic conductivity based on the jump diffusivity.

        Args:
            D_J (float): Jump diffusivity in units of cm^2/s.

        Returns:
            float: Ionic conductivity in mS/cm.
        """
        k = 8.617333262145 * 10 ** (-2)  # unit in meV/K

        n = self.n_mobile_ion_specie / self.volume
        conductivity = D_J * n * self.q**2 / (k * self.temperature) * 1.602 * 10**11

        return conductivity

    def show_current_info(self) -> None:
        """
        Log current simulation information and latest computed built-in properties.
        """
        if not self.results["time"]:
            logger.info(
                "%d\t%.3E\t%s\t%s\t%s\t%s\t%s\t%s",
                self.current_pass,
                self.time,
                "nan",
                "nan",
                "nan",
                "nan",
                "nan",
                "nan",
            )
            return

        logger.info(
            "%d\t%.3E\t%.3E\t%.3E\t%.3E\t%.3E\t%.3E\t%.3E",
            self.current_pass,
            self.results["time"][-1],
            self.results["msd"][-1],
            self.results["D_J"][-1],
            self.results["D_tracer"][-1],
            self.results["conductivity"][-1],
            self.results["H_R"][-1],
            self.results["f"][-1],
        )
        logger.debug(
            "Center of mass (%s): %s",
            self.mobile_ion_specie,
            np.mean(self.frac_coords[self.mobile_ion_specie_locations] @ self.latt.matrix, axis=0),
        )
        logger.debug(
            "MSD = %s, time = %s",
            np.linalg.norm(np.sum(self.displacement, axis=0)) ** 2,
            self.time,
        )

    def return_current_info(self) -> tuple:
        """
        Returns the latest built-in simulation properties as a tuple for unit testing.
        """
        if not self.results["time"]:
            raise ValueError("No property samples are available. Increase sampling frequency.")

        return (
            self.results["time"][-1],
            self.results["msd"][-1],
            self.results["D_J"][-1],
            self.results["D_tracer"][-1],
            self.results["conductivity"][-1],
            self.results["H_R"][-1],
            self.results["f"][-1],
        )

    def compute_properties(self) -> None:
        """
        Compute one immediate built-in property sample.

        This compatibility method ignores scheduling and appends one built-in result row
        at the current state time and step.
        """
        self._append_builtin_results(step=int(self.state.step), sim_time=float(self.time))

    def write_results(self, current_occupation: list, label: str = None) -> None:
        """
        Save simulation results to compressed files.

        Args:
            current_occupation (list): The current occupation state to be saved.
            label (str, optional): An optional label to prefix output files.

        Saves:
            displacement_{label}_{current_pass}.csv.gz (ndarray)
            hop_counter_{label}_{current_pass}.csv.gz (ndarray)
            current_occ_{label}_{current_pass}.csv.gz (list)
            results_{label}.csv.gz or results.csv.gz (DataFrame)
            custom_results_{label}.json.gz or custom_results.json.gz (callback data)
        """
        prefix = f"{label}_{self.current_pass}"
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
            custom_results_file = f"custom_results_{label}.json.gz"
        else:
            results_file = "results.csv.gz"
            custom_results_file = "custom_results.json.gz"
        pd.DataFrame(self.results).to_csv(results_file, compression="gzip", index=False)

        flat_custom_records = []
        for records in self._custom_property_records.values():
            for record in records:
                flat_custom_records.append(
                    {
                        "name": record["name"],
                        "step": int(record["step"]),
                        "time": float(record["time"]),
                        "value": _to_json_safe(record["value"]),
                    }
                )

        if flat_custom_records:
            with gzip.open(custom_results_file, "wt", encoding="utf-8") as fhandle:
                json.dump(flat_custom_records, fhandle, indent=2)

    def as_dict(self) -> dict:
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "T": self.temperature,
            "occ_initial": self.occ_initial,
            "frac_coords": self.frac_coords,
            "latt": self.latt.as_dict(),
            "volume": self.volume,
            "n_mobile_ion_specie_site": self.n_mobile_ion_specie_site,
            "mobile_ion_specie_locations": self.mobile_ion_specie_locations,
            "n_mobile_ion_specie": self.n_mobile_ion_specie,
            "displacement": self.displacement,
            "hop_counter": self.hop_counter,
            "time": self.time,
            "results": self.results,
            "custom_results": self.get_custom_results(),
            "r0": self.r0,
        }
        return d

    def to_json(self, fname) -> None:
        logger.info("Saving: %s", fname)
        with open(fname, "w") as fhandle:
            d = self.as_dict()
            jsonStr = json.dumps(
                d,
                indent=4,
                default=convert,
            )  # to get rid of errors of int64
            fhandle.write(jsonStr)

    @classmethod
    def from_json(self, fname) -> "Tracker":
        logger.info("Loading: %s", fname)
        with open(fname, "rb") as fhandle:
            objDict = json.load(fhandle)
        obj = Tracker()
        obj.__dict__ = objDict
        return obj
