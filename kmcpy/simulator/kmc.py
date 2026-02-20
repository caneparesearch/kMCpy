#!/usr/bin/env python
"""
This module provides the KMC class and associated functions for performing Kinetic Monte Carlo (kMC) simulations, 
particularly for modeling processes in materials such as ion diffusion. The KMC class manages the 
initialization, event handling, probability calculations, and simulation loop for kMC workflows. It supports 
loading input data from various sources, updating system states, and tracking simulation results.
"""
from numba import njit
from kmcpy.external.structure import StructureKMCpy
import numpy as np
import json
from kmcpy.simulator.tracker import (
    BUILTIN_PROPERTY_FIELDS,
    CallbackExecutionError,
    Tracker,
)
from kmcpy.simulator.property_engine import validate_max_records, validate_schedule
from kmcpy.event import Event, EventLib
from kmcpy.io import convert
import logging
import kmcpy
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from kmcpy.simulator.config import SimulationConfig
    from kmcpy.simulator.state import SimulationState
    from kmcpy.models.composite_lce_model import CompositeLCEModel

logger = logging.getLogger(__name__) 
logging.getLogger('numba').setLevel(logging.WARNING)

class KMC:
    """Kinetic Monte Carlo Simulation Class.

    This class implements a Kinetic Monte Carlo (kMC) simulation for modeling
    stochastic processes in materials, such as ion diffusion. It provides
    methods for initializing the simulation from various input sources,
    managing events, updating system states, and running the simulation loop.
    """
    def __init__(self, 
                structure: StructureKMCpy,
                model: 'CompositeLCEModel',
                event_lib: EventLib,
                config: "SimulationConfig",
                simulation_state: "SimulationState" = None,
                **kwargs) -> None:
        """Initialize the Kinetic Monte Carlo (kMC) simulation.

        Args:
            structure (StructureKMCpy): The structure object (already processed).
            model (CompositeLCEModel): The model with all parameters loaded.
            event_lib (EventLib): The event library with all events loaded.
            config (SimulationConfig): Configuration object containing all simulation parameters.
            simulation_state (SimulationState, optional): SimulationState object for state management.
            **kwargs: Additional keyword arguments.

        Note:
            SimulationState is the preferred way to manage mutable simulation state.
            The config object contains all immutable configuration parameters.
        """
        logger.info(kmcpy.get_logo())
        logger.info(f"Initializing kMC calculations ...")

        self._ensure_property_state()
        
        # Store configuration
        self.config = config
        
        # Store simulation state reference
        self.simulation_state = simulation_state
        
        # Store the already loaded structure
        self.structure = structure
        
        # Store the composite model (already loaded and configured)
        self.model = model
        
        # Store the event library (already loaded)
        self.event_lib = event_lib
        
        # Initialize occupation state from simulation_state.
        if simulation_state is not None:
            # Use SimulationState as single source of truth
            logger.info("Using SimulationState for occupation management")
            logger.debug(f"SimulationState occupations length: {len(simulation_state.occupations)}")
            self.occ_global = simulation_state.occupations
        else:
            raise ValueError("SimulationState must be provided for clean architecture")

        # Calculate initial probabilities from runtime configuration and state.
        logger.info("Initializing probabilities...")
        
        # Calculate probabilities for all events using composite model
        self.prob_list = np.empty(len(self.event_lib), dtype=np.float64)
        for i, event in enumerate(self.event_lib.events):
            # Update the occupation in simulation_state for each event calculation
            self.simulation_state.occupations = self.occ_global.copy()

            self.prob_list[i] = self.model.compute_probability(
                event=event,
                runtime_config=self.config.runtime_config,
                simulation_state=self.simulation_state
            )
        
        self.prob_cum_list = np.empty(len(self.event_lib), dtype=np.float64)
        np.cumsum(self.prob_list, out=self.prob_cum_list)
        
        logger.info(f"Event dependency matrix with {len(self.event_lib)} events")
        logger.info(f"Hopping probabilities: {self.prob_list}")
        logger.info(f"Cumulative sum of hopping probabilities: {self.prob_cum_list}")
        
        # Display dependency matrix statistics
        stats = self.event_lib.get_dependency_statistics()
        logger.info(f"Dependency matrix statistics: {stats}")

        logger.info("kMC initialization complete!")

    def _ensure_property_state(self) -> None:
        """Initialize KMC property attachment state if missing."""
        if not hasattr(self, "_attached_properties"):
            self._attached_properties: dict[str, dict[str, Any]] = {}
        if not hasattr(self, "_property_frequency_interval"):
            self._property_frequency_interval: Optional[int] = None
        if not hasattr(self, "_property_frequency_time_interval"):
            self._property_frequency_time_interval: Optional[float] = None
        if not hasattr(self, "_property_enabled"):
            self._property_enabled = {
                name: True for name in BUILTIN_PROPERTY_FIELDS
            }
        if not hasattr(self, "_active_tracker"):
            self._active_tracker: Optional[Tracker] = None
        
    @classmethod
    def from_config(cls, config: "SimulationConfig") -> "KMC":
        """Create KMC instance from SimulationConfig (recommended initialization method).

        This is the main initialization method that leverages SimulationConfigIO
        for all component loading operations and provides a clean interface.

        Args:
            config (SimulationConfig): Configuration object containing all necessary parameters.

        Returns:
            KMC: An instance of the KMC class.
        """
        # Use centralized component loading from SimulationConfigIO
        from kmcpy.io.config_io import SimulationConfigIO
        
        structure, model, event_lib, simulation_state = SimulationConfigIO.load_simulation_components(config)
        
        return cls(
            structure=structure,
            model=model,
            event_lib=event_lib,
            config=config,
            simulation_state=simulation_state
        )

    def show_project_info(self):
        """Log current probability vectors for quick diagnostics."""
        try:
            logger.info("Probabilities:")
            logger.info(self.prob_list)
            logger.info("Cumultative probability list:")
            logger.info(self.prob_cum_list / sum(self.prob_list))
        except Exception:
            pass

    def set_property_frequency(
        self,
        interval: Optional[int] = None,
        time_interval: Optional[float] = None,
    ) -> None:
        """
        Set global sampling frequency for built-in and attached properties.

        Args:
            interval: Global event-step interval.
            time_interval: Global simulation-time interval.
        """
        self._ensure_property_state()
        validate_schedule(interval=interval, time_interval=time_interval)
        self._property_frequency_interval = interval
        self._property_frequency_time_interval = time_interval

        if self._active_tracker is not None:
            self._active_tracker.set_global_property_frequency(
                interval=interval,
                time_interval=time_interval,
            )

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
        """
        Attach a custom property callback to the KMC run.

        Args:
            func: Callback with signature `(state, step, time)`.
            interval: Event-step interval override for this callback.
            time_interval: Simulation-time interval override for this callback.
            name: Optional callback name; defaults to function `__name__`.
            store: Whether callback return values should be stored.
            max_records: Optional cap on stored record count for this callback.
            on_error: Optional error handler. Return True to continue.

        Returns:
            str: The callback name used for registration.
        """
        self._ensure_property_state()
        if not callable(func):
            raise TypeError("func must be callable")
        validate_schedule(interval=interval, time_interval=time_interval)
        validate_max_records(max_records=max_records)

        callback_name = name or getattr(func, "__name__", "attached_property")
        if callback_name in BUILTIN_PROPERTY_FIELDS:
            raise ValueError(
                f"'{callback_name}' is reserved for built-in properties"
            )
        if callback_name in self._attached_properties:
            raise ValueError(f"Callback '{callback_name}' is already attached")

        attachment = {
            "func": func,
            "interval": interval,
            "time_interval": time_interval,
            "name": callback_name,
            "store": store,
            "max_records": max_records,
            "on_error": on_error,
            "enabled": enabled,
        }
        self._attached_properties[callback_name] = attachment
        self._property_enabled[callback_name] = bool(enabled)

        if self._active_tracker is not None:
            self._active_tracker.attach(**attachment)

        return callback_name

    def detach(self, name: str) -> None:
        """Detach a previously attached custom property callback."""
        self._ensure_property_state()
        if name not in self._attached_properties:
            raise ValueError(f"Callback '{name}' is not attached")
        del self._attached_properties[name]
        self._property_enabled.pop(name, None)

        if self._active_tracker is not None:
            self._active_tracker.detach(name)

    def clear_attachments(self) -> None:
        """Remove all attached custom property callbacks."""
        self._ensure_property_state()
        self._attached_properties.clear()
        self._property_enabled = {
            key: value
            for key, value in self._property_enabled.items()
            if key in BUILTIN_PROPERTY_FIELDS
        }
        if self._active_tracker is not None:
            self._active_tracker.clear_attachments()

    def list_attachments(self) -> list[str]:
        """Return names of currently attached custom property callbacks."""
        self._ensure_property_state()
        return list(self._attached_properties.keys())

    def set_property_enabled(self, name: str, enabled: bool) -> None:
        """Enable or disable a built-in summary field or attached callback."""
        self._ensure_property_state()

        if name not in self._property_enabled and name not in self._attached_properties:
            raise ValueError(f"Unknown property '{name}'")

        self._property_enabled[name] = bool(enabled)
        if name in self._attached_properties:
            self._attached_properties[name]["enabled"] = bool(enabled)

        if self._active_tracker is not None:
            self._active_tracker.set_property_enabled(name, bool(enabled))

    def _configure_tracker_properties(self, tracker: Tracker, pass_length: int) -> None:
        """Apply KMC-side property settings and attachments to a tracker instance."""
        self._ensure_property_state()

        global_interval = self._property_frequency_interval
        global_time_interval = self._property_frequency_time_interval

        # Preserve previous behavior: built-ins sampled once per pass by default.
        if global_interval is None and global_time_interval is None:
            global_interval = max(pass_length, 1)

        tracker.set_global_property_frequency(
            interval=global_interval,
            time_interval=global_time_interval,
        )

        for property_name, enabled in self._property_enabled.items():
            if property_name in BUILTIN_PROPERTY_FIELDS:
                tracker.set_property_enabled(property_name, enabled)

        for attachment in self._attached_properties.values():
            tracker.attach(**attachment)

    def propose(
        self,
        events: list,
    ) -> tuple[Event, float, int]:
        """Propose a new event to be updated by update().

        Args:
            events (list): List of events.

        Returns:
            tuple[Event, float, int]: The chosen event, the time for this event to occur, and the event index.
        """
        proposed_event_index, dt = _propose(prob_cum_list=self.prob_cum_list, rng=self.rng)
        event = events[proposed_event_index]
        return event, dt
    

    def update(self, event: Event, dt: float = 0.0) -> None:
        """
        Updates the system state and event probabilities after an event occurs.
        
        This method delegates state management to SimulationState, following clean
        architecture principles with single responsibility and separation of concerns.
        
        This method performs the following steps:
        1. Delegates occupation updates to SimulationState.apply_event()
        2. Automatically finds the event index in the event library
        3. Identifies all events that need probability updates using EventLib
        4. Recalculates probabilities for affected events
        5. Updates the cumulative probability list for event selection

        Args:
            event: The event object that has just occurred.
            dt (float, optional): Time increment for this event. Used for state tracking.
            
        Side Effects:
            Modifies occupation state and probability lists via SimulationState delegation.
        """
        # Delegate state update to SimulationState - clean architecture with single state object
        self.simulation_state.apply_event(event, dt)
        
        # Synchronize occupation reference for probability calculations
        self.occ_global = self.simulation_state.occupations
        
        # Find event index automatically from event library
        event_index = self.event_lib.events.index(event)
        
        # Use EventLib to get dependent events
        events_to_be_updated = self.event_lib.get_dependent_events(event_index)
        
        # Update probabilities for dependent events using composite model
        for e_index in events_to_be_updated:
            # Use single simulation_state for probability calculations
            self.simulation_state.occupations = self.occ_global
            
            # Recalculate probability using composite model
            self.prob_list[e_index] = self.model.compute_probability(
                event=self.event_lib.events[e_index],
                runtime_config=self.config.runtime_config,
                simulation_state=self.simulation_state
            )
        self.prob_cum_list = np.cumsum(self.prob_list)

    def run(self, config: "SimulationConfig", label: str = None) -> Tracker:
        """Run KMC simulation from a SimulationConfig object.

        This is the main method for running KMC simulations using the modern
        SimulationConfig format.

        Args:
            config (SimulationConfig): Configuration object containing all necessary parameters.
            label (str, optional): Label for the simulation run. Defaults to None.
                If None, will use config.name.

        Returns:
            kmcpy.tracker.Tracker: Tracker object containing simulation results.
            
        Example::
        
            # Using SimulationConfig
            config = SimulationConfig.create(name="Test", temperature=400.0, ...)
            tracker = kmc.run(config)
            
            # Alternative usage patterns:
            # 1. Create KMC and run in one step
            kmc = KMC.from_config(config)
            tracker = kmc.run(config)
        """
        # Set label from config if not provided
        if label is None:
            label = config.name
        
        # Initialize random number generator
        if config.random_seed is not None:
            self.rng = np.random.default_rng(seed=config.random_seed)
        else:
            self.rng = np.random.default_rng()

        logger.info(
            "Runtime config: v = %s T = %s",
            config.attempt_frequency,
            config.temperature,
        )
        
        # Calculate pass length based on mobile ions
        pass_length = len([
            el.symbol
            for el in self.structure.species
            if config.mobile_ion_specie in el.symbol
        ])
        
        logger.info("============================================================")
        logger.info("Start running kMC ... ")
        logger.info("Initial occ_global, prob_list and prob_cum_list")
        logger.info("Starting Equilibrium ...")
        
        # Equilibration phase
        for _ in np.arange(config.equilibration_passes):
            for _ in np.arange(pass_length):
                event, dt = self.propose(self.event_lib.events)
                # Keep equilibration out of production time accounting.
                self.update(event, dt=0.0)

        logger.info("Start running kMC ...")

        # Create Tracker using clean SimulationState architecture
        tracker = Tracker(config=config, structure=self.structure, initial_state=self.simulation_state)
        self._configure_tracker_properties(tracker=tracker, pass_length=pass_length)
        self._active_tracker = tracker
        logger.info("Using clean SimulationState architecture")
        
        logger.info("Tracker summaries are reported as dynamic property tables per pass.")

        # Main KMC loop
        for current_pass in np.arange(config.kmc_passes):
            for _ in np.arange(pass_length):
                event, dt = self.propose(self.event_lib.events)
                
                # Tracker observes the pre-event occupation snapshot.
                current_occupations = self.simulation_state.occupations
                tracker.update(event, current_occupations, dt)
                # KMC is the single owner of mutable simulation state updates.
                self.update(event, dt=dt)
                tracker.sample_properties(
                    step=int(self.simulation_state.step),
                    sim_time=float(self.simulation_state.time),
                )
            
            tracker.update_current_pass(current_pass)
            tracker.show_current_info()

        # Use SimulationState occupations for final output
        final_occupations = self.simulation_state.occupations
        tracker.write_results(final_occupations, label=label)
        return tracker

    def as_dict(self)-> dict:
        """Serialize KMC object state to dictionary form."""
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "structure": self.structure.as_dict(),
            "model": self.model.as_dict() if hasattr(self.model, 'as_dict') else str(self.model),
            "config": self.config.as_dict() if hasattr(self.config, 'as_dict') else str(self.config),
            "occ_global": self.occ_global,
            "event_lib": self.event_lib.as_dict(),
        }
        return d

    def to_json(self, fname)-> None:
        """Write serialized KMC state to a JSON file."""
        logger.info(f"Saving: {fname}")
        with open(fname, "w") as fhandle:
            d = self.as_dict()
            jsonStr = json.dumps(
                d, indent=4, default=convert
            )  # to get rid of errors of int64
            fhandle.write(jsonStr)

    @classmethod
    def from_json(cls, fname)-> "KMC":
        """Load a serialized KMC object state from JSON."""
        logger.info(f"Loading: {fname}")
        with open(fname, "rb") as fhandle:
            objDict = json.load(fhandle)
        obj = KMC()
        obj.__dict__ = objDict
        logger.info("load complete")
        return obj

@njit
def _propose(prob_cum_list, rng)-> tuple[int, float]:
    """Sample one event index and waiting time from cumulative rates."""
    random_seed = rng.random()
    random_seed_2 = rng.random()
    proposed_event_index = np.searchsorted(
        prob_cum_list / prob_cum_list[-1], random_seed, side="right"
    )
    dt = (-1.0 / prob_cum_list[-1]) * np.log(random_seed_2)
    return proposed_event_index, dt
