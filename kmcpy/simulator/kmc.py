#!/usr/bin/env python
"""
This module provides the KMC class and associated functions for performing Kinetic Monte Carlo (kMC) simulations, 
particularly for modeling processes in materials such as ion diffusion. The KMC class manages the 
initialization, event handling, probability calculations, and simulation loop for kMC workflows. It supports 
loading input data from various sources, updating system states, and tracking simulation results.
"""
from numba import njit
from pymatgen.core import Structure
import numpy as np
import importlib
import inspect
from kmcpy.simulator.tracker import (
    CallbackExecutionError,
    Tracker,
)
from kmcpy.simulator.hop import HopStateLookup
from kmcpy.simulator.property import PropertyPlan
from kmcpy.event import Event, EventLib
import logging
import kmcpy
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from kmcpy.simulator.config import Configuration
    from kmcpy.simulator.state import State
    from kmcpy.models.base import BaseModel

logger = logging.getLogger(__name__) 
logging.getLogger('numba').setLevel(logging.WARNING)


def _accepts_keyword(callable_obj, keyword: str) -> bool:
    """Return whether a callable accepts a specific keyword argument."""
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False
    return keyword in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


class KMC:
    """Kinetic Monte Carlo Simulation Class.

    This class implements a Kinetic Monte Carlo (kMC) simulation for modeling
    stochastic processes in materials, such as ion diffusion. It provides
    methods for initializing the simulation from various input sources,
    managing events, updating system states, and running the simulation loop.
    """
    def __init__(self, 
                structure: Structure,
                model: 'BaseModel',
                event_lib: EventLib,
                config: "Configuration",
                simulation_state: "State" = None,
                hop_state_lookup: HopStateLookup | None = None,
                active_site_order=None,
                **kwargs) -> None:
        """Initialize the Kinetic Monte Carlo (kMC) simulation.

        Args:
            structure (Structure): The structure object (already processed).
            model (BaseModel): The loaded model implementing compute_probability(...).
            event_lib (EventLib): The event library with all events loaded.
            config (Configuration): Configuration object containing all simulation parameters.
            simulation_state (State, optional): State object for state management.
            **kwargs: Additional keyword arguments.

        Note:
            State is the preferred way to manage mutable simulation state.
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
        
        # Store the pre-configured model
        self.model = model
        
        # Store the event library (already loaded)
        self.event_lib = event_lib

        # Store and apply precomputed hop-state metadata for fast direction checks.
        self.hop_state_lookup = hop_state_lookup
        self.active_site_order = active_site_order
        if self.hop_state_lookup is not None:
            self.hop_state_lookup.annotate_event_lib(self.event_lib)
        
        if simulation_state is not None:
            logger.info("Using State for occupation management")
            logger.debug(f"State occupations length: {len(simulation_state.occupations)}")
        else:
            raise ValueError("State must be provided for clean architecture")

        self._initialize_model_state()

        # Calculate initial probabilities from runtime configuration and state.
        logger.info("Initializing probabilities...")
        
        # Calculate probabilities for all events using configured model
        self.prob_list = np.empty(len(self.event_lib), dtype=np.float64)
        for i, event in enumerate(self.event_lib.events):
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

        # Apply property controls provided by RuntimeConfig/YAML.
        self._configure_properties_from_runtime_config(self.config.runtime_config)

        logger.info("kMC initialization complete!")

    def _initialize_model_state(self) -> None:
        """Let stateful models initialize external occupancy/cache state once."""
        initialize_state = getattr(self.model, "initialize_state", None)
        if callable(initialize_state):
            kwargs = {
                "simulation_state": self.simulation_state,
                "event_lib": self.event_lib,
                "structure": self.structure,
                "config": self.config,
            }
            if self.active_site_order is not None and _accepts_keyword(
                initialize_state,
                "active_site_order",
            ):
                kwargs["active_site_order"] = self.active_site_order
            initialize_state(**kwargs)

    def _apply_model_event(self, event: Event) -> None:
        """Let stateful models commit an accepted event after State updates."""
        apply_event = getattr(self.model, "apply_event", None)
        if callable(apply_event):
            apply_event(event=event, simulation_state=self.simulation_state)

    def _ensure_property_state(self) -> None:
        """Initialize the KMC property plan if missing."""
        if not hasattr(self, "property_plan"):
            self.property_plan = PropertyPlan()
        if not hasattr(self, "_active_tracker"):
            self._active_tracker: Optional[Tracker] = None

    @staticmethod
    def _resolve_callback_reference(callable_ref: str) -> Callable[["State", int, float], Any]:
        """Resolve callback path strings like `module.path:func` or `module.path.func`."""
        if ":" in callable_ref:
            module_path, attr_path = callable_ref.split(":", 1)
        else:
            module_path, _, attr_path = callable_ref.rpartition(".")
            if not module_path:
                raise ValueError(
                    f"Invalid callback reference '{callable_ref}'. "
                    "Use 'package.module:function' or 'package.module.function'."
                )

        module = importlib.import_module(module_path)
        callback_obj: Any = module
        for attr in attr_path.split("."):
            callback_obj = getattr(callback_obj, attr)

        if not callable(callback_obj):
            raise TypeError(f"Resolved callback '{callable_ref}' is not callable")
        return callback_obj

    def _configure_properties_from_runtime_config(self, runtime_config: Any) -> None:
        """Apply runtime property controls from configuration."""
        self._ensure_property_state()

        interval = getattr(runtime_config, "property_sampling_interval", None)
        time_interval = getattr(runtime_config, "property_sampling_time_interval", None)
        if interval is not None or time_interval is not None:
            self.set_property_frequency(interval=interval, time_interval=time_interval)

        for property_name, enabled in getattr(runtime_config, "builtin_property_enabled", {}).items():
            self.set_property_enabled(property_name, bool(enabled))

        for callback_spec in getattr(runtime_config, "property_callbacks", []):
            callback_ref = callback_spec["callable"]
            callback_func = self._resolve_callback_reference(callback_ref)
            self.attach(
                callback_func,
                interval=callback_spec.get("interval"),
                time_interval=callback_spec.get("time_interval"),
                name=callback_spec.get("name"),
                store=callback_spec.get("store", True),
                max_records=callback_spec.get("max_records"),
                enabled=callback_spec.get("enabled", True),
            )
        
    @classmethod
    def from_config(cls, config: "Configuration") -> "KMC":
        """Create a KMC instance from a Configuration."""
        from kmcpy.io.cif import load_labeled_structure_from_cif
        from kmcpy.models.base import BaseModel
        from kmcpy.simulator.state import State
        from kmcpy.structure.active_site_order import ActiveSiteOrder

        if config.site_mapping is None:
            raise ValueError(
                "site_mapping is required so kMC state, events, and model "
                "data use the same active-site index space."
            )

        full_structure = load_labeled_structure_from_cif(
            config.structure_file,
            primitive=config.convert_to_primitive_cell,
        )
        active_site_order = ActiveSiteOrder.from_structure_and_mapping(
            full_structure,
            config.site_mapping,
            supercell_shape=config.supercell_shape,
        )
        structure = active_site_order.active_structure()

        model = BaseModel.from_config(config)
        event_lib = EventLib.from_file(config.event_file)
        event_lib.validate_index_metadata(active_site_order)
        hop_state_lookup = HopStateLookup.from_active_site_order(
            active_site_order,
            config.mobile_ion_specie,
        )

        if config.initial_occupations is not None:
            simulation_state = State.from_occupations(
                config.initial_occupations,
                active_site_order=active_site_order,
            )
        elif config.initial_state_file:
            simulation_state = State.from_file(
                config.initial_state_file,
                supercell_shape=config.supercell_shape,
                active_site_order=active_site_order,
            )
        else:
            raise ValueError("Initial occupations could not be determined.")

        return cls(
            structure=structure,
            model=model,
            event_lib=event_lib,
            config=config,
            simulation_state=simulation_state,
            hop_state_lookup=hop_state_lookup,
            active_site_order=active_site_order,
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
        self.property_plan.set_frequency(
            interval=interval,
            time_interval=time_interval,
        )

        if self._active_tracker is not None:
            self._active_tracker.set_global_property_frequency(
                interval=interval,
                time_interval=time_interval,
            )

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
        callback_name = self.property_plan.attach(
            func,
            interval=interval,
            time_interval=time_interval,
            name=name,
            store=store,
            max_records=max_records,
            on_error=on_error,
            enabled=enabled,
        )

        if self._active_tracker is not None:
            self._active_tracker.attach(
                func,
                interval=interval,
                time_interval=time_interval,
                name=callback_name,
                store=store,
                max_records=max_records,
                on_error=on_error,
                enabled=enabled,
            )

        return callback_name

    def detach(self, name: str) -> None:
        """Detach a previously attached custom property callback."""
        self._ensure_property_state()
        self.property_plan.detach(name)

        if self._active_tracker is not None:
            self._active_tracker.detach(name)

    def clear_attachments(self) -> None:
        """Remove all attached custom property callbacks."""
        self._ensure_property_state()
        self.property_plan.clear_attachments()
        if self._active_tracker is not None:
            self._active_tracker.clear_attachments()

    def list_attachments(self) -> list[str]:
        """Return names of currently attached custom property callbacks."""
        self._ensure_property_state()
        return self.property_plan.list_attachments()

    def list_property_calculations(self) -> dict[str, list[str]]:
        """Return enabled/disabled built-ins and currently attached callbacks."""
        self._ensure_property_state()
        return self.property_plan.list_property_calculations()

    def set_property_enabled(self, name: str, enabled: bool) -> None:
        """Enable or disable a built-in summary field or attached callback."""
        self._ensure_property_state()
        self.property_plan.set_property_enabled(name, bool(enabled))

        if self._active_tracker is not None:
            self._active_tracker.set_property_enabled(name, bool(enabled))

    def _configure_tracker_properties(self, tracker: Tracker, pass_length: int) -> None:
        """Apply the KMC property plan to a tracker instance."""
        self._ensure_property_state()
        tracker.apply_property_plan(
            self.property_plan,
            default_interval=max(pass_length, 1),
        )

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
        
        This method delegates state management to State, following clean
        architecture principles with single responsibility and separation of concerns.
        
        This method performs the following steps:
        1. Delegates occupation updates to State.apply_event()
        2. Automatically finds the event index in the event library
        3. Identifies all events that need probability updates using EventLib
        4. Recalculates probabilities for affected events
        5. Updates the cumulative probability list for event selection

        Args:
            event: The event object that has just occurred.
            dt (float, optional): Time increment for this event. Used for state tracking.
            
        Side Effects:
            Modifies occupation state and probability lists via State delegation.
        """
        self.simulation_state.apply_event(event, dt)

        # Keep optional model-side state, such as external CE occupancy caches,
        # aligned with the accepted KMC event before future probabilities use it.
        self._apply_model_event(event)
        
        # Find event index automatically from event library
        event_index = self.event_lib.events.index(event)
        
        # Use EventLib to get dependent events
        events_to_be_updated = self.event_lib.get_dependent_events(event_index)
        
        # Update probabilities for dependent events using configured model
        for e_index in events_to_be_updated:
            # Recalculate probability using configured model
            self.prob_list[e_index] = self.model.compute_probability(
                event=self.event_lib.events[e_index],
                runtime_config=self.config.runtime_config,
                simulation_state=self.simulation_state
            )
        self.prob_cum_list = np.cumsum(self.prob_list)

    def run(self, label: str = None) -> Tracker:
        """Run KMC simulation using this instance's Configuration object.

        This is the main method for running KMC simulations using the modern
        Configuration format.

        Args:
            label (str, optional): Label for the simulation run. Defaults to None.
                If None, will use ``self.config.name``.

        Returns:
            kmcpy.tracker.Tracker: Tracker object containing simulation results.
            
        Example::
        
            # Using Configuration
            config = Configuration(name="Test", temperature=400.0, ...)
            tracker = kmc.run()
            
            # Alternative usage patterns:
            # 1. Create KMC and run in one step
            kmc = KMC.from_config(config)
            tracker = kmc.run()
        """
        config = self.config

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
        logger.info("Initial probabilities and cumulative probabilities")
        logger.info("Starting Equilibrium ...")
        
        # Equilibration phase
        for _ in np.arange(config.equilibration_passes):
            for _ in np.arange(pass_length):
                event, dt = self.propose(self.event_lib.events)
                # Keep equilibration out of production time accounting.
                self.update(event, dt=0.0)

        logger.info("Start running kMC ...")

        # Create Tracker using clean State architecture
        self._ensure_property_state()
        tracker = Tracker(
            config=config,
            structure=self.structure,
            initial_state=self.simulation_state,
            property_plan=self.property_plan,
            default_property_interval=max(pass_length, 1),
            hop_state_lookup=getattr(self, "hop_state_lookup", None),
        )
        self._active_tracker = tracker
        logger.info("Using clean State architecture")
        
        logger.info("Tracker summaries are reported as dynamic property tables per pass.")

        # Main KMC loop
        for current_pass in np.arange(config.kmc_passes):
            for _ in np.arange(pass_length):
                event, dt = self.propose(self.event_lib.events)
                
                tracker.update(event, dt)
                # KMC is the single owner of mutable simulation state updates.
                self.update(event, dt=dt)
                tracker.sample_properties(
                    step=int(self.simulation_state.step),
                    sim_time=float(self.simulation_state.time),
                )
            
            tracker.update_current_pass(current_pass)
            tracker.show_current_info()

        tracker.write_results(label=label)
        return tracker

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
