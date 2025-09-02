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
from kmcpy.simulator.tracker import Tracker
from kmcpy.event import Event, EventLib
from kmcpy.io.io import convert
import logging
import kmcpy
from typing import TYPE_CHECKING

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
        
        # Initialize occupation state from simulation_state
        # Initialize simulation state and condition objects
        if simulation_state is not None:
            # Use SimulationState as single source of truth
            logger.info("Using SimulationState for occupation management")
            logger.debug(f"SimulationState occupations length: {len(simulation_state.occupations)}")
            self.occ_global = simulation_state.occupations
        else:
            raise ValueError("SimulationState must be provided for clean architecture")

        # Initialize simulation condition and calculate initial probabilities
        logger.info("Initializing simulation condition and probabilities...")
        
        # Create simulation condition object using config
        from kmcpy.simulator.condition import SimulationCondition
        self.sim_condition = SimulationCondition(
            temperature=self.config.temperature, 
            attempt_frequency=self.config.attempt_frequency
        )
        
        # Calculate probabilities for all events using composite model
        self.prob_list = np.empty(len(self.event_lib), dtype=np.float64)
        for i, event in enumerate(self.event_lib.events):
            # Update the occupation in simulation_state for each event calculation
            self.simulation_state.occupations = self.occ_global.copy()

            self.prob_list[i] = self.model.compute_probability(
                event=event,
                simulation_condition=self.sim_condition,
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
        try:
            logger.info("Probabilities:")
            logger.info(self.prob_list)
            logger.info("Cumultative probability list:")
            logger.info(self.prob_cum_list / sum(self.prob_list))
        except Exception:
            pass

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
                simulation_condition=self.sim_condition,
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

        Returns:
            kmcpy.tracker.Tracker: Tracker object containing simulation results.
            
        Example:
            ```python
            # Using SimulationConfig
            config = SimulationConfig.create(name="Test", temperature=400.0, ...)
            tracker = kmc.run(config)
            
            # Alternative usage patterns:
            # 1. Create KMC and run in one step
            kmc = KMC.from_config(config)
            tracker = kmc.run(config)
            ```
        """
        # Set label from config if not provided
        if label is None:
            label = config.name
        
        # Initialize random number generator
        if config.random_seed is not None:
            self.rng = np.random.default_rng(seed=config.random_seed)
        else:
            self.rng = np.random.default_rng()

        logger.info(f"Simulation condition: v = {config.attempt_frequency} T = {config.temperature}")
        
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
                self.update(event)

        logger.info("Start running kMC ...")

        # Create Tracker using clean SimulationState architecture
        tracker = Tracker(config=config, structure=self.structure, initial_state=self.simulation_state)
        logger.info("Using clean SimulationState architecture")
        
        logger.info(
            "Pass\tTime\t\tMSD\t\tD_J\t\tD_tracer\tConductivity\tH_R\t\tf"
        )

        # Main KMC loop
        for current_pass in np.arange(config.kmc_passes):
            for _ in np.arange(pass_length):
                event, dt = self.propose(self.event_lib.events)
                
                # Standard workflow - let Tracker handle mobile ion tracking
                final_occupations = self.simulation_state.occupations
                tracker.update(event, final_occupations, dt)
                self.update(event)
            
            tracker.update_current_pass(current_pass)
            tracker.compute_properties()
            tracker.show_current_info()

        # Use SimulationState occupations for final output
        final_occupations = self.simulation_state.occupations
        tracker.write_results(final_occupations, label=label)
        return tracker

    def as_dict(self)-> dict:
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
        logger.info(f"Saving: {fname}")
        with open(fname, "w") as fhandle:
            d = self.as_dict()
            jsonStr = json.dumps(
                d, indent=4, default=convert
            )  # to get rid of errors of int64
            fhandle.write(jsonStr)

    @classmethod
    def from_json(cls, fname)-> "KMC":
        logger.info(f"Loading: {fname}")
        with open(fname, "rb") as fhandle:
            objDict = json.load(fhandle)
        obj = KMC()
        obj.__dict__ = objDict
        logger.info("load complete")
        return obj

@njit
def _propose(prob_cum_list, rng)-> tuple[int, float]:
    random_seed = rng.random()
    random_seed_2 = rng.random()
    proposed_event_index = np.searchsorted(
        prob_cum_list / prob_cum_list[-1], random_seed, side="right"
    )
    dt = (-1.0 / prob_cum_list[-1]) * np.log(random_seed_2)
    return proposed_event_index, dt
