#!/usr/bin/env python
"""
This module provides the KMC class and associated functions for performing Kinetic Monte Carlo (kMC) simulations, 
particularly for modeling        # Load composite model using             # Use the pre-calculated select_sites from before structure transformation
            initial_occ = _load_occ(
                fname=config.initial_state_file,
                shape=list(config.supercell_shape),
                select_sites=select_sites_for_occupation
            )ntralized from_json method
        model = CompositeLCEModel.from_json(
            lce_fname=config.cluster_expansion_file,
            fitting_results=config.fitting_results_file,
            lce_site_fname=getattr(config, 'cluster_expansion_site_file', None),
            fitting_results_site=getattr(config, 'fitting_results_site_file', None)
        )
        
        # Load events
        event_lib = EventLib()
        with open(config.event_file, "rb") as fhandle: processes in materials such as ion diffusion. The KMC class manages the 
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
        if simulation_state is not None:
            # Use SimulationState as single source of truth
            logger.info("Using SimulationState for occupation management")
            logger.debug(f"SimulationState occupations length: {len(simulation_state.occupations)}")
            self.occ_global = simulation_state.occupations
        else:
            raise ValueError("SimulationState must be provided")
        
        # Initialize simulation condition and state objects
        self._initialize_simulation()
        
        logger.info("kMC initialization complete!")
        
    def _initialize_simulation(self):
        """Initialize simulation state and calculate initial probabilities."""
        logger.info("Initializing simulation state and probabilities...")
        
        # Create simulation state and condition objects using config
        from kmcpy.simulator.state import SimulationState
        from kmcpy.simulator.condition import SimulationCondition
        self.sim_state = SimulationState(occupations=self.occ_global)
        self.sim_condition = SimulationCondition(
            temperature=self.config.temperature, 
            attempt_frequency=self.config.attempt_frequency
        )
        
        # Calculate probabilities for all events using composite model
        self.prob_list = np.empty(len(self.event_lib), dtype=np.float64)
        for i, event in enumerate(self.event_lib.events):
            # Update the occupation in sim_state for each event calculation
            self.sim_state.occupations = self.occ_global.copy()
            
            self.prob_list[i] = self.model.compute_probability(
                event=event,
                simulation_condition=self.sim_condition,
                simulation_state=self.sim_state
            )
        
        self.prob_cum_list = np.empty(len(self.event_lib), dtype=np.float64)
        np.cumsum(self.prob_list, out=self.prob_cum_list)
        
        logger.info(f"Event dependency matrix with {len(self.event_lib)} events")
        logger.info(f"Hopping probabilities: {self.prob_list}")
        logger.info(f"Cumulative sum of hopping probabilities: {self.prob_cum_list}")
        
        # Display dependency matrix statistics
        stats = self.event_lib.get_dependency_statistics()
        logger.info(f"Dependency matrix statistics: {stats}")
        
        
    @classmethod
    def from_config(cls, config: "SimulationConfig") -> "KMC":
        """Create KMC instance from SimulationConfig (recommended initialization method).

        This is the main initialization method that leverages the io.py module
        for all file loading operations and provides a clean interface.

        Args:
            config (SimulationConfig): Configuration object containing all necessary parameters.

        Returns:
            KMC: An instance of the KMC class.
        """
        from kmcpy.simulator.state import SimulationState
        
                # Load structure directly from config
        structure = StructureKMCpy.from_cif(
            config.structure_file, 
            primitive=config.convert_to_primitive_cell
        )
        
        # Calculate select_sites based on original structure BEFORE removing immutable sites
        select_sites_for_occupation = []
        if config.initial_state_file:
            immutable_sites = config.immutable_sites or []
            for index, site in enumerate(structure.sites):
                if site.specie.symbol not in immutable_sites:
                    select_sites_for_occupation.append(index)
            logger.debug(f"Select sites for occupation loading: {select_sites_for_occupation}")
        
        # Apply transformations AFTER calculating select_sites
        if config.immutable_sites:
            structure.remove_species(config.immutable_sites)
        
        if config.supercell_shape:
            supercell_shape_matrix = np.diag(config.supercell_shape)
            structure.make_supercell(supercell_shape_matrix)
        
        # Load models and events - use centralized loading methods
        from kmcpy.models.composite_lce_model import CompositeLCEModel
        from kmcpy.event import EventLib, Event
        
        # Load composite model using its centralized from_json method
        model = CompositeLCEModel.from_json(
            lce_fname=config.cluster_expansion_file,
            fitting_results=config.fitting_results_file,
            lce_site_fname=getattr(config, 'cluster_expansion_site_file', None),
            fitting_results_site=getattr(config, 'fitting_results_site_file', None)
        )
        
        # Load events
        event_lib = EventLib()
        with open(config.event_file, "rb") as fhandle:
            events_dict = json.load(fhandle)
        
        for event_dict in events_dict:
            event = Event.from_dict(event_dict)
            event_lib.add_event(event)
        
        event_lib.generate_event_dependencies()
        
        # Handle initial occupation from config
        initial_occ = None
        if config.initial_occupations:
            initial_occ = list(config.initial_occupations)
        elif config.initial_state_file:
            # Load initial state using the proper io.py method with correct site selection
            from kmcpy.io.io import _load_occ
            
            # Use the pre-calculated select_sites from before structure transformation
            initial_occ = _load_occ(
                fname=config.initial_state_file,
                shape=list(config.supercell_shape),
                select_sites=select_sites_for_occupation
            )
        
        # Always create a SimulationState (even if we have to use empty occupations)
        if initial_occ is not None:
            simulation_state = SimulationState(occupations=initial_occ)
        else:
            # Create with empty occupations - this will be populated during structure loading
            simulation_state = SimulationState(occupations=[])
        
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
        return event, dt, proposed_event_index
    

    def update(self, event: Event, event_index: int = None)-> None:
        """
        Updates the system state and event probabilities after an event occurs.
        
        This method works with SimulationState when available, eliminating state 
        duplication and improving consistency.
        
        This method performs the following steps:
        1. Flips the occupation values of the two mobile ion species involved in the event.
        2. Identifies all events that need to be updated due to the change in occupation using EventLib.
        3. Updates each affected event's properties and recalculates their probabilities.
        4. Updates the cumulative probability list for event selection.

        Args:
            event: The event object that has just occurred, containing indices of the affected mobile ion species.
            event_index (int, optional): Index of the executed event. If None, will find it automatically.
            
        Side Effects:
            Modifies occupation state and probability lists in place.
        """
        # Update occupations in SimulationState if available, otherwise use direct mode
        if self.simulation_state is not None:
            # Use SimulationState as single source of truth
            self.simulation_state.occupations[event.mobile_ion_indices[0]] *= -1
            self.simulation_state.occupations[event.mobile_ion_indices[1]] *= -1
            # Keep occ_global synchronized for event updates
            self.occ_global = self.simulation_state.occupations
        else:
            # Direct mode: update occ_global directly
            self.occ_global[event.mobile_ion_indices[0]] *= -1
            self.occ_global[event.mobile_ion_indices[1]] *= -1
        
        # Find event index if not provided
        if event_index is None:
            event_index = self.event_lib.events.index(event)
        
        # Use EventLib to get dependent events
        events_to_be_updated = self.event_lib.get_dependent_events(event_index)
        
        # Update probabilities for dependent events using composite model
        for e_index in events_to_be_updated:
            # Update the simulation state with new occupations
            self.sim_state.occupations = self.occ_global
            
            # Recalculate probability using composite model
            self.prob_list[e_index] = self.model.compute_probability(
                event=self.event_lib.events[e_index],
                simulation_condition=self.sim_condition,
                simulation_state=self.sim_state
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
                event, dt, event_index = self.propose(self.event_lib.events)
                self.update(event, event_index)

        logger.info("Start running kMC ...")

        # Create Tracker
        if self.simulation_state is not None:
            # Use existing SimulationState
            tracker = Tracker(config=config, structure=self.structure, initial_state=self.simulation_state)
            logger.info("Using SimulationState-centric architecture")
        else:
            # Create tracker with current occupations
            from kmcpy.simulator.state import SimulationState
            initial_state = SimulationState(occupations=self.occ_global)
            tracker = Tracker(config=config, structure=self.structure, initial_state=initial_state)
        
        logger.info(
            "Pass\tTime\t\tMSD\t\tD_J\t\tD_tracer\tConductivity\tH_R\t\tf"
        )

        # Main KMC loop
        for current_pass in np.arange(config.kmc_passes):
            for _ in np.arange(pass_length):
                event, dt, event_index = self.propose(self.event_lib.events)
                
                # Standard workflow - let Tracker handle mobile ion tracking
                final_occupations = self.simulation_state.occupations if self.simulation_state else self.occ_global
                tracker.update(event, final_occupations, dt)
                self.update(event, event_index)
            
            tracker.update_current_pass(current_pass)
            tracker.compute_properties()
            tracker.show_current_info()

        # Use SimulationState occupations if available
        final_occupations = self.simulation_state.occupations if self.simulation_state else self.occ_global
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
