#!/usr/bin/env python
"""
This module provides the KMC class and associated functions for performing Kinetic Monte Carlo (kMC) simulations, 
particularly for modeling stochastic processes in materials such as ion diffusion. The KMC class manages the 
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
from kmcpy.io.io import InputSet
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from kmcpy.simulator.condition import SimulationConfig
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
        self.sim_state = SimulationState(initial_occ=self.occ_global)
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
        from kmcpy.simulator.condition import SimulationConfig
        from kmcpy.simulator.state import SimulationState
        
        # Validate the configuration
        config.validate()
        
        # Convert to InputSet to leverage existing file loading infrastructure in io.py
        inputset = config.to_inputset()
        
        # Load structure using InputSet's capabilities (delegating to io.py)
        structure = StructureKMCpy.from_cif(
            inputset.template_structure_fname, 
            primitive=inputset.convert_to_primitive_cell
        )
        
        # Apply transformations
        if hasattr(inputset, 'immutable_sites') and inputset.immutable_sites:
            structure.remove_species(inputset.immutable_sites)
        
        if hasattr(inputset, 'supercell_shape'):
            supercell_shape_matrix = np.diag(inputset.supercell_shape)
            structure.make_supercell(supercell_shape_matrix)
        
        # Load models and events - use centralized loading methods
        from kmcpy.models.composite_lce_model import CompositeLCEModel
        from kmcpy.event import EventLib, Event
        
        # Load composite model using its centralized from_json method
        model = CompositeLCEModel.from_json(
            lce_fname=inputset.lce_fname,
            fitting_results=inputset.fitting_results,
            lce_site_fname=getattr(inputset, 'lce_site_fname', None),
            fitting_results_site=getattr(inputset, 'fitting_results_site', None)
        )
        
        # Load events
        event_lib = EventLib()
        with open(inputset.event_fname, "rb") as fhandle:
            events_dict = json.load(fhandle)
        
        for event_dict in events_dict:
            event = Event.from_dict(event_dict)
            event_lib.add_event(event)
        
        event_lib.generate_event_dependencies()
        
        # Handle initial occupation using io.py utilities
        initial_occ = getattr(inputset, 'initial_occ', None)
        if initial_occ is None and hasattr(inputset, 'initial_state') and inputset.initial_state:
            # Use InputSet's load_occ capability which handles this properly
            temp_inputset = inputset  # inputset should have load_occ method
            if hasattr(temp_inputset, 'load_occ'):
                temp_inputset.load_occ()
                initial_occ = getattr(temp_inputset, 'initial_occ', None)
        
        simulation_state = SimulationState(initial_occ=initial_occ) if initial_occ is not None else None
        
        return cls(
            structure=structure,
            model=model,
            event_lib=event_lib,
            config=config,
            simulation_state=simulation_state
        )
        
    @classmethod
    def from_inputset(cls, inputset: InputSet)-> "KMC":
        """Initialize KMC from InputSet object (legacy support).

        Note: This method is kept for backward compatibility but delegates
        to the configuration-based approach for consistency.

        Args:
            inputset (kmcpy.io.InputSet): InputSet object containing all
            necessary parameters.

        Returns:
            KMC: An instance of the KMC class initialized with parameters
            from the InputSet.
        """
        # Convert InputSet to SimulationConfig for consistency
        from kmcpy.simulator.condition import SimulationConfig
        
        config = SimulationConfig.from_inputset(inputset)
        
        return cls.from_config(config)



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

    def run(self, inputset: Union[InputSet, "SimulationConfig"], label: str = None) -> Tracker:
        """Run KMC simulation from an InputSet or SimulationConfig object.

        This is the main method for running KMC simulations and supports both legacy InputSet
        objects and modern SimulationConfig objects.

        Args:
            inputset (InputSet or SimulationConfig): Configuration object containing all necessary parameters.
                - InputSet: Legacy format for backward compatibility
                - SimulationConfig: Modern, structured configuration format (recommended)
            label (str, optional): Label for the simulation run. Defaults to None.
                If None and using SimulationConfig, will use config.name.

        Returns:
            kmcpy.tracker.Tracker: Tracker object containing simulation results.
            
        Example:
            ```python
            # Using InputSet (legacy method)
            inputset = InputSet.from_json("input.json")
            tracker = kmc.run(inputset)
            
            # Using SimulationConfig (recommended method)
            config = SimulationConfig(name="Test", temperature=400.0, ...)
            tracker = kmc.run(config)
            
            # Alternative usage patterns:
            # 1. Create KMC and run in one step
            kmc = KMC.from_config(config)
            tracker = kmc.run(config)
            ```
        """
        from kmcpy.simulator.condition import SimulationConfig
        
        # Enhanced handling of SimulationConfig
        original_config = None
        config = None
        if isinstance(inputset, SimulationConfig):
            original_config = inputset
            config = inputset
            config.validate()
            
            # Direct parameter usage where possible
            if label is None:
                label = config.name
            
            # For now, still convert to InputSet for compatibility,
            # but this reduces the dependency on InputSet.to_inputset()
            inputset = config.to_inputset()
        elif hasattr(inputset, 'to_inputset'):
            # Handle other objects that can convert to InputSet
            inputset = inputset.to_inputset()
        
        # Store the original config for potential future use
        self._original_config = original_config
        
        # Continue with existing run logic
        self.inputset = inputset
        
        # Initialize random number generator
        # Priority: config random_seed > inputset random_seed > default
        if hasattr(self.config, 'random_seed') and self.config.random_seed is not None:
            self.rng = np.random.default_rng(seed=self.config.random_seed)
        elif hasattr(inputset, "random_seed") and inputset.random_seed is not None:
            self.rng = np.random.default_rng(seed=inputset.random_seed)
        else:
            self.rng = np.random.default_rng()

        logger.info(f"Simulation condition: v = {inputset.v} T = {inputset.temperature}")
        pass_length = len(
            [
            el.symbol
            for el in self.structure.species
            if inputset.mobile_ion_specie in el.symbol
            ]
        )
        logger.info("============================================================")
        logger.info("Start running kMC ... ")
        logger.info("Initial occ_global, prob_list and prob_cum_list")
        logger.info("Starting Equilbrium ...")
        for _ in np.arange(inputset.equ_pass):
            for _ in np.arange(pass_length):
                event, dt, event_index = self.propose(self.event_lib.events)
                self.update(event, event_index)

        logger.info("Start running kMC ...")

        # Enhanced Tracker creation with better SimulationState integration
        if self.simulation_state is not None:
            # Use existing SimulationState
            tracker = Tracker(config=config or self.config, structure=self.structure, initial_state=self.simulation_state)
            logger.info("Using SimulationState-centric architecture")
        elif hasattr(inputset, 'to_inputset'):
            # It's a SimulationConfig but no SimulationState provided
            tracker = Tracker.from_config(config=inputset, structure=self.structure, occ_initial=self.occ_global)
        else:
            # It's an InputSet (backward compatibility)
            tracker = Tracker.from_inputset(inputset=inputset, structure=self.structure, occ_initial=self.occ_global)
        
        logger.info(
            "Pass\tTime\t\tMSD\t\tD_J\t\tD_tracer\tConductivity\tH_R\t\tf"
        )

        # Get the appropriate pass count
        kmc_pass = inputset.kmc_passes if hasattr(inputset, 'kmc_passes') else inputset.kmc_pass
        
        for current_pass in np.arange(kmc_pass):
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
