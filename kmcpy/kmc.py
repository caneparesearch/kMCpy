#!/usr/bin/env python
"""
This module provides the KMC class and associated functions for performing Kinetic Monte Carlo (kMC) simulations, particularly for modeling stochastic processes in materials such as ion diffusion. The KMC class manages the initialization, event handling, probability calculations, and simulation loop for kMC workflows. It supports loading input data from various sources, updating system states, and tracking simulation results.
"""
from numba.typed import List
from numba import njit
from kmcpy.external.structure import StructureKMCpy
import numpy as np
import pandas as pd
from copy import copy
import json
import os
from kmcpy.model.local_cluster_expansion import LocalClusterExpansion
from kmcpy.tracker import Tracker
from kmcpy.event import Event, EventLib
from kmcpy.io import convert
import logging
import kmcpy
from kmcpy.io import InputSet
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from kmcpy.simulation_condition import SimulationConfig, SimulationState

logger = logging.getLogger(__name__) 
logging.getLogger('numba').setLevel(logging.WARNING)

class KMC:
    """Kinetic Monte Carlo Simulation Class.

    This class implements a Kinetic Monte Carlo (kMC) simulation for modeling
    stochastic processes in materials, such as ion diffusion. It provides
    methods for initializing the simulation from various input sources,
    managing events, updating system states, and running the simulation loop.
    """
    def __init__(self, initial_occ: list = None,
                supercell_shape: list = None,
                fitting_results: str = None,
                fitting_results_site: str = None,
                lce_fname: str = None,
                lce_site_fname: str = None,
                template_structure_fname: str = None,
                event_fname: str = None,
                event_dependencies: str = None,
                initial_state: str = None,  # Path to initial state JSON file
                v: float = 1e13,
                temperature: float = 300,
                convert_to_primitive_cell: bool = False,
                immutable_sites: list = None,
                simulation_state: "SimulationState" = None,
                random_seed: int = None,
                **kwargs) -> None:
        """Initialize the Kinetic Monte Carlo (kMC) simulation.

        Args:
            initial_occ (list, optional): The initial occupation list.
            supercell_shape (list, optional): Shape of the supercell.
            fitting_results (str, optional): Path to fitting results JSON file.
            fitting_results_site (str, optional): Path to site fitting results JSON file.
            lce_fname (str, optional): Path to LCE model JSON file.
            lce_site_fname (str, optional): Path to site LCE model JSON file.
            template_structure_fname (str, optional): Path to template structure CIF file.
            event_fname (str, optional): Path to events JSON file.
            event_dependencies (str, optional): Path to event dependencies file.
            initial_state (str, optional): Path to initial state JSON file.
            v (float, optional): Attempt frequency. Defaults to 1e13 Hz.
            temperature (float, optional): Simulation temperature in Kelvin. Defaults to 300 K.
            convert_to_primitive_cell (bool, optional): Whether to convert to primitive cell.
            immutable_sites (list, optional): List of immutable sites to remove.
            simulation_state (SimulationState, optional): SimulationState object for state management.
            random_seed (int, optional): Random seed for reproducible simulations.
            **kwargs: Additional keyword arguments.

        Note:
            When simulation_state is provided, it becomes the single source of truth for
            all mutable simulation state. This is the preferred mode for new code.
            
            For occupation initialization, the priority is:
            1. simulation_state (if provided)
            2. initial_occ (if provided)
            3. initial_state (if provided)
        """
        # Set default values
        if immutable_sites is None:
            immutable_sites = []
            
        # Store random seed for later use
        self.random_seed = random_seed
        
        # Validate that event_dependencies is provided
        if event_dependencies is None:
            raise ValueError("event_dependencies must be provided.")
        
        logger.info(kmcpy.get_logo())
        logger.info(f"Initializing kMC calculations ...")
        
        # Store simulation state reference
        self.simulation_state = simulation_state
        
        # Initialize structure
        self.structure = StructureKMCpy.from_cif(
            template_structure_fname, primitive=convert_to_primitive_cell
        )
        
        # Handle supercell shape
        if supercell_shape is None:
            supercell_shape = [1, 1, 1]
            
        supercell_shape_matrix = np.diag(supercell_shape)
        logger.info(f"Supercell Shape:\n {supercell_shape_matrix}")

        logger.info("Converting to the supercell ...")
        
        # IMPORTANT: Calculate mutable sites BEFORE removing immutable sites
        # This ensures consistent behavior with InputSet approach
        mutable_sites = self._get_mutable_sites(immutable_sites)
        
        logger.debug("removing the immutable sites: {immutable_sites}")
        self.structure.remove_species(immutable_sites)
        self.structure.make_supercell(supercell_shape_matrix)

        # Load fitting results and models
        self._load_fitting_results(fitting_results, fitting_results_site)
        self._load_lce_models(lce_fname, lce_site_fname)
        
        # Initialize occupation state
        if simulation_state is not None:
            # Use SimulationState as single source of truth
            logger.info("Using SimulationState for occupation management")
            self.occ_global = simulation_state.occupations
        elif initial_occ is not None:
            # Use provided initial_occ
            logger.info(f"Loading occupation: {initial_occ}")
            self.occ_global = copy(initial_occ)
        elif initial_state is not None:
            # Load occupation from initial_state file
            logger.info(f"Loading occupation from initial_state file: {initial_state}")
            from kmcpy.io import _load_occ
            self.occ_global = _load_occ(
                initial_state,
                supercell_shape,
                # Use the pre-calculated mutable sites to ensure consistency with InputSet approach
                mutable_sites
            )
        else:
            raise ValueError("Either initial_occ, initial_state, or simulation_state must be provided")
        
        # Initialize events
        self._initialize_events(event_fname, event_dependencies, v, temperature)
        
        logger.info("kMC initialization complete!")
        
    def _load_fitting_results(self, fitting_results: str, fitting_results_site: str):
        """Load fitting results for E_kra and site energy."""
        logger.info("Loading fitting results: E_kra ...")
        self.fitting_results_data = (
            pd.read_json(fitting_results, orient="index")
            .sort_values(by=["time_stamp"], ascending=False)
            .iloc[0]
        )
        self.keci = self.fitting_results_data.keci
        self.empty_cluster = self.fitting_results_data.empty_cluster

        # Only load site fitting results if file exists
        if fitting_results_site is not None and os.path.exists(fitting_results_site):
            logger.info("Loading fitting results: site energy ...")
            self.fitting_results_site_data = (
                pd.read_json(fitting_results_site, orient="index")
                .sort_values(by=["time_stamp"], ascending=False)
                .iloc[0]
            )
            self.keci_site = self.fitting_results_site_data.keci
            self.empty_cluster_site = self.fitting_results_site_data.empty_cluster
        else:
            logger.info("No site fitting results file found - using zero site energy contributions")
            self.fitting_results_site_data = None
            self.keci_site = None
            self.empty_cluster_site = None
        
    def _load_lce_models(self, lce_fname: str, lce_site_fname: str):
        """Load Local Cluster Expansion models."""
        logger.info(f"Loading LCE models from: {lce_fname}")
        local_cluster_expansion = LocalClusterExpansion.from_json(lce_fname)
        
        # Load site LCE if file exists
        if lce_site_fname is not None and os.path.exists(lce_site_fname):
            logger.info(f"Loading site LCE from: {lce_site_fname}")
            local_cluster_expansion_site = LocalClusterExpansion.from_json(lce_site_fname)
            sublattice_indices_site = _convert_list(
                local_cluster_expansion_site.sublattice_indices
            )
        else:
            logger.info("No site LCE file found - using empty site indices")
            # Create empty sublattice indices for site model
            sublattice_indices_site = List([])

        # sublattice_indices are the orbits in table S3 of support information
        # Convert them into Numba List for faster execution
        sublattice_indices = _convert_list(local_cluster_expansion.sublattice_indices)
        
        # Store for later use
        self.sublattice_indices = sublattice_indices
        self.sublattice_indices_site = sublattice_indices_site
        
    def _initialize_events(self, event_fname: str, event_dependencies: str, v: float, temperature: float):
        """Initialize events and event library."""
        logger.info(f"Loading events from: {event_fname}")
        
        # Create EventLib and load events
        self.event_lib = EventLib()
        
        with open(event_fname, "rb") as fhandle:
            events_dict = json.load(fhandle)

        logger.info("Initializing correlation matrix and E_kra for all events ...")
        for event_dict in events_dict:
            event = Event.from_dict(event_dict)
            event.set_sublattice_indices(self.sublattice_indices, self.sublattice_indices_site)
            event.initialize_corr()
            event.update_event(
                self.occ_global,
                v,
                temperature,
                self.keci,
                self.empty_cluster,
                self.keci_site,
                self.empty_cluster_site,
            )
            self.event_lib.add_event(event)

        # Generate event dependencies using the optimized algorithm
        logger.info("Generating event dependency matrix...")
        self.event_lib.generate_event_dependencies()
        
        # Keep reference to events for compatibility
        self.events = self.event_lib.events

        logger.info("Initializing hopping probabilities ...")
        # Preallocate prob_list and prob_cum_list arrays for reuse
        self.prob_list = np.empty(len(self.event_lib), dtype=np.float64)
        for i, event in enumerate(self.event_lib.events):
            self.prob_list[i] = event.probability
        self.prob_cum_list = np.empty(len(self.event_lib), dtype=np.float64)
        np.cumsum(self.prob_list, out=self.prob_cum_list)

        logger.info("Fitted time and error (LOOCV,RMS)")
        logger.info(f"{self.fitting_results_data.time}, {self.fitting_results_data.loocv}, {self.fitting_results_data.rmse}")
        logger.info("Fitted KECI and empty cluster E_kra")
        logger.info(f"{self.keci}, {self.empty_cluster}")

        logger.info("Fitted time and error (LOOCV,RMS) (site energy)")
        logger.info(
            f"{self.fitting_results_site_data.time}, {self.fitting_results_site_data.loocv}, {self.fitting_results_site_data.rmse}"
        )
        logger.info("Fitted KECI and empty cluster (site energy or E_end)")
        logger.info(f"{self.keci_site}, {self.empty_cluster_site}")
        logger.info(f"Event dependency matrix generated with {len(self.event_lib)} events")
        logger.info(f"Hopping probabilities: {self.prob_list}")
        logger.info(f"Cumulative sum of hopping probabilities: {self.prob_cum_list}")
        
        # Display dependency matrix statistics
        stats = self.event_lib.get_dependency_statistics()
        logger.info(f"Dependency matrix statistics: {stats}")

    @classmethod
    def from_inputset(cls, inputset: InputSet)-> "KMC":
        """Initialize KMC from InputSet object.

        Args:
            inputset (kmcpy.io.InputSet): InputSet object containing all
            necessary parameters.

        Returns:
            KMC: An instance of the KMC class initialized with parameters
            from the InputSet.
        """
        params = {k: v for k, v in inputset._parameters.items() if k != "task"}
        return cls(**params)

    @classmethod
    def from_simulation_config(cls, config: "SimulationConfig") -> "KMC":
        """Initialize KMC from SimulationConfig object.

        Args:
            config (SimulationConfig): SimulationConfig object containing all
            necessary parameters in a structured format.

        Returns:
            KMC: An instance of the KMC class initialized with parameters
            from the SimulationConfig.
            
        Example:
            ```python
            from kmcpy.simulation_condition import SimulationConfig
            
            config = SimulationConfig(
                name="NASICON_Simulation",
                temperature=573.0,
                attempt_frequency=1e13,
                fitting_results="path/to/fitting_results.json",
                # ... other parameters
            )
            
            kmc = KMC.from_simulation_config(config)
            ```
        """
        from kmcpy.simulation_condition import SimulationConfig
        
        # Validate the configuration
        config.validate()
        
        # Direct parameter mapping from SimulationConfig
        kmc_params = {
            # Core simulation parameters
            'temperature': config.temperature,
            'v': config.attempt_frequency,
            'supercell_shape': config.supercell_shape,
            'convert_to_primitive_cell': config.convert_to_primitive_cell,
            'immutable_sites': config.immutable_sites,
            'random_seed': config.random_seed,
            'equ_pass': config.equilibration_passes,
            'kmc_pass': config.kmc_passes,
            'dimension': config.dimension,
            'elem_hop_distance': config.elementary_hop_distance,
            'q': config.mobile_ion_charge,
            'mobile_ion_specie': config.mobile_ion_specie,
            
            # File parameters
            'fitting_results': config.fitting_results,
            'fitting_results_site': config.fitting_results_site,
            'lce_fname': config.lce_fname,
            'lce_site_fname': config.lce_site_fname,
            'template_structure_fname': config.template_structure_fname,
            'event_fname': config.event_fname,
        }
        
        # Handle initial occupation - if both initial_occ and initial_state exist, prefer initial_occ
        if config.initial_occ:
            kmc_params['initial_occ'] = config.initial_occ
        elif config.initial_state:
            kmc_params['initial_state'] = config.initial_state
        
        # Handle event dependencies if provided
        if hasattr(config, 'event_dependencies') and config.event_dependencies:
            kmc_params['event_dependencies'] = config.event_dependencies
        
        # Remove None values to avoid parameter errors
        kmc_params = {k: v for k, v in kmc_params.items() if v is not None}
        
        # Create KMC instance directly with validated parameters
        return cls(**kmc_params)

    @classmethod
    def run_simulation(cls, config: "SimulationConfig", label: str = None) -> Tracker:
        """Create KMC instance and run simulation in one step.

        Args:
            config (SimulationConfig): SimulationConfig object containing all simulation parameters.
            label (str, optional): Label for the simulation run. Defaults to config.name.

        Returns:
            kmcpy.tracker.Tracker: Tracker object containing simulation results.
            
        Example:
            ```python
            from kmcpy.simulation_condition import SimulationConfig
            
            config = SimulationConfig(
                name="NASICON_Simulation",
                temperature=573.0,
                # ... other parameters
            )
            
            # Create and run in one step
            tracker = KMC.run_simulation(config)
            ```
        """
        # Use direct SimulationConfig workflow
        kmc = cls.from_simulation_config(config)
        return kmc.run(config, label)

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
            self.simulation_state.occupations[event.mobile_ion_specie_1_index] *= -1
            self.simulation_state.occupations[event.mobile_ion_specie_2_index] *= -1
            # Keep occ_global synchronized for event updates
            self.occ_global = self.simulation_state.occupations
        else:
            # Direct mode: update occ_global directly
            self.occ_global[event.mobile_ion_specie_1_index] *= -1
            self.occ_global[event.mobile_ion_specie_2_index] *= -1
        
        # Find event index if not provided
        if event_index is None:
            event_index = self.event_lib.events.index(event)
        
        # Use EventLib to get dependent events
        events_to_be_updated = self.event_lib.get_dependent_events(event_index)
        
        for e_index in events_to_be_updated:
            self.event_lib.events[e_index].update_event(
                self.occ_global,
                self.inputset.v,
                self.inputset.temperature,
                self.keci,
                self.empty_cluster,
                self.keci_site,
                self.empty_cluster_site,
            )
            self.prob_list[e_index] = copy(self.event_lib.events[e_index].probability)
        self.prob_cum_list = np.cumsum(self.prob_list)

    def run(self, inputset: Union[InputSet, "SimulationConfig"], label: str = None) -> Tracker:
        """Run KMC simulation from an InputSet or SimulationConfig object.

        Args:
            inputset (InputSet or SimulationConfig): InputSet or SimulationConfig object containing all necessary parameters.
            label (str, optional): Label for the simulation run. Defaults to None.

        Returns:
            kmcpy.tracker.Tracker: Tracker object containing simulation results.
            
        Example:
            ```python
            # Using InputSet (existing method)
            inputset = InputSet.from_json("input.json")
            tracker = kmc.run(inputset)
            
            # Using SimulationConfig (new method - Phase 4 improvement)
            config = SimulationConfig(name="Test", temperature=400.0, ...)
            tracker = kmc.run(config)
            ```
        """
        from kmcpy.simulation_condition import SimulationConfig
        
        # Enhanced handling of SimulationConfig
        original_config = None
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
        # Priority: instance random_seed > inputset random_seed > default
        if self.random_seed is not None:
            self.rng = np.random.default_rng(seed=self.random_seed)
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
                event, dt, event_index = self.propose(self.events)
                self.update(event, event_index)

        logger.info("Start running kMC ...")

        # Enhanced Tracker creation with better SimulationState integration
        if self.simulation_state is not None:
            # Use existing SimulationState
            tracker = Tracker(config=config, structure=self.structure, initial_state=self.simulation_state)
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
                event, dt, event_index = self.propose(self.events)
                
                # Optimized update workflow
                if self.simulation_state is not None:
                    # Update SimulationState directly, no need for occupation synchronization
                    tracker.state.update_from_event(event, dt)
                    self.update(event, event_index)  # This will sync from SimulationState
                else:
                    # Standard workflow
                    tracker.update(event, self.occ_global, dt)
                    self.update(event, event_index)
            
            tracker.update_current_pass(current_pass)
            tracker.compute_properties()
            tracker.show_current_info()

        # Use SimulationState occupations if available
        final_occupations = self.simulation_state.occupations if self.simulation_state else self.occ_global
        tracker.write_results(final_occupations, label=label)
        return tracker

    def run_with_config(self, config: "SimulationConfig", label: str = None) -> Tracker:
        """Run KMC simulation from a SimulationConfig object.

        Args:
            config (SimulationConfig): SimulationConfig object containing all simulation parameters.
            label (str, optional): Label for the simulation run. Defaults to config.name.

        Returns:
            kmcpy.tracker.Tracker: Tracker object containing simulation results.
            
        Example:
            ```python
            from kmcpy.simulation_condition import SimulationConfig
            
            config = SimulationConfig(
                name="NASICON_Simulation",
                temperature=573.0,
                equilibration_passes=1000,
                kmc_passes=10000,
                # ... other parameters
            )
            
            kmc = KMC.from_simulation_config(config)
            tracker = kmc.run_with_config(config)
            ```
        """
        from kmcpy.simulation_condition import SimulationConfig
        
        # Validate the configuration
        config.validate()
        
        # Convert to InputSet for compatibility with existing run method
        inputset = config.to_inputset()
        
        # Use default label if none provided
        if label is None:
            label = config.name
        
        return self.run(inputset, label)

    @classmethod
    def from_simulation_state(cls, config: "SimulationConfig", simulation_state: "SimulationState") -> "KMC":
        """Initialize KMC from SimulationConfig and SimulationState objects.
        
        This method eliminates state duplication and provides better integration 
        with SimulationState.

        Args:
            config (SimulationConfig): SimulationConfig object containing all configuration parameters.
            simulation_state (SimulationState): SimulationState object containing mutable simulation state.

        Returns:
            KMC: An instance of the KMC class initialized with the provided configuration and state.
            
        Example:
            ```python
            from kmcpy.simulation_condition import SimulationConfig, SimulationState
            
            config = SimulationConfig(
                name="NASICON_Simulation",
                temperature=573.0,
                # ... other parameters
            )
            
            state = SimulationState(
                initial_occ=[1, -1, 1, -1],
                structure=structure,
                mobile_ion_specie="Na"
            )
            
            kmc = KMC.from_simulation_state(config, state)
            ```
        """
        from kmcpy.simulation_condition import SimulationConfig, SimulationState
        
        # Validate the configuration
        config.validate()
        
        # Create KMC instance with SimulationState
        kmc = cls(
            initial_occ=simulation_state.occupations,
            supercell_shape=config.supercell_shape,
            fitting_results=config.fitting_results,
            fitting_results_site=config.fitting_results_site,
            lce_fname=config.lce_fname,
            lce_site_fname=config.lce_site_fname,
            template_structure_fname=config.template_structure_fname,
            event_fname=config.event_fname,
            event_dependencies=config.event_dependencies,
            v=config.attempt_frequency,
            temperature=config.temperature,
            convert_to_primitive_cell=config.convert_to_primitive_cell,
            immutable_sites=config.immutable_sites,
            simulation_state=simulation_state  # Pass SimulationState
        )
        
        return kmc

    def as_dict(self)-> dict:
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "structure": self.structure.as_dict(),
            "keci": self.keci,
            "empty_cluster": self.empty_cluster,
            "keci_site": self.keci_site,
            "empty_cluster_site": self.empty_cluster_site,
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

    def _get_mutable_sites(self, immutable_sites: list) -> list:
        """Get list of mutable site indices for occupation loading."""
        if immutable_sites is None:
            immutable_sites = []
            
        mutable_sites = []
        for index, site in enumerate(self.structure.sites):
            if site.specie.symbol not in immutable_sites:
                mutable_sites.append(index)
        
        logger.debug(f"Mutable sites: {mutable_sites}")
        return mutable_sites
    

def _convert_list(list_to_convert)-> List:
    converted_list = List([List(List(j) for j in i) for i in list_to_convert])
    return converted_list


@njit
def _propose(prob_cum_list, rng)-> tuple[int, float]:
    random_seed = rng.random()
    random_seed_2 = rng.random()
    proposed_event_index = np.searchsorted(
        prob_cum_list / prob_cum_list[-1], random_seed, side="right"
    )
    dt = (-1.0 / prob_cum_list[-1]) * np.log(random_seed_2)
    return proposed_event_index, dt
