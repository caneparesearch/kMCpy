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
from kmcpy.model.local_cluster_expansion import LocalClusterExpansion
from kmcpy.tracker import Tracker
from kmcpy.event import Event
from kmcpy.io import convert
import logging
import kmcpy
from kmcpy.io import InputSet

logger = logging.getLogger(__name__) 
logging.getLogger('numba').setLevel(logging.WARNING)

class KMC:
    """Kinetic Monte Carlo Simulation Class.

    This class implements a Kinetic Monte Carlo (kMC) simulation for modeling
    stochastic processes in materials, such as ion diffusion. It provides
    methods for initializing the simulation from various input sources,
    managing events, updating system states, and running the simulation loop.
    """
    def __init__(self, initial_occ: list,
                supercell_shape: list,
                fitting_results: str,
                fitting_results_site: str,
                lce_fname: str,
                lce_site_fname: str,
                template_structure_fname: str,
                event_fname: str,
                event_kernel: str,
                v: float = 1e13,
                temperature: float = 300,
                convert_to_primitive_cell: bool=False,
                immutable_sites: list=[],**kwargs)-> None:
        """Initialize the Kinetic Monte Carlo (kMC) simulation.

        It is recommended to use the `from_inputset` method to initialize the
        KMC object from an InputSet.

        Args:
            initial_occ (list): The initial occupation list representing the
            configuration of the system.
            supercell_shape (list): Shape of the supercell as a list of
            integers (e.g., [2, 2, 2]). This should be consistent with
            events.
            fitting_results (str): Path to the JSON file containing the
            fitting results for E_kra.
            fitting_results_site (str): Path to the JSON file containing the
            fitting results for site energy difference.
            lce_fname (str): Path to the JSON file containing the Local
            Cluster Expansion (LCE) model.
            lce_site_fname (str): Path to the JSON file containing the site
            LCE model for computing site energy differences.
            template_structure_fname (str): Path to the CIF file of the
            template structure (with all sites filled).
            event_fname (str): Path to the JSON file containing the list of
            events.
            event_kernel (str): Path to the event kernel file.
            v (float, optional): Attempt frequency (prefactor) for hopping
            events. Defaults to 1e13 Hz.
            temperature (float, optional): Simulation temperature in Kelvin.
            Defaults to 300 K.
            convert_to_primitive_cell (bool, optional): Whether to convert the
            structure to its primitive cell (default: False).
            immutable_sites (list, optional): List of sites to be treated as
            immutable and removed from the simulation (default: []).

        Notes:
            - Loads structure, fitting results, LCE models, and events.
            - Initializes the supercell, occupation, event list, and hopping
              probabilities.
            - Logs detailed information about the initialization process.

        """
        logger.info(kmcpy.get_logo())
        logger.info(f"Initializing kMC calculations ...")
        self.structure = StructureKMCpy.from_cif(
            template_structure_fname, primitive=convert_to_primitive_cell
        )
        supercell_shape_matrix = np.diag(supercell_shape)
        logger.info(f"Supercell Shape:\n {supercell_shape_matrix}")

        logger.info("Converting to the supercell ...")
        logger.debug("removing the immutable sites: {immutable_sites}")
        self.structure.remove_species(immutable_sites)
        self.structure.make_supercell(supercell_shape_matrix)

        logger.info("Loading fitting results: E_kra ...")
        fitting_results = (
            pd.read_json(fitting_results, orient="index")
            .sort_values(by=["time_stamp"], ascending=False)
            .iloc[0]
        )
        self.keci = fitting_results.keci
        self.empty_cluster = fitting_results.empty_cluster

        logger.info("Loading fitting results: site energy ...")
        fitting_results_site = (
            pd.read_json(fitting_results_site, orient="index")
            .sort_values(by=["time_stamp"], ascending=False)
            .iloc[0]
        )
        self.keci_site = fitting_results_site.keci
        self.empty_cluster_site = fitting_results_site.empty_cluster

        logger.info(f"Loading occupation: {initial_occ}")
        self.occ_global = copy(initial_occ)

        logger.info(f"Loading LCE models from: {lce_fname} and {lce_site_fname}")
        local_cluster_expansion = LocalClusterExpansion.from_json(lce_fname)
        local_cluster_expansion_site = LocalClusterExpansion.from_json(lce_site_fname)

        # sublattice_indices are the orbits in table S3 of support information
        # Convert them into Numba List for faster execution
        sublattice_indices = _convert_list(local_cluster_expansion.sublattice_indices)
        sublattice_indices_site = _convert_list(
            local_cluster_expansion_site.sublattice_indices
        )
        events_site_list = []

        logger.info(f"Loading events from: {event_fname}")
        with open(event_fname, "rb") as fhandle:
            events_dict = json.load(fhandle)

        logger.info("Initializing correlation matrix and E_kra for all events ...")
        events = []
        for event_dict in events_dict:
            event = Event.from_dict(event_dict)
            event.set_sublattice_indices(sublattice_indices, sublattice_indices_site)
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
            events_site_list.append(event.local_env_indices_list)
            events.append(event)

        self.events = events

        logger.info("Initializing hopping probabilities ...")
        # Preallocate prob_list and prob_cum_list arrays for reuse
        self.prob_list = np.empty(len(events), dtype=np.float64)
        for i, event in enumerate(events):
            self.prob_list[i] = event.probability
        self.prob_cum_list = np.empty(len(events), dtype=np.float64)
        np.cumsum(self.prob_list, out=self.prob_cum_list)

        logger.info("Fitted time and error (LOOCV,RMS)")
        logger.info(f"{fitting_results.time}, {fitting_results.loocv}, {fitting_results.rmse}")
        logger.info("Fitted KECI and empty cluster E_kra")
        logger.info(f"{self.keci}, {self.empty_cluster}")

        logger.info("Fitted time and error (LOOCV,RMS) (site energy)")
        logger.info(
            f"{fitting_results_site.time}, {fitting_results_site.loocv}, {fitting_results_site.rmse}"
        )
        logger.info("Fitted KECI and empty cluster (site energy or E_end)")
        logger.info(f"{self.keci_site}, {self.empty_cluster_site}")
        logger.info(f"Lists for each event {events_site_list}")
        logger.info(f"Hopping probabilities: {self.prob_list}")
        logger.info(f"Cumulative sum of hopping probabilities: {self.prob_cum_list}")
        logger.info(f"Loading event kernel from: {event_kernel}")
        self.load_site_event_list(fname = event_kernel)

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

    def load_site_event_list(
        self, fname:str="../event_kernal.csv"
    )->None:  # workout the site_event_list -> site_event_list[site_index] will return a list of event index to update if a site_index is chosen
        logger.info("Working at the site_event_list ...")

        logger.info(f"Loading {fname}")
        site_event_list = []
        with open(fname) as f:
            data = f.readlines()
        for x in data:
            if len(x.strip()) == 0:
                site_event_list.append([])
            else:
                site_event_list.append([int(y) for y in x.strip().split()])
        self.site_event_list = site_event_list

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
    ) -> tuple[Event, float]:
        """Propose a new event to be updated by update().

        Args:
            events (list): List of events.

        Returns:
            tuple[Event, float]: The chosen event and the time for this event to occur.
        """
        proposed_event_index, dt = _propose(prob_cum_list=self.prob_cum_list, rng=self.rng)
        event = events[proposed_event_index]
        return event, dt
    

    def update(self, event: Event, events: list[Event])-> None:
        """
        Updates the system state and event probabilities after an event occurs.
        This method performs the following steps:
        1. Flips the occupation values of the two mobile ion species involved in the event.
        2. Identifies all events that need to be updated due to the change in occupation.
        3. Updates each affected event's properties and recalculates their probabilities.
        4. Updates the cumulative probability list for event selection.

        Args:
            event: The event object that has just occurred, containing indices of the affected mobile ion species.
            events (list): List of all event objects in the system.
            
        Side Effects:
            Modifies `self.occ_global`, `self.prob_list`, and `self.prob_cum_list` in place.
        """

        self.occ_global[event.mobile_ion_specie_1_index] *= -1
        self.occ_global[event.mobile_ion_specie_2_index] *= -1
        events_to_be_updated = copy(
            self.site_event_list[event.mobile_ion_specie_2_index]
        )  # event_to_be_updated= list, include the indices, of the event that need to be updated.
        for e_index in events_to_be_updated:
            events[e_index].update_event(
                self.occ_global,
                self.inputset.v,
                self.inputset.temperature,
                self.keci,
                self.empty_cluster,
                self.keci_site,
                self.empty_cluster_site,
            )
            self.prob_list[e_index] = copy(events[e_index].probability)
        self.prob_cum_list = np.cumsum(self.prob_list)

    def run(self, inputset: InputSet, label: str = None)-> Tracker:
        """Run KMC simulation from an InputSet object.

        Args:
            inputset (InputSet): InputSet object containing all necessary parameters.
            label (str, optional): Label for the simulation run. Defaults to None.

        Returns:
            kmcpy.tracker.Tracker: Tracker object containing simulation results.
        """
        self.inputset = inputset
        if hasattr(inputset, "random_seed") and inputset.random_seed is not None:
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
                event, dt = self.propose(self.events)
                self.update(event, self.events)

        logger.info("Start running kMC ...")

        tracker = Tracker.from_inputset(inputset = inputset,
                                        structure = self.structure,
                                        occ_initial = self.occ_global)
        
        logger.info(
            "Pass\tTime\t\tMSD\t\tD_J\t\tD_tracer\tConductivity\tH_R\t\tf"
        )
        for current_pass in np.arange(inputset.kmc_pass):
            for _ in np.arange(pass_length):
                event, dt = self.propose(self.events)
                tracker.update(event, self.occ_global, dt)
                self.update(event, self.events)
            
            tracker.update_current_pass(current_pass)
            tracker.compute_properties()
            tracker.show_current_info()

        tracker.write_results(self.occ_global, label=label)
        return tracker

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