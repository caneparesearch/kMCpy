"""
Function and classes for running kMC

Author: Zeyu Deng
Email: dengzeyu@gmail.com
"""

from numba.typed import List
from numba import njit
from kmcpy.external.structure import StructureKMCpy
import numpy as np
import pandas as pd
from copy import copy
import json
from kmcpy.model import LocalClusterExpansion
from kmcpy.tracker import Tracker
from kmcpy.event import Event
from kmcpy.io import convert
import logging
import kmcpy
from kmcpy.io import InputSet

logger = logging.getLogger(__name__) 
logging.getLogger('numba').setLevel(logging.WARNING)

class KMC:
    """
    main function of kinetic monte carlo
    """

    def __init__(self):
        pass

    @classmethod
    def from_inputset(cls, inputset: InputSet):
        """
        Initialize KMC from InputSet object.

        Args:
            inputset (kmcpy.io.InputSet): InputSet object containing all necessary parameters.

        Returns:
            KMC: An instance of the KMC class initialized with parameters from the InputSet.
        """
        kmc = cls()
        logger.info(kmcpy.get_logo())
        logger.info(f"Initializing kMC calculations with pirm.cif at {inputset.prim_fname} ...")
        kmc.structure = StructureKMCpy.from_cif(
            inputset.prim_fname, primitive=inputset.convert_to_primitive_cell
        )
        supercell_shape_matrix = np.diag(inputset.supercell_shape)
        logger.info(f"Supercell Shape:\n {supercell_shape_matrix}")

        logger.info("Converting to the supercell ...")
        logger.debug("removing the immutable sites: {immutable_sites}")
        kmc.structure.remove_species(inputset.immutable_sites)
        kmc.structure.make_supercell(supercell_shape_matrix)

        logger.info("Loading fitting results: E_kra ...")
        fitting_results = (
            pd.read_json(inputset.fitting_results, orient="index")
            .sort_values(by=["time_stamp"], ascending=False)
            .iloc[0]
        )
        kmc.keci = fitting_results.keci
        kmc.empty_cluster = fitting_results.empty_cluster

        logger.info("Loading fitting results: site energy ...")
        fitting_results_site = (
            pd.read_json(inputset.fitting_results_site, orient="index")
            .sort_values(by=["time_stamp"], ascending=False)
            .iloc[0]
        )
        kmc.keci_site = fitting_results_site.keci
        kmc.empty_cluster_site = fitting_results_site.empty_cluster

        logger.info(f"Loading occupation: {inputset.initial_occ}")
        kmc.occ_global = copy(inputset.initial_occ)

        logger.info(f"Loading LCE models from: {inputset.lce_fname} and {inputset.lce_site_fname}")
        local_cluster_expansion = LocalClusterExpansion.from_json(inputset.lce_fname)
        local_cluster_expansion_site = LocalClusterExpansion.from_json(inputset.lce_site_fname)

        # sublattice_indices are the orbits in table S3 of support information
        # Convert them into Numba List for faster execution
        sublattice_indices = _convert_list(local_cluster_expansion.sublattice_indices)
        sublattice_indices_site = _convert_list(
            local_cluster_expansion_site.sublattice_indices
        )
        events_site_list = []

        logger.info(f"Loading events from: {inputset.event_fname}")
        with open(inputset.event_fname, "rb") as fhandle:
            events_dict = json.load(fhandle)

        logger.info("Initializing correlation matrix and E_kra for all events ...")
        events = []
        for event_dict in events_dict:
            event = Event.from_dict(event_dict)
            event.set_sublattice_indices(sublattice_indices, sublattice_indices_site)
            event.initialize_corr()
            event.update_event(
            kmc.occ_global,
            inputset.v,
            inputset.T,
            kmc.keci,
            kmc.empty_cluster,
            kmc.keci_site,
            kmc.empty_cluster_site,
            )
            events_site_list.append(event.local_env_indices_list)
            events.append(event)

        kmc.events = events

        logger.info("Initializing hopping probabilities ...")
        # Preallocate prob_list and prob_cum_list arrays for reuse
        kmc.prob_list = np.empty(len(events), dtype=np.float64)
        for i, event in enumerate(events):
            kmc.prob_list[i] = event.probability
        kmc.prob_cum_list = np.empty(len(events), dtype=np.float64)
        np.cumsum(kmc.prob_list, out=kmc.prob_cum_list)

        logger.info("Fitted time and error (LOOCV,RMS)")
        logger.info(f"{fitting_results.time}, {fitting_results.loocv}, {fitting_results.rmse}")
        logger.info("Fitted KECI and empty cluster E_kra")
        logger.info(f"{kmc.keci}, {kmc.empty_cluster}")

        logger.info("Fitted time and error (LOOCV,RMS) (site energy)")
        logger.info(
            f"{fitting_results_site.time}, {fitting_results_site.loocv}, {fitting_results_site.rmse}"
        )
        logger.info("Fitted KECI and empty cluster (site energy or E_end)")
        logger.info(f"{kmc.keci_site}, {kmc.empty_cluster_site}")
        logger.info(f"Lists for each event {events_site_list}")
        logger.info(f"Hopping probabilities: {kmc.prob_list}")
        logger.info(f"Cumulative sum of hopping probabilities: {kmc.prob_cum_list}")
        logger.info(f"Loading event kernel from: {inputset.event_kernel}")
        kmc.load_site_event_list(fname = inputset.event_kernel)
        return kmc

    def load_site_event_list(
        self, fname="../event_kernal.csv"
    ):  # workout the site_event_list -> site_event_list[site_index] will return a list of event index to update if a site_index is chosen
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
        events,
        random_seed=123456,
        rng=np.random.default_rng(),
    ) -> tuple[Event, float]:
        """propose a new event to be updated by update()

        Args:
            events (list): list of event
            random_seed (int or bool, optional): random seed, if None, then no randomseed. Defaults to 114514.
            rng (np.random.generator object, optional): a random number generator object. Defaults to np.random.default_rng(). Theoratically this function will receive a random number generator object as a input

        Returns:
            event and dt: what event is chosed by the random, and the time for this event to occur
        """
        proposed_event_index, dt = _propose(prob_cum_list = self.prob_cum_list, rng = rng)
        event = events[proposed_event_index]
        return event, dt
    

    def update(self, event, events):
        self.occ_global[event.mobile_ion_specie_1_index] *= -1
        self.occ_global[event.mobile_ion_specie_2_index] *= -1
        events_to_be_updated = copy(
            self.site_event_list[event.mobile_ion_specie_2_index]
        )  # event_to_be_updated= list, include the indices, of the event that need to be updated.
        for e_index in events_to_be_updated:
            events[e_index].update_event(
                self.occ_global,
                self.inputset.v,
                self.inputset.T,
                self.keci,
                self.empty_cluster,
                self.keci_site,
                self.empty_cluster_site,
            )
            self.prob_list[e_index] = copy(events[e_index].probability)
        self.prob_cum_list = np.cumsum(self.prob_list)

    def run(self, inputset: InputSet, label: str = None):
        """Run KMC simulation from an InputSet object.

        Args:
            inputset (InputSet): InputSet object containing all necessary parameters.
            label (str, optional): Label for the simulation run. Defaults to None.

        Returns:
            kmcpy.tracker.Tracker: Tracker object containing simulation results.
        """
        self.inputset = inputset
        self.rng = np.random.default_rng(seed=inputset.random_seed)

        logger.info(f"Simulation condition: v = {inputset.v} T = {inputset.T}")
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
        for current_pass in np.arange(inputset.equ_pass):
            for _ in np.arange(pass_length):
                event, dt = self.propose(
                    self.events,
                    random_seed=inputset.random_seed,
                    rng=self.rng,
                )
                self.update(event, self.events)

        logger.info("Start running kMC ...")

        tracker = Tracker.from_inputset(inputset = inputset,
                                        structure = self.structure,
                                        occ_initial = self.occ_global)
        
        logger.info(
            "Pass\tTime\t\tMSD\t\tD_J\t\tD_tracer\tConductivity\tH_R\t\tf"
        )
        for current_pass in np.arange(inputset.kmc_pass):
            for this_kmc in np.arange(pass_length):
                event, dt = self.propose(
                    self.events,
                    random_seed=inputset.random_seed,
                    rng=self.rng,
                )
                tracker.update(event, self.occ_global, dt)
                self.update(event, self.events)

            previous_conduct = tracker.summary(current_pass)
            tracker.show_current_info(current_pass)

        tracker.write_results(current_pass, self.occ_global, label= label)
        return tracker

    def as_dict(self):
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

    def to_json(self, fname):
        logger.info(f"Saving: {fname}")
        with open(fname, "w") as fhandle:
            d = self.as_dict()
            jsonStr = json.dumps(
                d, indent=4, default=convert
            )  # to get rid of errors of int64
            fhandle.write(jsonStr)

    @classmethod
    def from_json(cls, fname):
        logger.info(f"Loading: {fname}")
        with open(fname, "rb") as fhandle:
            objDict = json.load(fhandle)
        obj = KMC()
        obj.__dict__ = objDict
        logger.info("load complete")
        return obj


def _convert_list(list_to_convert):
    converted_list = List([List(List(j) for j in i) for i in list_to_convert])
    return converted_list


@njit
def _propose(prob_cum_list, rng):
    random_seed = rng.random()
    random_seed_2 = rng.random()
    proposed_event_index = np.searchsorted(
        prob_cum_list / prob_cum_list[-1], random_seed, side="right"
    )
    dt = (-1.0 / prob_cum_list[-1]) * np.log(random_seed_2)
    return proposed_event_index, dt