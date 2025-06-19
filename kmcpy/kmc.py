"""
Function and classes for running kMC

Author: Zeyu Deng
Email: dengzeyu@gmail.com
"""

import random
import numpy as np
from numba.typed import List
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

logger = logging.getLogger(__name__) 

class KMC:
    """
    main function of kinetic monte carlo
    """

    def __init__(self):
        pass

    def initialization(
        self,
        occ=np.array(
            [
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ]
        ),
        prim_fname="./inputs/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",
        fitting_results="./inputs/fitting_results.json",
        fitting_results_site="./inputs/fitting_results_site.json",
        event_fname="./inputs/events.json",
        supercell_shape=[2, 1, 1],
        v=5000000000000,
        T=298,
        lce_fname="./inputs/lce.json",
        lce_site_fname="./inputs/lce_site.json",
        immutable_sites=["Zr", "O"],
        convert_to_primitive_cell=False,
        **kwargs
    ):
        """
        XIEWEIHANG 220608

        the 3rd version of initialization process.


        Args:
            occ (np.array, optional): this is the chebyshev occupation of sites, representing the initial state of the model. Defaults to np.array([-1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1]).
            prim_fname (str, optional): the primitive cell of the input structure. Defaults to './inputs/prim.cif'.
            fitting_results (str, optional): This is the fitting results matrix, related to the activation barrier E_kra. Defaults to './inputs/fitting_results.json'.
            fitting_results_site (str, optional):  This is the fitting results matrix, related to the site energy E_end Defaults to './inputs/fitting_results_site.json'.
            event_fname (str, optional): record all possible event. Defaults to "./inputs/events.json".
            supercell_shape (list, optional): _description_. Defaults to [2,1,1].
            v (int, optional): vibration frequency. Defaults to 5000000000000.
            T (int, optional): temperature. Defaults to 298.
            lce_fname (str, optional): This contains all lce orbit related to the activation barrier E_kra. Defaults to "./inputs/lce.json".
            lce_site_fname (str, optional): This contains all lce orbit related to the activation barrier E_end. Defaults to "./inputs/lce_site.json".
            immutable_sites (list, optional):the sites that do not participate in the monte carlo process. For example,. in NaSICON, the Zr and O do not participate, the Na/Vac and P/S pairs are considered. Defaults to ["Zr","O"].
            convert_to_primitive_cell(bool): whether or not convert input cif file to primitive cell
            **kwargs: other parameters that are not used in this function. This is for the compatibility of different versions of initialization function.
        Returns:
            list: a list of kmc.event.Event object
        """
        logger.info(kmcpy.get_logo())
        logger.info(f"Initializing kMC calculations with pirm.cif at {prim_fname} ...")
        self.structure = StructureKMCpy.from_cif(
            prim_fname, primitive=convert_to_primitive_cell
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

        logger.info("Loading occupation: {occ}")
        self.occ_global = copy(occ)

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
            T,
            self.keci,
            self.empty_cluster,
            self.keci_site,
            self.empty_cluster_site,
            )
            events_site_list.append(event.local_env_indices_list)
            events.append(event)

        logger.info("Initializing hopping probabilities ...")
        self.prob_list = [event.probability for event in events]
        self.prob_cum_list = np.cumsum(self.prob_list)

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
        logger.info(f"Parameters that are not used during the initialization process: {kwargs}")
        return events

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
        **kwargs
    ):  # propose() will propose an event to be updated by update()
        """propose a new event to be updated by update()

        Args:
            events (list): list of event
            random_seed (int or bool, optional): random seed, if None, then no randomseed. Defaults to 114514.
            rng (np.random.generator object, optional): a random number generator object. Defaults to np.random.default_rng(). Theoratically this function will receive a random number generator object as a input

        Returns:
            event and time_change: what event is chosed by the random, and the time for this event to occur
        """

        random_seed = rng.random()
        random_seed_2 = rng.random()
        proposed_event_index = np.searchsorted(
            self.prob_cum_list / (self.prob_cum_list[-1]), random_seed, side="right"
        )
        time_change = (-1.0 / self.prob_cum_list[-1]) * np.log(random_seed_2)

        event = events[proposed_event_index]

        return event, time_change
    
    def update(self, event, events):
        self.occ_global[event.mobile_ion_specie_1_index] *= -1
        self.occ_global[event.mobile_ion_specie_2_index] *= -1
        events_to_be_updated = copy(
            self.site_event_list[event.mobile_ion_specie_2_index]
        )  # event_to_be_updated= list, include the indices, of the event that need to be updated.
        for e_index in events_to_be_updated:
            events[e_index].update_event(
                self.occ_global,
                self.v,
                self.T,
                self.keci,
                self.empty_cluster,
                self.keci_site,
                self.empty_cluster_site,
            )
            self.prob_list[e_index] = copy(events[e_index].probability)
        self.prob_cum_list = np.cumsum(self.prob_list)

    def run_from_database(
        self,
        kmc_pass=1000,
        equ_pass=1,
        v=5000000000000,
        T=298,
        events="./inputs/events.json",
        comp=1,
        random_seed=114514,
        mobile_ion_specie="Na",
        q=1.0,
        dimension=3,
        structure_idx=1,
        elem_hop_distance=3.47782,
        **kwargs
    ):
        """kmc main function version 3. Previous versions are removed for readability

        Args:
            kmc_pass (int, optional): number of pass to run. Defaults to 1000.
            equ_pass (int, optional): _description_. Defaults to 1.
            v (int, optional): refer to paper. Defaults to 5000000000000.
            T (int, optional): refer to paper, temperature. Defaults to 298.
            events (str, optional): path to event.json. Defaults to "./inputs/events.json".
            comp (int, optional): composition, refer to paper. Defaults to 1.
            random_seed (int, optional): random seed for KMC event propose. Defaults to 114514.
            mobile_ion_specie (str, optional): mobile ion specie to track. Defaults to 'Na'.
            q (float, optional): charge of mobile ion specie. Defaults to 1.0.
            dimension (int, optional): dimension of migration. For LiCoO2 it is 2 (2D migration). for NaSICON it is 3. Defaults to 3.
            structure_idx (int, optional): structure index . Defaults to 1.
            elem_hop_distance (float, optional): the hopping distance for the mobile ion specie, for NasiCON, this is the distance between Na1 and nearest Na2 site, for LiCoO2, this is the distance between 2 Li site. Defaults to 3.47782.

        Returns:
            kmcpy.tracker.Tracker: return the tracker for testing unit to assert some functions.
        """
        self.rng = np.random.default_rng(seed=random_seed)

        logger.info(f"Simulation condition: v = {v} T = {T}")
        self.v = v
        self.T = T
        pass_length = len(
            [
            el.symbol
            for el in self.structure.species
            if mobile_ion_specie in el.symbol
            ]
        )
        logger.info("============================================================")
        logger.info("Start running kMC ... ")
        logger.info("Initial occ_global, prob_list and prob_cum_list")
        logger.info("Starting Equilbrium ...")
        for current_pass in np.arange(equ_pass):
            for this_kmc in np.arange(pass_length):
                event, time_change = self.propose(
                    events,
                    random_seed=random_seed,
                    rng=self.rng,
                )
                self.update(event, events)

        logger.info("Start running kMC ...")
        tracker = Tracker()

        tracker.initialization(
            occ_initial=self.occ_global,
            structure=self.structure,
            T=self.T,
            v=self.v,
            q=q,
            mobile_ion_specie=mobile_ion_specie,
            dimension=dimension,
            elem_hop_distance=elem_hop_distance,
            **kwargs
        )
        logger.info(
            "Pass\tTime\t\tMSD\t\tD_J\t\tD_tracer\tConductivity\tH_R\t\tf"
        )
        for current_pass in np.arange(kmc_pass):
            for this_kmc in np.arange(pass_length):
                event, time_change = self.propose(
                    events,
                    random_seed=random_seed,
                    rng=self.rng,
                )
                tracker.update(event, self.occ_global, time_change)
                self.update(event, events)

            previous_conduct = tracker.summary(comp, current_pass)
            tracker.show_current_info(current_pass)

        tracker.write_results(comp, structure_idx, current_pass, self.occ_global)
        logger.debug("kmc.KMC.run is called. ")
        return tracker

    # def run(self,kmc_pass,equ_pass,v,T,events):
    #     print('Simulation condition: v =',v,'T = ',T)
    #     self.v = v
    #     self.T = T
    #     pass_length = len([el.symbol for el in self.structure.species if 'Na' in el.symbol])
    #     print('============================================================')
    #     print('Start running kMC ... ')
    #     print('\nInitial occ_global, prob_list and prob_cum_list')
    #     print('Starting Equilbrium ...')
    #     for current_pass in np.arange(equ_pass):
    #         for this_kmc in np.arange(pass_length):
    #             event,time_change = self.propose(events)
    #             self.update(event,events)

    #     print('Start running kMC ...')
    #     tracker = Tracker()
    #     tracker.initialization(self.occ_global,self.structure,self.T,self.v)
    #     print('Pass\tTime\t\tMSD\t\tD_J\t\tD_tracer\tConductivity\tH_R\t\tf\t\tOccNa(1)\tOccNa(2)')
    #     for current_pass in np.arange(kmc_pass):
    #         for this_kmc in np.arange(pass_length):
    #             event,time_change = self.propose(events)
    #             tracker.update(event,self.occ_global,time_change)
    #             self.update(event,events)

    #         previous_conduct = tracker.summary(comp,current_pass)
    #         tracker.show_current_info(current_pass)

    #     tracker.write_results(None,None,current_pass,self.occ_global)

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
