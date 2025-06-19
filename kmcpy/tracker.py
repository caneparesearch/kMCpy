"""
Tracker is an object to track trajectories of diffusing ions

Author: Zeyu Deng
Email: dengzeyu@gmail.com
"""

import numpy as np
import numpy as np
import pandas as pd
from copy import copy
import json
from kmcpy.io import convert
import logging

logger = logging.getLogger(__name__) 

class Tracker:
    """
    Tracker has a data structure of tracker[na_si_idx]
    """

    def __init__(self):
        pass

    def initialization(
        self,
        occ_initial=[1, -1, -1, 1],
        structure=None,
        T=298,
        v=5e13,
        q=1.0,
        mobile_ion_specie="Na",
        dimension=3,
        elem_hop_distance=3.4778,
        **kwargs
    ):
        """

        220609

        XIEWEIHANG

        updated initialization function

        Args:

            occ_initial (list, optional): list for the initial occupation received from io.Inputset(). Defaults to [1,-1,-1,1].

            structure (external.pymatgen_structure.Structure, optional): structure object for extracting the mobile ion content. Defaults to None.

            T (int, optional): temperature. Defaults to 298.

            v (float, optional): frequency constant, see the article for detail. Defaults to 5E13.

            q (float, optional): charge of mobile ion specie, for Na ion, it is 1.0. Defaults to 1.0.

            mobile_ion_specie (str, optional): mobile ion specie identifier to search from the structure. Defaults to "Na".

            dimension (int, optional): dimension of migration, for NaSICON it is 3D migration so dimension=3, for LiCoO2 it is 2D migration so dimension=2. Defaults to 3.

            elem_hop_distance (float, optional): hopping distance of mobile ion. IN nasicon, this is the distance in Angstrom from Na1 to its nearest Na2. Planning to automatically calculate it. Defaults to 3.4778.

        """
        logger.info("Initializing Tracker ...")
        self.dimension = dimension
        self.q = q
        self.elem_hop_distance = elem_hop_distance
        self.T = T
        self.v = v
        self.occ_initial = copy(occ_initial)
        self.frac_coords = structure.frac_coords
        self.latt = structure.lattice
        self.volume = structure.volume
        self.mobile_ion_specie = mobile_ion_specie
        self.n_mobile_ion_specie_site = len(
            [el.symbol for el in structure.species if mobile_ion_specie in el.symbol]
        )
        self.mobile_ion_specie_locations = np.where(
            self.occ_initial[0 : self.n_mobile_ion_specie_site] == -1
        )[
            0
        ]  # na_si_site_indices[na_si_indices]
        logger.debug('Initial mobile ion locations = %s', self.mobile_ion_specie_locations)
        self.n_mobile_ion_specie = len(self.mobile_ion_specie_locations)

        logger.info("number of mobile ion specie = %d", self.n_mobile_ion_specie)

        self.displacement = np.zeros(
            (len(self.mobile_ion_specie_locations), 3)
        )  # displacement stores the displacement vector for each ion
        self.hop_counter = np.zeros(
            len(self.mobile_ion_specie_locations), dtype=np.int64
        )
        self.time = 0
        # self.barrier = []
        self.results = {
            "time": [],
            "D_J": [],
            "D_tracer": [],
            "conductivity": [],
            "f": [],
            "H_R": [],
            "msd": [],
        }

        logger.info(
            "Center of mass (Na): %s",
            np.mean(
            self.frac_coords[self.mobile_ion_specie_locations] @ self.latt.matrix,
            axis=0,
            ),
        )
        self.r0 = self.frac_coords[self.mobile_ion_specie_locations] @ self.latt.matrix


    def update(
        self, event, current_occ, time_change
    ):  # this should be called after update() of KMC run
        mobile_ion_specie_1_coord = copy(
            self.frac_coords[event.mobile_ion_specie_1_index]
        )
        mobile_ion_specie_2_coord = copy(
            self.frac_coords[event.mobile_ion_specie_2_index]
        )
        mobile_ion_specie_1_occ = current_occ[event.mobile_ion_specie_1_index]
        mobile_ion_specie_2_occ = current_occ[event.mobile_ion_specie_2_index]

        logger.debug('---------------------Tracker info---------------------')
        logger.debug('%s(1) at %s with idx: %s %s', self.mobile_ion_specie,mobile_ion_specie_1_coord, event.mobile_ion_specie_1_index, mobile_ion_specie_1_occ)
        logger.debug('%s(2) at %s with idx: %s %s', self.mobile_ion_specie,mobile_ion_specie_2_coord, event.mobile_ion_specie_2_index, mobile_ion_specie_2_occ)
        logger.debug('Time now: %s', self.time)
        logger.debug('Hop counter: %s', self.hop_counter)
        logger.debug('Probability: %s', event.probability)
        logger.debug('Before update %s locations = %s', self.mobile_ion_specie,self.mobile_ion_specie_locations)
        logger.debug('Occupation before update: %s', current_occ)
        direction = int(
            (mobile_ion_specie_2_occ - mobile_ion_specie_1_occ) / 2
        )  # na1 -> na2 direction = 1; na2 -> na1 direction = -1
        displacement_frac = copy(
            direction * (mobile_ion_specie_2_coord - mobile_ion_specie_1_coord)
        )
        displacement_frac -= np.array(
            [int(round(i)) for i in displacement_frac]
        )  # for periodic condition
        displacement_cart = copy(self.latt.get_cartesian_coords(displacement_frac))
        if direction == -1:  # Na(2) -> Na(1)
            logger.debug(f'Diffuse direction: {self.mobile_ion_specie}(2) -> {self.mobile_ion_specie}(1)')
            specie_to_diff = np.where(
            self.mobile_ion_specie_locations == event.mobile_ion_specie_2_index
            )[0][0]
            self.mobile_ion_specie_locations[specie_to_diff] = (
            event.mobile_ion_specie_1_index
            )
        elif direction == 1:  # Na(1) -> Na(2)
            logger.debug(f'Diffuse direction: {self.mobile_ion_specie}(1) -> {self.mobile_ion_specie}(2)')
            specie_to_diff = np.where(
            self.mobile_ion_specie_locations == event.mobile_ion_specie_1_index
            )[0][0]
            self.mobile_ion_specie_locations[specie_to_diff] = (
            event.mobile_ion_specie_2_index
            )
        else:
            logger.error("Proposed a wrong event! Please check the code!")
        self.displacement[specie_to_diff] += copy(np.array(displacement_cart))
        self.hop_counter[specie_to_diff] += 1
        self.time += time_change
        # self.frac_na_at_na1.append(np.count_nonzero(self.mobile_ion_specie_location < self.n_mobile_ion_specie_site/4)/self.n_mobile_ion_specie)

    def calc_D_J(self, d=3):
        displacement_vector_tot = np.linalg.norm(np.sum(self.displacement, axis=0))

        D_J = (
            displacement_vector_tot**2
            / (2 * self.dimension * self.time * self.n_mobile_ion_specie)
            * 10 ** (-16)
        )  # to cm^2/s

        return D_J

    def calc_D_tracer(self, d=3):
        D_tracer = (
            np.mean(np.linalg.norm(self.displacement, axis=1) ** 2)
            / (2 * self.dimension * self.time)
            * 10 ** (-16)
        )

        return D_tracer

    def calc_corr_factor(self):  # a is the hopping distance in Angstrom
        corr_factor = np.linalg.norm(self.displacement, axis=1) ** 2 / (
            self.hop_counter * self.elem_hop_distance**2
        )

        corr_factor = np.nan_to_num(corr_factor, nan=0)

        return np.mean(corr_factor)

    def calc_conductivity(self, D_J, D_tracer, q=1, T=300):

        k = 8.617333262145 * 10 ** (-2)  # unit in meV/K

        n = (
            self.n_mobile_ion_specie
        ) / self.volume  # e per Angst^3 vacancy is the carrier
        conductivity = D_J * n * self.q**2 / (k * self.T) * 1.602 * 10**11  # to mS/cm

        return conductivity
    
    def show_current_info(self, current_pass):
        logger.info(
            "%d\t%.3E\t%.3E\t%.3E\t%.3E\t%.3E\t%.3E\t%.3E",
            current_pass,
            self.time,
            self.results["msd"][-1],
            self.results["D_J"][-1],
            self.results["D_tracer"][-1],
            self.results["conductivity"][-1],
            self.results["H_R"][-1],
            self.results["f"][-1],
        )
        logger.debug('Center of mass (%s): %s', self.mobile_ion_specie, np.mean(self.frac_coords[self.mobile_ion_specie_locations] @ self.latt.matrix, axis=0))
        logger.debug('MSD = %s, time = %s', np.linalg.norm(np.sum(self.displacement, axis=0))**2, self.time)

    def return_current_info(self):
        return (
            self.time,
            self.results["msd"][-1],
            self.results["D_J"][-1],
            self.results["D_tracer"][-1],
            self.results["conductivity"][-1],
            self.results["H_R"][-1],
            self.results["f"][-1],
        )

    def summary(self, comp, current_pass):
        logger.debug('\nTracker Summary:')
        logger.debug('comp = %s', comp)
        logger.debug('Time elapsed: %s', self.time)
        logger.debug('Current pass: %s', current_pass)
        logger.debug('T = %s K, v = %s', self.T, self.v)
        logger.debug(f'{self.mobile_ion_specie} ratio ({self.mobile_ion_specie}/({self.mobile_ion_specie}+Va)) = {self.n_mobile_ion_specie/self.n_mobile_ion_specie_site}')
        # logger.debug('Si ratio (Si/(Si+P)) = %s', self.n_si/self.n_si_sites)
        # logger.debug('Displacement vectors r_i = %s', self.displacement)
        # logger.debug('Hopping counts n_i = %s', self.hop_counter)
        # logger.debug('average n_Na%% @ Na(1) = %s', sum(self.frac_na_at_na1)/len(self.frac_na_at_na1))
        # logger.debug('final n_Na%% @ Na(1) = %s', self.frac_na_at_na1[-1])
        # logger.debug('final Occ Na(1): %s', (4-3*comp)*self.frac_na_at_na1[-1])
        # logger.debug('final Occ Na(2): %s', (4-3*comp)/3*(1-self.frac_na_at_na1[-1]))

        D_J = self.calc_D_J()
        D_tracer = self.calc_D_tracer()
        f = self.calc_corr_factor()
        conductivity = self.calc_conductivity(
            D_J=D_J, D_tracer=D_tracer, q=self.q, T=self.T
        )
        H_R = D_tracer / D_J

        msd = np.mean(
            np.linalg.norm(self.displacement, axis=1) ** 2
        )  # MSD = sum_i(|r_i|^2)/N

        logger.debug("Haven's ratio H_R = %s", H_R)

        self.results["D_J"].append(D_J)
        self.results["D_tracer"].append(D_tracer)
        self.results["f"].append(f)
        self.results["H_R"].append(H_R)
        self.results["conductivity"].append(conductivity)
        self.results["time"].append(copy(self.time))
        self.results["msd"].append(msd)

        return conductivity

    def write_results(self, comp, structure_idx, current_pass, current_occupation):
        np.savetxt(
            "displacement_"
            + str(comp)
            + "_"
            + str(structure_idx)
            + "_"
            + str(current_pass)
            + ".csv.gz",
            self.displacement,
            delimiter=",",
        )
        np.savetxt(
            "hop_counter_"
            + str(comp)
            + "_"
            + str(structure_idx)
            + "_"
            + str(current_pass)
            + ".csv.gz",
            self.hop_counter,
            delimiter=",",
        )
        #    np.savetxt('barrier_'+str(comp)+'_'+str(structure_idx)+'_'+str(current_pass)+'.csv.gz',self.barrier,delimiter=',')
        np.savetxt(
            "current_occ_"
            + str(comp)
            + "_"
            + str(structure_idx)
            + "_"
            + str(current_pass)
            + ".csv.gz",
            current_occupation,
            delimiter=",",
        )

        df = pd.DataFrame(self.results)
        df.to_csv(
            "results_" + str(comp) + "_" + str(structure_idx) + ".csv.gz",
            compression="gzip",
        )

    def as_dict(self):

        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "T": self.T,
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
            "r0": self.r0,
        }
        return d

    def to_json(self, fname):
        logger.info("Saving: %s", fname)
        with open(fname, "w") as fhandle:
            d = self.as_dict()
            jsonStr = json.dumps(
                d, indent=4, default=convert
            )  # to get rid of errors of int64
            fhandle.write(jsonStr)

    @classmethod
    def from_json(self, fname):
        logger.info("Loading: %s", fname)
        with open(fname, "rb") as fhandle:
            objDict = json.load(fhandle)
        obj = Tracker()
        obj.__dict__ = objDict
        return obj
