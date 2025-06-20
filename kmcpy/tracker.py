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
from kmcpy.io import convert, InputSet, Results
import logging
from kmcpy.external.structure import StructureKMCpy

logger = logging.getLogger(__name__) 

class Tracker:
    """
    Tracker has a data structure of tracker[na_si_idx]
    """

    def __init__(self, occ_initial:list, structure:StructureKMCpy, mobile_ion_specie:str,  elem_hop_distance:float,
                 dimension:int=3, q:float=1.0,temperature:float=300, v:float=1e13, **kwargs):
        """
        Initialize a Tracker object for monitoring mobile ion species in a structure during kinetic Monte Carlo simulations.
        Parameters
        ----------
        occ_initial : list
            Initial occupation list indicating the occupancy of each site.
        structure : StructureKMCpy
            Structure object containing fractional coordinates, lattice, and species information.
        mobile_ion_specie : str
            Symbol of the mobile ion species to be tracked (e.g., 'Na').
        elem_hop_distance : float
            Elementary hop distance for the mobile ion species.
        dimension : int, optional
            Dimensionality of the system (default is 3).
        q : float, optional
            Charge of the mobile ion species (default is 1.0).
        temperature : float, optional
            Simulation temperature in Kelvin (default is 300).
        v : float, optional
            Attempt frequency (pre-exponential factor) in Hz (default is 1e13).
        Notes
        -----
        - Initializes displacement and hop counters for each mobile ion.
        - Identifies initial locations of mobile ions in the structure.
        - Computes and logs the center of mass for the mobile ions.
        """
        logger.info("Initializing Tracker ...")

        self.dimension = dimension
        self.q = q
        self.elem_hop_distance = elem_hop_distance
        self.temperature = temperature
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
        self.results = Results()

        logger.info(
            f"Center of mass ({mobile_ion_specie}): {np.mean(
            self.frac_coords[self.mobile_ion_specie_locations] @ self.latt.matrix,
            axis=0,
            )}")
        self.r0 = self.frac_coords[self.mobile_ion_specie_locations] @ self.latt.matrix

    @classmethod
    def from_inputset(cls, inputset:InputSet,
                      structure:StructureKMCpy,
                      occ_initial:list,
                      ):
        """
        Create a Tracker object from an InputSet object.
        Args:
            inputset (InputSet): An InputSet object containing the necessary parameters for Tracker initialization.
            structure (StructureKMCpy): A StructureKMCpy object containing the structure information.
            occ_initial (list): Initial occupation list for the mobile ion specie.
        Returns:
            Tracker: An instance of the Tracker class initialized
        """
        params = {k: v for k, v in inputset._parameters.items() if k != "task"}
        return cls(structure=structure, occ_initial=occ_initial, **params)


    def update(self, event, current_occ, time_change):  # this should be called after update() of KMC run
        """
        Update the tracker state after a KMC event.

        This method should be called after the KMC run's update() method. It updates the positions, occupation, displacement,
        hop counters, and simulation time for mobile ions based on the provided event and current occupation.

        Args:
            event: An object representing the KMC event, containing indices and properties of the mobile ions involved.
            current_occ (np.ndarray): The current occupation array indicating the occupation state of each site.
            time_change (float): The time increment to add to the simulation time.

        Side Effects:
            - Updates the internal state of the tracker, including:
                - `mobile_ion_specie_locations`: The indices of the mobile ions after the event.
                - `displacement`: The cumulative displacement of each mobile ion.
                - `hop_counter`: The number of hops performed by each mobile ion.
                - `time`: The current simulation time.
            - Logs detailed debug information if the logger is set to DEBUG level.

        Raises:
            Logs an error if the event direction cannot be determined (i.e., if the event is invalid).
        """
        mobile_ion_specie_1_coord = copy(
            self.frac_coords[event.mobile_ion_specie_1_index]
        )
        mobile_ion_specie_2_coord = copy(
            self.frac_coords[event.mobile_ion_specie_2_index]
        )
        mobile_ion_specie_1_occ = current_occ[event.mobile_ion_specie_1_index]
        mobile_ion_specie_2_occ = current_occ[event.mobile_ion_specie_2_index]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('--------------------- Tracker Update Start ---------------------')
            logger.debug(
            "%s(1): idx=%d, coord=%s, occ=%s",
            self.mobile_ion_specie, event.mobile_ion_specie_1_index, np.array2string(mobile_ion_specie_1_coord, precision=4), mobile_ion_specie_1_occ
            )
            logger.debug(
            "%s(2): idx=%d, coord=%s, occ=%s",
            self.mobile_ion_specie, event.mobile_ion_specie_2_index, np.array2string(mobile_ion_specie_2_coord, precision=4), mobile_ion_specie_2_occ
            )
            logger.debug("Current simulation time: %.6f", self.time)
            logger.debug("Hop counters: %s", np.array2string(self.hop_counter, precision=0, separator=', '))
            logger.debug("Event probability: %s", getattr(event, "probability", None))
            logger.debug("Mobile ion locations before update: %s", self.mobile_ion_specie_locations)
            logger.debug("Occupation before update: %s", np.array2string(current_occ, precision=0, separator=', '))
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
        logger.debug('------------------------ Tracker Update End --------------------')
        # self.frac_na_at_na1.append(np.count_nonzero(self.mobile_ion_specie_location < self.n_mobile_ion_specie_site/4)/self.n_mobile_ion_specie)

    def calc_D_J(self):
        """
        Calculate the center of the mass diffusion coefficient (D_J) based on the total displacement vector.

        This method computes the diffusion coefficient using the total displacement
        of mobile ions (in Angstrom) over a given time period, normalized by the system's dimensionality,
        the number of mobile ion species, and the elapsed time. The result is converted to
        units of cm^2/s.

        Returns:
            float: The calculated diffusion coefficient (D_J) in cm^2/s.
        """
        displacement_vector_tot = np.linalg.norm(np.sum(self.displacement, axis=0))

        D_J = (
            displacement_vector_tot**2
            / (2 * self.dimension * self.time * self.n_mobile_ion_specie)
            * 10 ** (-16)
        )  # to cm^2/s

        return D_J

    def calc_D_tracer(self):
        """
        Calculate the tracer diffusivity (D_tracer).

        This method computes the tracer diffusivity based on the mean squared displacement
        of particles over time, normalized by the system's dimensionality and total elapsed time.
        The result is converted to in cm^2/s.

        Returns:
            float: The calculated tracer diffusivity.
        """
        D_tracer = (
            np.mean(np.linalg.norm(self.displacement, axis=1) ** 2)
            / (2 * self.dimension * self.time)
            * 10 ** (-16)
        )

        return D_tracer

    def calc_corr_factor(self):  # a is the hopping distance in Angstrom
        """
        Calculate the correlation factor for the tracked hops.

        The correlation factor is computed as the mean squared norm of the displacement
        vectors divided by the product of the number of hops and the square of the 
        elementary hop distance. NaN values in the result are replaced with zero.

        Returns:
            float: The mean correlation factor for the tracked hops.
        """
        corr_factor = np.linalg.norm(self.displacement, axis=1) ** 2 / (
            self.hop_counter * self.elem_hop_distance**2
        )

        corr_factor = np.nan_to_num(corr_factor, nan=0)

        return np.mean(corr_factor)

    def calc_conductivity(self, D_J):

        k = 8.617333262145 * 10 ** (-2)  # unit in meV/K

        n = (
            self.n_mobile_ion_specie
        ) / self.volume  # e per Angst^3 vacancy is the carrier
        conductivity = D_J * n * self.q**2 / (k * self.temperature) * 1.602 * 10**11  # to mS/cm

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

    def summary(self, current_pass):
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
        conductivity = self.calc_conductivity(D_J=D_J)
        H_R = D_tracer / D_J

        msd = np.mean(
            np.linalg.norm(self.displacement, axis=1) ** 2
        )  # MSD = sum_i(|r_i|^2)/N

        summary_data = [
            ["Time elapsed", self.time],
            ["Current pass", current_pass],
            ["Temperature (K)", self.temperature],
            ["Attempt frequency (v)", self.v],
            [f"{self.mobile_ion_specie} ratio ({self.mobile_ion_specie}/({self.mobile_ion_specie}+Va))", self.n_mobile_ion_specie/self.n_mobile_ion_specie_site],
            ["Haven's ratio H_R", H_R],
        ]
        table_str = "\n" + pd.DataFrame(summary_data, columns=["Property", "Value"]).to_string(index=False)
        logger.debug('Tracker Summary:%s', table_str)

        self.results.add(
            copy(self.time), D_J, D_tracer, conductivity, f, H_R, msd
        )

        return conductivity

    def write_results(self,  current_pass:int, current_occupation:list, label:str = None)-> None:
        """
        Save simulation results to compressed CSV files.

        Parameters
        ----------
        current_pass : int
            The current simulation pass or iteration number.
        current_occupation : list
            The current occupation state to be saved.
        label : str, optional
            An optional label to prefix output files. If not provided, files are saved without a label.

        Saves
        -----
        displacement_{label}_{current_pass}.csv.gz : ndarray
            The displacement data for the current pass.
        hop_counter_{label}_{current_pass}.csv.gz : ndarray
            The hop counter data for the current pass.
        current_occ_{label}_{current_pass}.csv.gz : list
            The current occupation data for the current pass.
        results_{label}.csv.gz or results.csv.gz : DataFrame
            The results DataFrame, saved with gzip compression. The filename includes the label if provided.
        """
        prefix = f"{label}_{current_pass}"
        np.savetxt(
            f"displacement_{prefix}.csv.gz",
            self.displacement,
            delimiter=",",
        )
        np.savetxt(
            f"hop_counter_{prefix}.csv.gz",
            self.hop_counter,
            delimiter=",",
        )
        np.savetxt(
            f"current_occ_{prefix}.csv.gz",
            current_occupation,
            delimiter=",",
        )

        if label:
            results_file = f"results_{label}.csv.gz"
        else:
            results_file = "results.csv.gz"
        self.results.to_dataframe().to_csv(results_file, compression="gzip", index=False)

    def as_dict(self)-> dict:
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "T": self.temperature,
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

    def to_json(self, fname)-> None:
        logger.info("Saving: %s", fname)
        with open(fname, "w") as fhandle:
            d = self.as_dict()
            jsonStr = json.dumps(
                d, indent=4, default=convert
            )  # to get rid of errors of int64
            fhandle.write(jsonStr)

    @classmethod
    def from_json(self, fname)-> "Tracker":
        logger.info("Loading: %s", fname)
        with open(fname, "rb") as fhandle:
            objDict = json.load(fhandle)
        obj = Tracker()
        obj.__dict__ = objDict
        return obj
