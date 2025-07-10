#!/usr/bin/env python
"""
This module defines a Tracker class for monitoring mobile ion species in kinetic Monte Carlo (kMC) simulations.
"""

import numpy as np
import pandas as pd
from copy import copy
import json
from kmcpy.io import convert, InputSet, Results
import logging
from kmcpy.external.structure import StructureKMCpy
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from kmcpy.simulation.condition import SimulationConfig
    from kmcpy.simulation.state import SimulationState

logger = logging.getLogger(__name__) 

class Tracker:
    """
    Tracker class for monitoring mobile ion species in kinetic Monte Carlo (kMC) simulations.

    The Tracker class is responsible for tracking the positions, displacements, hop counts, and related transport properties
    of mobile ion species within a given structure during kMC simulations. It provides methods to update the tracked state
    after each kMC event, calculate diffusion coefficients, correlation factors, conductivity, and to summarize and save
    simulation results.
    """

    def __init__(self, config: "SimulationConfig", structure: StructureKMCpy, 
                 initial_state: Optional["SimulationState"] = None, **kwargs) -> None:
        """Initialize a Tracker object for monitoring mobile ion species.

        Args:
            config (SimulationConfig): Configuration object containing simulation parameters.
            structure (StructureKMCpy): Structure object.
            initial_state (SimulationState, optional): Initial simulation state. If None, creates from config or occ_initial.
            **kwargs: Additional legacy parameters for backward compatibility.

        """
        logger.info("Initializing Tracker ...")

        # Store configuration reference (no parameter duplication)
        self.config = config
        self.structure = structure
        
        # Initialize or use provided simulation state
        if initial_state is not None:
            self.state = initial_state
        else:
            raise ValueError("SimulationState must be provided to Tracker")
        
        # Initialize mobile ion tracking in Tracker (where it belongs)
        initial_occ = initial_state.occupations if initial_state else occ_initial
        self._initialize_mobile_ion_tracking(initial_occ)
        
        # Results storage (only responsibility of Tracker)
        self.results = Results()
        self.current_pass = 0
        
        logger.info("number of mobile ion specie = %d", self.n_mobile_ion_specie)
        logger.info(
            f"""Center of mass ({self.mobile_ion_specie}): {np.mean(
            self.r0, axis=0
            )}""")
    
    def _initialize_mobile_ion_tracking(self, initial_occ: list):
        """Initialize mobile ion tracking arrays."""
        import numpy as np
        
        # Find mobile ion sites (working version - mobile ion sites are first N sites)
        self.n_mobile_ion_specie_site = len(
            [el.symbol for el in self.structure.species if self.mobile_ion_specie in el.symbol]
        )
        self.mobile_ion_specie_locations = np.where(
            np.array(initial_occ[0:self.n_mobile_ion_specie_site]) == -1
        )[0]
        self.n_mobile_ion_specie = len(self.mobile_ion_specie_locations)
        
        logger.debug('Initial mobile ion locations = %s', self.mobile_ion_specie_locations)
        
        # Initialize tracking arrays
        self.displacement = np.zeros((self.n_mobile_ion_specie, 3))
        self.hop_counter = np.zeros(self.n_mobile_ion_specie, dtype=np.int64)
        self.r0 = self.frac_coords[self.mobile_ion_specie_locations] @ self.latt.matrix

    # Properties to access state information (delegation to state)
    @property
    def occ_initial(self) -> list:
        """Get initial occupation from state."""
        return self.state.occupations  # Current occupations

    @property
    def frac_coords(self) -> np.ndarray:
        """Get fractional coordinates from structure."""
        return self.structure.frac_coords
    
    @property
    def latt(self):
        """Get lattice from structure."""
        return self.structure.lattice
    
    @property
    def volume(self) -> float:
        """Get volume from structure."""
        return self.structure.volume

    # Properties to access configuration parameters (no duplication)
    @property
    def dimension(self) -> int:
        """Get dimension from configuration."""
        return self.config.dimension
    
    @property
    def q(self) -> float:
        """Get mobile ion charge from configuration."""
        return self.config.mobile_ion_charge
    
    @property
    def elem_hop_distance(self) -> float:
        """Get elementary hop distance from configuration."""
        return self.config.elementary_hop_distance
    
    @property
    def temperature(self) -> float:
        """Get temperature from configuration."""
        return self.config.temperature
    
    @property
    def v(self) -> float:
        """Get attempt frequency from configuration."""
        return self.config.attempt_frequency
    
    @property
    def time(self) -> float:
        """Get current simulation time from state."""
        return self.state.time
    
    @time.setter
    def time(self, value: float):
        """Set simulation time in state (for backward compatibility)."""
        self.state.time = value

    @property
    def mobile_ion_specie(self) -> str:
        """Get mobile ion species from configuration."""
        return self.config.mobile_ion_specie

    @classmethod
    def from_inputset(cls, inputset: InputSet, structure: StructureKMCpy, occ_initial: list) -> "Tracker":
        """
        Create a Tracker object from an InputSet object.
        
        This method maintains backward compatibility by converting InputSet to SimulationConfig.

        Args:
            inputset (InputSet): An InputSet object containing the necessary parameters for Tracker initialization.
            structure (StructureKMCpy): A StructureKMCpy object containing the structure information.
            occ_initial (list): Initial occupation list for the mobile ion specie.
            
        Returns:
            Tracker: An instance of the Tracker class initialized with parameters from the InputSet.
        """
        # Convert InputSet to SimulationConfig for clean architecture
        from kmcpy.simulation.condition import SimulationConfig
        from kmcpy.simulation.state import SimulationState
        config = SimulationConfig.from_inputset(inputset)
        
        # Create SimulationState with initial occupation
        initial_state = SimulationState(initial_occ=occ_initial)
        
        return cls(config=config, structure=structure, initial_state=initial_state)
    
    @classmethod
    def from_config(cls, config: "SimulationConfig", structure: StructureKMCpy, occ_initial: list) -> "Tracker":
        """
        Create a Tracker object from a SimulationConfig object.
        
        This is the preferred method for creating Tracker instances.

        Args:
            config (SimulationConfig): A SimulationConfig object containing simulation parameters.
            structure (StructureKMCpy): A StructureKMCpy object containing the structure information.
            occ_initial (list): Initial occupation list for the mobile ion specie.
            
        Returns:
            Tracker: An instance of the Tracker class.
        """
        # Create SimulationState with initial occupation
        from kmcpy.simulation.state import SimulationState
        initial_state = SimulationState(initial_occ=occ_initial)
        
        return cls(config=config, structure=structure, initial_state=initial_state)


    def update(self, event, current_occ, dt)->None:  # this should be called after update() of KMC run
        """
        Update the tracker state after a KMC event.

        This method should be called after the KMC run's update() method. It updates the positions, occupation, displacement,
        hop counters, and simulation time for mobile ions based on the provided event and current occupation.

        Args:
            event: An object representing the KMC event, containing indices and properties of the mobile ions involved.
            current_occ (np.ndarray): The current occupation array indicating the occupation state of each site.
            dt (float): The time increment to add to the simulation time.

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
            self.frac_coords[event.mobile_ion_indices[0]]
        )
        mobile_ion_specie_2_coord = copy(
            self.frac_coords[event.mobile_ion_indices[1]]
        )
        mobile_ion_specie_1_occ = current_occ[event.mobile_ion_indices[0]]
        mobile_ion_specie_2_occ = current_occ[event.mobile_ion_indices[1]]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('--------------------- Tracker Update Start ---------------------')
            logger.debug(
            "%s(1): idx=%d, coord=%s, occ=%s",
            self.mobile_ion_specie, event.mobile_ion_indices[0], np.array2string(mobile_ion_specie_1_coord, precision=4), mobile_ion_specie_1_occ
            )
            logger.debug(
            "%s(2): idx=%d, coord=%s, occ=%s",
            self.mobile_ion_specie, event.mobile_ion_indices[1], np.array2string(mobile_ion_specie_2_coord, precision=4), mobile_ion_specie_2_occ
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
            self.mobile_ion_specie_locations == event.mobile_ion_indices[1]
            )[0][0]
            self.mobile_ion_specie_locations[specie_to_diff] = (
            event.mobile_ion_indices[0]
            )
        elif direction == 1:  # Na(1) -> Na(2)
            logger.debug(f'Diffuse direction: {self.mobile_ion_specie}(1) -> {self.mobile_ion_specie}(2)')
            specie_to_diff = np.where(
            self.mobile_ion_specie_locations == event.mobile_ion_indices[0]
            )[0][0]
            self.mobile_ion_specie_locations[specie_to_diff] = (
            event.mobile_ion_indices[1]
            )
        else:
            logger.error("Proposed a wrong event! Please check the code!")
        self.displacement[specie_to_diff] += copy(np.array(displacement_cart))
        self.hop_counter[specie_to_diff] += 1
        self.state.time += dt
        logger.debug('------------------------ Tracker Update End --------------------')
        # self.frac_na_at_na1.append(np.count_nonzero(self.mobile_ion_specie_location < self.n_mobile_ion_specie_site/4)/self.n_mobile_ion_specie)

    def update_current_pass(self, current_pass:int)-> None:
        """
        Update the current pass number for the tracker.

        This method updates the current pass number, which is used to track the progress of the simulation.

        Args:
            current_pass (int): The new current pass number.
        """
        self.current_pass = current_pass


    def calc_D_J(self)-> float:
        """
        Calculate the jump diffusivity (D_J) based on the total displacement vector.

        This method computes the jump diffusivity using the total displacement
        of mobile ions (in Angstrom) over a given time period, normalized by the system's dimensionality,
        the number of mobile ion species, and the elapsed time. The result is converted to
        units of cm^2/s.

        Returns:
            float: The calculated jump diffusivity (D_J) in cm^2/s.
        """
        displacement_vector_tot = np.linalg.norm(np.sum(self.displacement, axis=0))

        D_J = (
            displacement_vector_tot**2
            / (2 * self.dimension * self.time * self.n_mobile_ion_specie)
            * 10 ** (-16)
        )  # to cm^2/s

        return D_J

    def calc_D_tracer(self)-> float:
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

    def calc_corr_factor(self)->float:  # a is the hopping distance in Angstrom
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

    def calc_conductivity(self, D_J)-> float:
        """
        Calculate the ionic conductivity based on the jump diffusivity.

        Args:
            D_J (float): Jump diffusivity in units of cm^2/s.

        Returns:
            float: Ionic conductivity in mS/cm.

        Notes:
            - The calculation uses the Nernst-Einstein relation.
            - `self.n_mobile_ion_specie` is the number of mobile ions.
            - `self.volume` is the simulation cell volume in Å^3.
            - `self.q` is the charge of the mobile ion (in elementary charge units).
            - `self.temperature` is the temperature in Kelvin.
            - The Boltzmann constant `k` is used in meV/K.
            - The factor `1.602 * 10**11` converts the units to mS/cm.
        """

        k = 8.617333262145 * 10 ** (-2)  # unit in meV/K

        n = (
            self.n_mobile_ion_specie
        ) / self.volume  # e per Angst^3 vacancy is the carrier
        conductivity = D_J * n * self.q**2 / (k * self.temperature) * 1.602 * 10**11  # to mS/cm

        return conductivity
    
    def show_current_info(self)-> None:
        """
        Logs the current simulation information, including pass number, time, and various computed results.

        The method outputs a summary of the current state using the logger at the INFO level, displaying:
            - Current pass number
            - Simulation time
            - Mean squared displacement (MSD)
            - Jump diffusion coefficient (D_J)
            - Tracer diffusion coefficient (D_tracer)
            - Ionic conductivity
            - Haven ratio (H_R)
            - Correlation factor (f)

        Additionally, it logs at the DEBUG level:
            - The center of mass of the mobile ion species
            - The mean squared displacement and current time

        This method is intended for tracking and debugging the progress of kinetic Monte Carlo simulations.
        """
        logger.info(
            "%d\t%.3E\t%.3E\t%.3E\t%.3E\t%.3E\t%.3E\t%.3E",
            self.current_pass,
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

    def return_current_info(self)-> tuple:
        """
        Returns the current simulation information as a tuple for unit testing purposes.

        Returns:
            tuple: A tuple containing the following elements:
                - self.time (float): The current simulation time.
                - self.results["msd"][-1] (float): The most recent mean squared displacement value.
                - self.results["D_J"][-1] (float): The most recent jump diffusion coefficient.
                - self.results["D_tracer"][-1] (float): The most recent tracer diffusion coefficient.
                - self.results["conductivity"][-1] (float): The most recent conductivity value.
                - self.results["H_R"][-1] (float): The most recent Haven ratio.
                - self.results["f"][-1] (float): The most recent correlation factor.
        """
        return (
            self.time,
            self.results["msd"][-1],
            self.results["D_J"][-1],
            self.results["D_tracer"][-1],
            self.results["conductivity"][-1],
            self.results["H_R"][-1],
            self.results["f"][-1],
        )

    def compute_properties(self)-> None:
        """
        Compute properties of the current simulation state, log key properties, and update results.

        This method calculates and logs various transport properties such as the jump diffusivity (D_J),
        tracer diffusivity (D_tracer), correlation factor (f), ionic conductivity, Haven's ratio (H_R),
        and mean squared displacement (MSD). It also logs a summary table of important simulation parameters and
        adds the computed results to the results tracker.

        """
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

        msd = np.mean(np.linalg.norm(self.displacement, axis=1) ** 2)  # MSD = sum_i(|r_i|^2)/N

        if logger.isEnabledFor(logging.DEBUG):
            summary_data = [
                ["Time elapsed", self.time],
                ["Current pass", self.current_pass],
                ["Temperature (K)", self.temperature],
                ["Attempt frequency (v)", self.v],
                [f"{self.mobile_ion_specie} ratio ({self.mobile_ion_specie}/({self.mobile_ion_specie}+Va))", self.n_mobile_ion_specie/self.n_mobile_ion_specie_site],
                ["Haven's ratio H_R", H_R],
            ]
            table_str = "\n" + pd.DataFrame(summary_data, columns=["Property", "Value"]).to_string(index=False)
            logger.debug('Tracker Summary:%s', table_str)

        self.results.add(copy(self.time), D_J, D_tracer, conductivity, f, H_R, msd)

    def write_results(self, current_occupation:list, label:str = None)-> None:
        """
        Save simulation results to compressed CSV files.

        Args:
            current_occupation (list): The current occupation state to be saved.
            label (str, optional): An optional label to prefix output files.
                If not provided, files are saved without a label. Defaults to None.

        Saves:
            displacement_{label}_{current_pass}.csv.gz (ndarray)
            hop_counter_{label}_{current_pass}.csv.gz (ndarray)
            current_occ_{label}_{current_pass}.csv.gz (list)
            results_{label}.csv.gz or results.csv.gz (DataFrame):
        """
        prefix = f"{label}_{self.current_pass}"
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
