"""
Composite Local Cluster Expansion Model

This module implements a composite model that combines two LocalClusterExpansion
models to provide a unified interface for different energy contributions (e.g., site
energies and barrier energies). This follows an ASE-like design where the model is
composable and generic.

Author: Zeyu Deng
"""

import logging
from typing import Optional
import numpy as np
import pandas as pd
import os

from kmcpy.models.model import CompositeModel
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
from kmcpy.event import Event
from kmcpy.simulator.condition import SimulationCondition
from kmcpy.simulator.state import SimulationState

logger = logging.getLogger(__name__)


class CompositeLCEModel(CompositeModel):
    """
    A composite model that combines exactly two LocalClusterExpansion models.
    
    This class combines two LCE models: one for site energy difference and one for E_KRA. 
    The model is designed to be stateless and generic, following ASE-like 
    patterns where the model is separated from the structure and calculation context.
    
    The composite model provides:
    - compute(): compute transition probability from an event
    
    Example:
        # Create individual LCE models with parameters
        site_model = LocalClusterExpansion(...)
        site_model.load_parameters_from_file("site_parameters.json")
        
        kra_model = LocalClusterExpansion(...)
        kra_model.load_parameters_from_file("kra_parameters.json")
        
        # Combine them
        composite = CompositeLCEModel(site_model, kra_model)
        
        # Use the composite model with SimulationState (preferred)
        probability = composite.compute(
            event=event,
            simulation_condition=simulation_condition,
            simulation_state=simulation_state
        )
    """
    
    def __init__(self, site_model: Optional[LocalClusterExpansion] = None, 
                 kra_model: Optional[LocalClusterExpansion] = None, *args, **kwargs):
        """
        Initialize a composite LCE model with two component models.
        
        Args:
            site_model: model for site energy difference calculations
            kra_model: model for E_KRA calculations
        """
        if site_model is not None and not isinstance(site_model, LocalClusterExpansion):
            raise TypeError(f"Site model must be a LocalClusterExpansion instance, got {type(site_model)}")
        
        if kra_model is not None and not isinstance(kra_model, LocalClusterExpansion):
            raise TypeError(f"KRA model must be a LocalClusterExpansion instance, got {type(kra_model)}")

        models = []
        if site_model:
            models.append(site_model)
        if kra_model:
            models.append(kra_model)

        super().__init__(models=models, *args, **kwargs)
        
        self.site_model = site_model
        self.kra_model = kra_model
        
    def compute(self) -> None:
        """
        This method is not used in the CompositeLCEModel.
        It is provided to maintain compatibility with the BaseModel interface.
        """
        raise NotImplementedError("CompositeLCEModel does not support direct compute method. Use compute_probability instead.")

    def compute_probability(self, event: Event, 
                simulation_condition: SimulationCondition,
                simulation_state:SimulationState) -> float:
        """
        Compute the transition probability for a given event using the composite LCE model.

        This method calculates the transition probability for a migration event by:
        - Computing the site energy (e_site) using the site LocalClusterExpansion model and its stored parameters.
        - Computing the barrier energy (e_kra) using the barrier LocalClusterExpansion model and its stored parameters.
        - Determining the direction of the event from the occupation vector in the SimulationState.
        - Calculating the effective barrier as: e_barrier = e_kra + direction * e_site / 2
        - Using the Arrhenius equation to compute the probability:
              probability = |direction| * v * exp(-e_barrier / (k * T))

        Args:
            event (Event): The migration event, containing mobile ion indices and local environment info.
            simulation_condition (SimulationCondition): Contains attempt frequency (v) and temperature (T).
            simulation_state (SimulationState): Contains the current occupation vector.

        Returns:
            float: The computed transition probability for the event.
        """

        # Boltzmann constant in meV/K (matching event.py)
        k = 8.617333262145 * 10 ** (-2)
        # Compute site energy (esite) using stored parameters
        e_site = self.site_model.compute(simulation_state=simulation_state, event=event)
        # Compute barrier energy (ekra) using stored parameters
        e_kra = self.kra_model.compute(simulation_state=simulation_state, event=event)
        # Get occupation from simulation_state
        occ = simulation_state.occupations
        
        # Determine the direction of the event
        direction = (occ[event.mobile_ion_indices[1]] - occ[event.mobile_ion_indices[0]])/2

        # Calculate effective barrier
        e_barrier = e_kra + direction * e_site / 2
        
        # Get temperature and attempt frequency from SimulationCondition
        temperature = simulation_condition.temperature
        v = simulation_condition.attempt_frequency
        
        # Compute probability using Arrhenius equation
        probability = abs(direction) * v * np.exp(-e_barrier / (k * temperature))
        
        return probability

    def __str__(self):
        return f"CompositeLCEModel(site_model={self.site_model}, kra_model={self.kra_model})"

    def __repr__(self):
        return f"CompositeLCEModel(site_model={self.site_model}, kra_model={self.kra_model})"

    def as_dict(self):
        """
        Convert the CompositeLCEModel to a dictionary representation.
        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "site_model": self.site_model.as_dict() if self.site_model else None,
            "kra_model": self.kra_model.as_dict() if self.kra_model else None,
            "name": self.name,
        }
    
    def build(self, *args, **kwargs):
        """
        Build the composite model based on the provided parameters.
        
        This method is a placeholder and does not perform any specific actions.
        It is provided to maintain compatibility with the BaseModel interface.
        """
        self.site_model.build(*args, **kwargs) if self.site_model else None
        self.kra_model.build(*args, **kwargs) if self.kra_model else None

    def get_occ_from_structure(self, structure, use_model='site_model'):
        """
        Get the occupation vector from a structure.
        
        Args:
            structure (StructureKMCpy): The structure from which to compute the occupation vector.
            use_model (str): Specify which model to use for occupation calculation ('site_model' or 'kra_model').
        
        Returns:
            np.ndarray: The occupation vector for the given structure.
        """
        if use_model == 'site_model' and self.site_model:
            return self.site_model.get_occ_from_structure(structure)
        elif use_model == 'kra_model' and self.kra_model:
            return self.kra_model.get_occ_from_structure(structure)
        else:
            raise ValueError(f"Invalid model specified: {use_model}. Available models are 'site_model' and 'kra_model'.")
    
    def get_corr_from_structure(self, structure, use_model='site_model', tol=1e-2, angle_tol=5):
        """
        Get the correlation vector from a structure.
        
        Args:
            structure (StructureKMCpy): The structure from which to compute the correlation vector.
            use_model (str): Specify which model to use for correlation calculation ('site_model' or 'kra_model').
            tol (float): Tolerance for occupation comparison.
            angle_tol (float): Angle tolerance for structure matching.
        
        Returns:
            np.ndarray: The correlation vector for the given structure.
        """
        if use_model == 'site_model' and self.site_model:
            return self.site_model.get_corr_from_structure(structure, tol=tol, angle_tol=angle_tol)
        elif use_model == 'kra_model' and self.kra_model:
            return self.kra_model.get_corr_from_structure(structure, tol=tol, angle_tol=angle_tol)
        else:
            raise ValueError(f"Invalid model specified: {use_model}. Available models are 'site_model' and 'kra_model'.")
        
    @classmethod
    def from_dict(cls, d):
        """
        Create a CompositeLCEModel from a dictionary.
        """
        from kmcpy.io.io import convert
        site_model = convert(d["site_model"]) if d.get("site_model") else None
        kra_model = convert(d["kra_model"]) if d.get("kra_model") else None
        return cls(site_model=site_model, kra_model=kra_model, name=d.get("name"))

    @classmethod
    def from_json(cls, 
                  lce_fname: str,
                  fitting_results: str,
                  lce_site_fname: str = None,
                  fitting_results_site: str = None) -> "CompositeLCEModel":
        """
        Create a CompositeLCEModel from JSON files.
        
        This method centralizes the loading of LCE models and their parameters,
        replacing the need for file loading logic in other modules.
        
        Args:
            lce_fname: Path to KRA LCE model JSON file
            fitting_results: Path to KRA fitting results JSON file
            lce_site_fname: Path to site LCE model JSON file (optional)
            fitting_results_site: Path to site fitting results JSON file (optional)
            
        Returns:
            CompositeLCEModel: Configured composite model with loaded parameters
        """
        import pandas as pd
        import os
        
        logger.info(f"Loading composite model from: {lce_fname}")
        
        # Load KRA model
        kra_model = LocalClusterExpansion.from_json(lce_fname)
        
        # Load fitting results for KRA model
        logger.info("Loading fitting results: E_kra ...")
        fitting_results_data = (
            pd.read_json(fitting_results, orient="index")
            .sort_values(by=["time_stamp"], ascending=False)
            .iloc[0]
        )
        kra_params = {
            'keci': fitting_results_data.keci,
            'empty_cluster': fitting_results_data.empty_cluster
        }
        kra_model.set_parameters(kra_params)
        
        # Load site model if available
        site_model = None
        if lce_site_fname is not None and os.path.exists(lce_site_fname):
            logger.info(f"Loading site LCE from: {lce_site_fname}")
            site_model = LocalClusterExpansion.from_json(lce_site_fname)
            
            # Load fitting results for site model if available
            if fitting_results_site is not None and os.path.exists(fitting_results_site):
                logger.info("Loading fitting results: site energy ...")
                fitting_results_site_data = (
                    pd.read_json(fitting_results_site, orient="index")
                    .sort_values(by=["time_stamp"], ascending=False)
                    .iloc[0]
                )
                site_params = {
                    'keci': fitting_results_site_data.keci,
                    'empty_cluster': fitting_results_site_data.empty_cluster
                }
                site_model.set_parameters(site_params)
            else:
                logger.info("No site fitting results file found - using zero site energy contributions")
        else:
            logger.info("No site LCE file found - using None for site model")
        
        # Create composite model with pre-configured models
        return cls(site_model=site_model, kra_model=kra_model)

    @classmethod
    def from_config(cls, config: 'SimulationConfig') -> "CompositeLCEModel":
        """
        Create a CompositeLCEModel from a SimulationConfig object.
        
        Args:
            config: SimulationConfig containing model file paths
            
        Returns:
            CompositeLCEModel: Configured composite model with loaded parameters
        """
        return cls.from_json(
            lce_fname=config.cluster_expansion_file,
            fitting_results=config.fitting_results_file,
            lce_site_fname=config.cluster_expansion_site_file,
            fitting_results_site=config.fitting_results_site_file
        )
