"""
Composite Local Cluster Expansion Model

This module implements a composite model that combines two LocalClusterExpansion
models to provide a unified interface for different energy contributions (e.g., site
energies and barrier energies). This follows an ASE-like design where the model is
composable and generic.

Author: Zeyu Deng
"""

import logging
from typing import Any, Optional, TYPE_CHECKING
import numpy as np

from kmcpy.models.base import CompositeModel, MODEL_FILETYPE, require_model_type
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
from kmcpy.event import Event
from kmcpy.simulator.state import State
from kmcpy.units import BOLTZMANN_CONSTANT_MEV_PER_K

if TYPE_CHECKING:
    from kmcpy.simulator.config import Configuration, RuntimeConfig

logger = logging.getLogger(__name__)


class CompositeLCEModel(CompositeModel):
    """
    A composite model that combines exactly two LocalClusterExpansion models.
    
    This class combines two LCE models: one for site energy difference and one for E_KRA. 
    The model is designed to be stateless and generic, following ASE-like 
    patterns where the model is separated from the structure and calculation context.
    
    The composite model provides:
    
    - compute_probability(): compute transition probability from an event
    
    Example::
    
        # Create individual LCE models with parameters
        site_model = LocalClusterExpansion(...)
        site_model.load_parameters_from_file("site_parameters.json")
        
        kra_model = LocalClusterExpansion(...)
        kra_model.load_parameters_from_file("kra_parameters.json")
        
        # Combine them
        composite = CompositeLCEModel(site_model, kra_model)
        
        # Use the composite model with State (preferred)
        probability = composite.compute_probability(
            event=event,
            runtime_config=runtime_config,
            simulation_state=simulation_state
        )
    """
    
    def __init__(
        self,
        site_model: Optional[LocalClusterExpansion] = None,
        kra_model: Optional[LocalClusterExpansion] = None,
        kra_fit_metadata: Optional[dict[str, Any]] = None,
        site_fit_metadata: Optional[dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize a composite LCE model with two submodels.
        
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
        self.kra_fit_metadata = kra_fit_metadata or {"time_stamp": None, "time": None}
        self.site_fit_metadata = site_fit_metadata or {"time_stamp": None, "time": None}
        
    def compute(self) -> None:
        """
        This method is not used in the CompositeLCEModel.
        It is provided to maintain compatibility with the BaseModel interface.
        """
        raise NotImplementedError("CompositeLCEModel does not support direct compute method. Use compute_probability instead.")

    def fit(self, *args, **kwargs):
        """Composite models are assembled from separately fitted LCE models."""
        raise NotImplementedError(
            "Fit LocalClusterExpansion models separately, apply their "
            "parameters with set_parameters(...), then pass them to "
            "CompositeLCEModel(site_model=..., kra_model=...)."
        )

    def compute_probability(
        self,
        event: Event,
        runtime_config: "RuntimeConfig",
        simulation_state: State,
    ) -> float:
        """
        Compute the transition probability/rate in Hz for a given event using the composite LCE model.

        This method calculates the transition probability for a migration event by:
        
        - Computing the site energy (e_site, meV) using the site LocalClusterExpansion model and its stored parameters.
        - Computing the barrier energy (e_kra, meV) using the barrier LocalClusterExpansion model and its stored parameters.
        - Determining the direction of the event from the occupation vector in the State.
        - Calculating the effective barrier as: e_barrier = e_kra + direction * e_site / 2
        - Using the Arrhenius equation to compute the probability:
          probability = abs(direction) * v * np.exp(-e_barrier / (k * temperature))

        Args:
            event (Event): The migration event, containing mobile ion indices and local environment info.
            runtime_config (RuntimeConfig): Contains attempt frequency (v) and temperature (T).
            simulation_state (State): Contains the current occupation vector.

        Returns:
            float: The computed transition probability/rate in Hz.
        """

        # Boltzmann constant in meV/K.
        k = BOLTZMANN_CONSTANT_MEV_PER_K
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
        
        # Get temperature and attempt frequency from runtime configuration
        temperature = runtime_config.temperature
        v = runtime_config.attempt_frequency
        
        # Compute probability using Arrhenius equation
        probability = abs(direction) * v * np.exp(-e_barrier / (k * temperature))
        
        return probability

    def __str__(self):
        return f"CompositeLCEModel(site_model={self.site_model}, kra_model={self.kra_model})"

    def __repr__(self):
        return f"CompositeLCEModel(site_model={self.site_model}, kra_model={self.kra_model})"

    def as_dict(self):
        """Serialize this composite model to the standard model-file payload."""
        if self.kra_model is None:
            raise ValueError("Cannot serialize composite model: kra_model is missing")

        data = {
            "filetype": MODEL_FILETYPE,
            "model_type": "composite_lce",
            "kra": self._submodel_as_dict(
                self.kra_model,
                fit_metadata=self.kra_fit_metadata,
                label="kra",
            ),
        }
        if self.site_model is not None:
            data["site"] = self._submodel_as_dict(
                self.site_model,
                fit_metadata=self.site_fit_metadata,
                label="site",
            )

        self._validate_dict(data)
        return data

    @staticmethod
    def _parameter_payload(model: LocalClusterExpansion, label: str) -> dict:
        """Extract fitted parameters from one LCE submodel."""
        if not hasattr(model, "keci") or not hasattr(model, "empty_cluster"):
            raise ValueError(
                f"Cannot serialize '{label}' model: missing fitted parameters "
                "(expected attributes 'keci' and 'empty_cluster')."
            )
        parameters = {
            "keci": model.keci,
            "empty_cluster": model.empty_cluster,
            "orbit_fingerprints": model.get_orbit_fingerprints(),
        }
        if getattr(model, "local_environment_hash", None) is not None:
            parameters["local_environment_hash"] = model.local_environment_hash
        return parameters

    @classmethod
    def _submodel_as_dict(
        cls,
        model: LocalClusterExpansion,
        fit_metadata: dict[str, Any],
        label: str,
    ) -> dict[str, Any]:
        return {
            "lce": model.as_dict(),
            "parameters": cls._parameter_payload(model, label),
            "fit_metadata": fit_metadata,
        }

    @staticmethod
    def _validate_submodel_payload(name: str, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise ValueError(f"Composite LCE submodel '{name}' must be an object")
        if "lce" not in data or not isinstance(data["lce"], dict):
            raise ValueError(
                f"Composite LCE submodel '{name}' must contain object key 'lce'"
            )

        parameters = data.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError(
                f"Composite LCE submodel '{name}' must contain object key "
                "'parameters'"
            )
        if "keci" not in parameters or "empty_cluster" not in parameters:
            raise ValueError(
                f"Composite LCE submodel '{name}.parameters' must contain "
                "keys 'keci' and 'empty_cluster'"
            )

    @classmethod
    def _validate_dict(cls, data: dict[str, Any]) -> None:
        """Validate a composite LCE serialized payload."""
        payload = require_model_type(data, "composite_lce")
        if "kra" not in payload:
            raise ValueError("Composite LCE model must contain required key 'kra'")

        cls._validate_submodel_payload("kra", payload["kra"])
        if "site" in payload and payload["site"] is not None:
            cls._validate_submodel_payload("site", payload["site"])

    def to(self, filename: str, indent: int = 2) -> None:
        """Write this composite model to a serialized model file."""
        from monty.serialization import dumpfn

        logger.info("Saving composite model file to: %s", filename)
        dumpfn(self.as_dict(), filename, indent=indent)

    def build(self, *args, **kwargs):
        """Composite models are assembled from separately built LCE models."""
        raise NotImplementedError(
            "Build LocalClusterExpansion models separately, then pass them to "
            "CompositeLCEModel(site_model=..., kra_model=...)."
        )

    def get_occ_from_structure(self, structure, use_model='site_model'):
        """
        Get the occupation vector from a structure.
        
        Args:
            structure (Structure): The structure from which to compute the occupation vector.
            use_model (str): Specify which model to use for occupation calculation ('site_model' or 'kra_model').
        
        Returns:
            np.ndarray: The occupation vector for the given structure.
        """
        if use_model == 'site_model' and self.site_model:
            return self.site_model.local_lattice_structure.get_occ_from_structure(structure)
        elif use_model == 'kra_model' and self.kra_model:
            return self.kra_model.local_lattice_structure.get_occ_from_structure(structure)
        else:
            raise ValueError(f"Invalid model specified: {use_model}. Available models are 'site_model' and 'kra_model'.")
    
    def get_corr_from_structure(self, structure, use_model='site_model', tol=1e-2, angle_tol=5):
        """
        Get the correlation vector from a structure.
        
        Args:
            structure (Structure): The structure from which to compute the correlation vector.
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
    def from_dict(cls, data: dict[str, Any]) -> "CompositeLCEModel":
        """Create a CompositeLCEModel from a serialized payload."""
        cls._validate_dict(data)

        kra_data = data["kra"]
        kra_model = LocalClusterExpansion.from_dict(kra_data["lce"])
        kra_model.set_parameters(kra_data["parameters"])

        site_model = None
        site_fit_metadata = None
        if data.get("site") is not None:
            site_data = data["site"]
            site_model = LocalClusterExpansion.from_dict(site_data["lce"])
            site_model.set_parameters(site_data["parameters"])
            site_fit_metadata = site_data.get("fit_metadata")

        return cls(
            site_model=site_model,
            kra_model=kra_model,
            kra_fit_metadata=kra_data.get("fit_metadata"),
            site_fit_metadata=site_fit_metadata,
        )

    @classmethod
    def from_file(cls, model_file: str) -> "CompositeLCEModel":
        """Create a CompositeLCEModel from a serialized model file."""
        from monty.serialization import loadfn

        logger.info("Loading composite model file from: %s", model_file)
        return cls.from_dict(loadfn(model_file, cls=None))
