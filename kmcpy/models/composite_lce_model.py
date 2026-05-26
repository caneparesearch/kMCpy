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

from kmcpy.models.base import CompositeModel
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
from kmcpy.models.schema import MODEL_FILETYPE, require_model_type
from kmcpy.event import Event
from kmcpy.simulator.state import State

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
        self.kra_fit_metadata = kra_fit_metadata or {"time_stamp": None, "time": None}
        self.site_fit_metadata = site_fit_metadata or {"time_stamp": None, "time": None}
        
    def compute(self) -> None:
        """
        This method is not used in the CompositeLCEModel.
        It is provided to maintain compatibility with the BaseModel interface.
        """
        raise NotImplementedError("CompositeLCEModel does not support direct compute method. Use compute_probability instead.")

    def fit(
        self,
        kra_fit_kwargs: dict,
        site_fit_kwargs: Optional[dict] = None,
    ) -> dict:
        """
        Fit component models and apply fitted parameters to attached submodels.

        Args:
            kra_fit_kwargs: Keyword arguments for fitting the KRA LocalClusterExpansion.
            site_fit_kwargs: Keyword arguments for fitting the site LocalClusterExpansion.
                If None, site fitting is skipped.

        Returns:
            dict: Mapping of component name to fitter result tuple
                ``(model_parameters, y_pred, y_true)``.
        """
        kra_fit_model = (
            self.kra_model if self.kra_model is not None else LocalClusterExpansion()
        )
        kra_fit_result = kra_fit_model.fit(**kra_fit_kwargs)
        results = {"kra": kra_fit_result}
        if site_fit_kwargs is not None:
            site_fit_model = (
                self.site_model
                if self.site_model is not None
                else LocalClusterExpansion()
            )
            results["site"] = site_fit_model.fit(**site_fit_kwargs)

        if self.kra_model is not None:
            self.kra_model.set_parameters(results["kra"][0])
        if self.site_model is not None and "site" in results:
            self.site_model.set_parameters(results["site"][0])

        return results

    def compute_probability(
        self,
        event: Event,
        runtime_config: "RuntimeConfig",
        simulation_state: State,
    ) -> float:
        """
        Compute the transition probability for a given event using the composite LCE model.

        This method calculates the transition probability for a migration event by:
        
        - Computing the site energy (e_site) using the site LocalClusterExpansion model and its stored parameters.
        - Computing the barrier energy (e_kra) using the barrier LocalClusterExpansion model and its stored parameters.
        - Determining the direction of the event from the occupation vector in the State.
        - Calculating the effective barrier as: e_barrier = e_kra + direction * e_site / 2
        - Using the Arrhenius equation to compute the probability:
          probability = abs(direction) * v * np.exp(-e_barrier / (k * temperature))

        Args:
            event (Event): The migration event, containing mobile ion indices and local environment info.
            runtime_config (RuntimeConfig): Contains attempt frequency (v) and temperature (T).
            simulation_state (State): Contains the current occupation vector.

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

    @staticmethod
    def _extract_parameters_for_model_file(model: LocalClusterExpansion, label: str) -> dict:
        """Extract fitted parameters from a local model for model-file serialization."""
        if not hasattr(model, "keci") or not hasattr(model, "empty_cluster"):
            raise ValueError(
                f"Cannot serialize '{label}' model to file: missing fitted parameters "
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

    @staticmethod
    def _validate_model_file_component(name: str, component: dict[str, Any]) -> None:
        if not isinstance(component, dict):
            raise ValueError(f"Model component '{name}' must be an object")
        if "lce" not in component or not isinstance(component["lce"], dict):
            raise ValueError(f"Model component '{name}' must contain object key 'lce'")

        parameters = component.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError(
                f"Model component '{name}' must contain object key 'parameters'"
            )
        if "keci" not in parameters or "empty_cluster" not in parameters:
            raise ValueError(
                f"Model component '{name}.parameters' must contain keys "
                "'keci' and 'empty_cluster'"
            )

    @classmethod
    def validate_model_file_dict(cls, model_data: dict[str, Any]) -> None:
        """Validate a composite LCE model-file payload."""
        data = require_model_type(model_data, "composite_lce")
        if "kra" not in data:
            raise ValueError("Composite model file must contain required key 'kra'")

        cls._validate_model_file_component("kra", data["kra"])
        if "site" in data and data["site"] is not None:
            cls._validate_model_file_component("site", data["site"])

    def to_model_file_dict(self) -> dict:
        """Convert this composite model into a model-file payload."""
        if self.kra_model is None:
            raise ValueError("Cannot serialize composite model: kra_model is missing")

        model_data = {
            "filetype": MODEL_FILETYPE,
            "model_type": "composite_lce",
            "kra": {
                "lce": self.kra_model.as_dict(),
                "parameters": self._extract_parameters_for_model_file(self.kra_model, "kra"),
                "fit_metadata": self.kra_fit_metadata,
            },
        }

        if self.site_model is not None:
            model_data["site"] = {
                "lce": self.site_model.as_dict(),
                "parameters": self._extract_parameters_for_model_file(self.site_model, "site"),
                "fit_metadata": self.site_fit_metadata,
            }

        self.validate_model_file_dict(model_data)
        return model_data

    def to(self, filename: str, indent: int = 2) -> None:
        """Write this composite model as a serialized model file."""
        from monty.serialization import dumpfn

        logger.info("Saving composite model file to: %s", filename)
        dumpfn(self.to_model_file_dict(), filename, indent=indent)

    def to_json(self, fname: str) -> None:
        """Compatibility alias for JSON model writing."""
        self.to(fname)

    @staticmethod
    def _latest_fit_record(fit_file: str) -> dict[str, Any]:
        from monty.serialization import loadfn

        payload = loadfn(fit_file, cls=None)
        if not isinstance(payload, dict):
            raise ValueError(f"Fitting results file {fit_file} must contain a JSON object")
        rows = [row for row in payload.values() if isinstance(row, dict)]
        if not rows:
            raise ValueError(f"No fitting rows found in {fit_file}")
        rows_with_ts = [row for row in rows if row.get("time_stamp") is not None]
        return max(rows_with_ts, key=lambda row: row["time_stamp"]) if rows_with_ts else rows[-1]

    @classmethod
    def _component_from_legacy_files(
        cls,
        lce_file: str,
        fit_file: str,
    ) -> tuple[LocalClusterExpansion, dict[str, Any]]:
        fit = cls._latest_fit_record(fit_file)
        if "keci" not in fit or "empty_cluster" not in fit:
            raise ValueError(
                f"Fitting results file {fit_file} must contain keys keci and empty_cluster"
            )

        model = LocalClusterExpansion.from_file(lce_file)
        parameters = {"keci": fit["keci"], "empty_cluster": fit["empty_cluster"]}
        if fit.get("orbit_fingerprints") is not None:
            parameters["orbit_fingerprints"] = fit["orbit_fingerprints"]
        if fit.get("local_environment_hash") is not None:
            parameters["local_environment_hash"] = fit["local_environment_hash"]
        if fit.get("ordering_convention") is not None:
            parameters["ordering_convention"] = fit["ordering_convention"]
        model.set_parameters(parameters)
        metadata = {
            key: value
            for key, value in fit.items()
            if key not in {"keci", "empty_cluster"}
        }
        return model, metadata

    @classmethod
    def from_legacy_files(
        cls,
        kra_lce: str,
        kra_fit: str,
        site_lce: str | None = None,
        site_fit: str | None = None,
    ) -> "CompositeLCEModel":
        """Build a composite LCE model from legacy LCE and fitting JSON files."""
        kra_model, kra_fit_metadata = cls._component_from_legacy_files(kra_lce, kra_fit)
        site_model = None
        site_fit_metadata = None
        if site_lce is not None or site_fit is not None:
            if site_lce is None or site_fit is None:
                raise ValueError("Both site_lce and site_fit are required for a site model")
            site_model, site_fit_metadata = cls._component_from_legacy_files(
                site_lce, site_fit
            )

        return cls(
            site_model=site_model,
            kra_model=kra_model,
            kra_fit_metadata=kra_fit_metadata,
            site_fit_metadata=site_fit_metadata,
        )

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
            return self.site_model.local_lattice_structure.get_occ_from_structure(structure)
        elif use_model == 'kra_model' and self.kra_model:
            return self.kra_model.local_lattice_structure.get_occ_from_structure(structure)
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
        site_model = (
            LocalClusterExpansion.from_dict(d["site_model"])
            if d.get("site_model")
            else None
        )
        kra_model = (
            LocalClusterExpansion.from_dict(d["kra_model"])
            if d.get("kra_model")
            else None
        )
        return cls(site_model=site_model, kra_model=kra_model, name=d.get("name"))

    @classmethod
    def from_model_file_dict(cls, model_data: dict[str, Any]) -> "CompositeLCEModel":
        """Create a CompositeLCEModel from an in-memory model-file payload."""
        cls.validate_model_file_dict(model_data)

        kra_component = model_data["kra"]
        kra_model = LocalClusterExpansion.from_dict(kra_component["lce"])
        kra_model.set_parameters(kra_component["parameters"])

        site_model = None
        site_fit_metadata = None
        if model_data.get("site") is not None:
            site_component = model_data["site"]
            site_model = LocalClusterExpansion.from_dict(site_component["lce"])
            site_model.set_parameters(site_component["parameters"])
            site_fit_metadata = site_component.get("fit_metadata")

        return cls(
            site_model=site_model,
            kra_model=kra_model,
            kra_fit_metadata=kra_component.get("fit_metadata"),
            site_fit_metadata=site_fit_metadata,
        )

    @classmethod
    def from_file(cls, model_file: str) -> "CompositeLCEModel":
        """Create a CompositeLCEModel from a serialized model file."""
        from monty.serialization import loadfn

        logger.info("Loading composite model file from: %s", model_file)
        return cls.from_model_file_dict(loadfn(model_file, cls=None))

    @classmethod
    def from_json(cls, model_file: str) -> "CompositeLCEModel":
        """
        Compatibility alias for JSON model loading.
        """
        return cls.from_file(model_file)

    @classmethod
    def from_config(cls, config: 'Configuration') -> "CompositeLCEModel":
        """
        Create a CompositeLCEModel from a Configuration object.
        
        Args:
            config: Configuration containing `model_file` path
            
        Returns:
            CompositeLCEModel: Configured composite model with loaded parameters
        """
        return cls.from_file(model_file=config.model_file)
