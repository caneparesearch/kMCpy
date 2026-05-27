"""
Composite Local Cluster Expansion Model

This module implements a composite model that combines two LocalClusterExpansion
models to provide a unified interface for different energy contributions (e.g., site
energies and barrier energies). This follows an ASE-like design where the model is
composable and generic.

Author: Zeyu Deng
"""

import importlib
import logging
from typing import Any, Optional, TYPE_CHECKING
import numpy as np

from kmcpy.models.base import BaseModel, MODEL_FILETYPE, require_model_type
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
from kmcpy.event import Event
from kmcpy.simulator.hop import event_direction
from kmcpy.simulator.state import State
from kmcpy.units import BOLTZMANN_CONSTANT_MEV_PER_K

if TYPE_CHECKING:
    from kmcpy.simulator.config import Configuration, RuntimeConfig

logger = logging.getLogger(__name__)


class CompositeLCEModel(BaseModel):
    """
    A composite model that combines a KRA LCE with a site-energy difference model.
    
    This class combines one ``LocalClusterExpansion`` for ``E_KRA`` with a
    site-energy contribution. The site-energy contribution may be either the
    historical kMCpy ``LocalClusterExpansion`` convention or any model exposing
    ``compute_delta(event=..., simulation_state=...)``. External smol/CLEASE
    adapters should return the actual event energy change,
    ``E_after_hop - E_before_hop``, in meV.
    
    The composite model provides:
    
    - compute_probability(): compute transition probability from an event
    
    Example::
    
        # Create individual models with parameters
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
        site_model: Optional[Any] = None,
        kra_model: Optional[LocalClusterExpansion] = None,
        kra_fit_metadata: Optional[dict[str, Any]] = None,
        site_fit_metadata: Optional[dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize a composite LCE model.
        
        Args:
            site_model: model for site energy difference calculations. A
                ``LocalClusterExpansion`` uses the historical directional
                convention. Other models must expose ``compute_delta`` and
                return ``E_after_hop - E_before_hop`` in meV.
            kra_model: model for E_KRA calculations
        """
        if kra_model is not None and not isinstance(kra_model, LocalClusterExpansion):
            raise TypeError(f"KRA model must be a LocalClusterExpansion instance, got {type(kra_model)}")
        if site_model is not None and not isinstance(
            site_model, LocalClusterExpansion
        ) and not callable(getattr(site_model, "compute_delta", None)):
            raise TypeError(
                "Site model must be a LocalClusterExpansion or expose "
                f"compute_delta(event=..., simulation_state=...), got {type(site_model)}"
            )

        models = []
        if site_model:
            models.append(site_model)
        if kra_model:
            models.append(kra_model)

        super().__init__(*args, **kwargs)
        
        self.models = models
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
            "Fit LocalClusterExpansion models separately, build external "
            "site-energy adapters separately, then pass them to "
            "CompositeLCEModel(site_model=..., kra_model=...)."
        )

    def _compute_site_energy_delta(
        self,
        event: Event,
        simulation_state: State,
        direction: int,
    ) -> float:
        """Return ``E_after_hop - E_before_hop`` in meV."""
        if self.site_model is None:
            return 0.0
        if isinstance(self.site_model, LocalClusterExpansion):
            # Historical kMCpy site LCE convention: compute() returns the
            # canonical forward site-energy term, so direction sets the event
            # sign. External adapters should return the signed delta directly.
            return float(
                direction
                * self.site_model.compute(
                    simulation_state=simulation_state,
                    event=event,
                )
            )
        return float(
            self.site_model.compute_delta(
                event=event,
                simulation_state=simulation_state,
            )
        )

    def initialize_state(
        self,
        *,
        simulation_state: State,
        event_lib=None,
        structure=None,
        config=None,
    ) -> None:
        """Initialize optional stateful submodel caches."""
        for model in (self.kra_model, self.site_model):
            initialize_state = getattr(model, "initialize_state", None)
            if callable(initialize_state):
                initialize_state(
                    simulation_state=simulation_state,
                    event_lib=event_lib,
                    structure=structure,
                    config=config,
                )

    def apply_event(self, *, event: Event, simulation_state: State) -> None:
        """Commit an accepted event to optional stateful submodels."""
        for model in (self.kra_model, self.site_model):
            apply_event = getattr(model, "apply_event", None)
            if callable(apply_event):
                apply_event(event=event, simulation_state=simulation_state)

    def compute_probability(
        self,
        event: Event,
        runtime_config: "RuntimeConfig",
        simulation_state: State,
    ) -> float:
        """
        Compute the transition probability/rate in Hz for a given event using the composite LCE model.

        This method calculates the transition probability for a migration event by:
        
        - Computing the site-energy change (delta_e_site, meV) using the site model.
        - Computing the barrier energy (e_kra, meV) using the barrier LocalClusterExpansion model and its stored parameters.
        - Determining the direction of the event from the occupation vector in the State.
        - Calculating the effective barrier as: e_barrier = e_kra + delta_e_site / 2
        - Using the Arrhenius equation to compute the probability:
          probability = hop_available * v * np.exp(-e_barrier / (k * temperature))

        Args:
            event (Event): The migration event, containing mobile ion indices and local environment info.
            runtime_config (RuntimeConfig): Contains attempt frequency (v) and temperature (T).
            simulation_state (State): Contains the current occupation vector.

        Returns:
            float: The computed transition probability/rate in Hz.
        """

        # Get occupation from simulation_state
        occ = simulation_state.occupations

        # Determine the direction of the event
        direction = event_direction(occ, event)
        if direction == 0:
            return 0.0

        # Boltzmann constant in meV/K.
        k = BOLTZMANN_CONSTANT_MEV_PER_K
        # Compute barrier energy (ekra) using stored parameters
        e_kra = self.kra_model.compute(simulation_state=simulation_state, event=event)
        # Compute signed site-energy change in meV.
        delta_e_site = self._compute_site_energy_delta(
            event=event,
            simulation_state=simulation_state,
            direction=direction,
        )

        # Calculate effective barrier
        e_barrier = e_kra + delta_e_site / 2
        
        # Get temperature and attempt frequency from runtime configuration
        temperature = runtime_config.temperature
        v = runtime_config.attempt_frequency
        
        # Compute probability using Arrhenius equation
        probability = v * np.exp(-e_barrier / (k * temperature))
        
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
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "filetype": MODEL_FILETYPE,
            "model_type": "composite_lce",
            "kra": self._submodel_as_dict(
                self.kra_model,
                fit_metadata=self.kra_fit_metadata,
                label="kra",
            ),
        }
        if self.site_model is not None:
            data["site"] = self._site_model_as_dict(
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

    @classmethod
    def _site_model_as_dict(
        cls,
        model: Any,
        fit_metadata: dict[str, Any],
        label: str,
    ) -> dict[str, Any]:
        if isinstance(model, LocalClusterExpansion):
            return cls._submodel_as_dict(
                model,
                fit_metadata=fit_metadata,
                label=label,
            )
        if not callable(getattr(model, "as_dict", None)):
            raise ValueError(
                f"Cannot serialize '{label}' model: missing as_dict()."
            )
        return {
            "model_type": getattr(model, "MODEL_TYPE", None),
            "model": model.as_dict(),
            "fit_metadata": fit_metadata,
            "delta_convention": "after_minus_before",
            "units": "meV",
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

    @staticmethod
    def _validate_site_model_payload(data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise ValueError("Composite LCE submodel 'site' must be an object")
        if "lce" in data:
            CompositeLCEModel._validate_submodel_payload("site", data)
            return
        if "model" not in data or not isinstance(data["model"], dict):
            raise ValueError(
                "Composite LCE external site model must contain object key 'model'"
            )

    @classmethod
    def _validate_dict(cls, data: dict[str, Any]) -> None:
        """Validate a composite LCE serialized payload."""
        payload = require_model_type(data, "composite_lce")
        if "kra" not in payload:
            raise ValueError("Composite LCE model must contain required key 'kra'")

        cls._validate_submodel_payload("kra", payload["kra"])
        if "site" in payload and payload["site"] is not None:
            cls._validate_site_model_payload(payload["site"])

    def to(self, filename: str, indent: int = 2) -> None:
        """Write this composite model to a serialized model file."""
        from monty.serialization import dumpfn

        logger.info("Saving composite model file to: %s", filename)
        dumpfn(self.as_dict(), filename, indent=indent)

    def build(self, *args, **kwargs):
        """Composite models are assembled from separately built LCE models."""
        raise NotImplementedError(
            "Build LocalClusterExpansion models separately, build external "
            "site-energy adapters separately, then pass them to "
            "CompositeLCEModel(site_model=..., kra_model=...)."
        )

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
            if "lce" in site_data:
                site_model = LocalClusterExpansion.from_dict(site_data["lce"])
                site_model.set_parameters(site_data["parameters"])
            else:
                site_model = cls._site_model_from_dict(site_data)
            site_fit_metadata = site_data.get("fit_metadata")

        return cls(
            site_model=site_model,
            kra_model=kra_model,
            kra_fit_metadata=kra_data.get("fit_metadata"),
            site_fit_metadata=site_fit_metadata,
        )

    @staticmethod
    def _site_model_from_dict(site_data: dict[str, Any]):
        payload = site_data["model"]
        module_path = payload.get("@module")
        class_name = payload.get("@class")
        if not module_path or not class_name:
            raise ValueError(
                "External site model payload must include '@module' and '@class'"
            )
        model_cls = getattr(importlib.import_module(module_path), class_name)
        if not callable(getattr(model_cls, "from_dict", None)):
            raise ValueError(
                f"External site model class '{module_path}.{class_name}' "
                "must provide from_dict()."
            )
        model = model_cls.from_dict(payload)
        if not callable(getattr(model, "compute_delta", None)):
            raise TypeError(
                f"External site model '{module_path}.{class_name}' must expose "
                "compute_delta(event=..., simulation_state=...)."
            )
        return model

    @classmethod
    def from_file(cls, model_file: str) -> "CompositeLCEModel":
        """Create a CompositeLCEModel from a serialized model file."""
        from monty.serialization import loadfn

        logger.info("Loading composite model file from: %s", model_file)
        return cls.from_dict(loadfn(model_file, cls=None))
