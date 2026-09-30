"""
Composite Local Cluster Expansion Model

This module combines a KRA model with an optional site-energy-difference model
for transition-rate calculations.

Author: Zeyu Deng
"""

import importlib
import inspect
import logging
from typing import Any, Optional, TYPE_CHECKING
import numba as nb
import numpy as np

from kmcpy.models.base import BaseModel, MODEL_FILETYPE, require_model_type
from kmcpy.models.local_cluster_expansion import (
    LocalClusterExpansion,
    _calc_corr,
    _calc_corr_decorated,
)
from kmcpy.event import Event, event_direction
from kmcpy.event.hop import DEFAULT_HOP_STATE_CODES
from kmcpy.simulator.state import State
from kmcpy.units import BOLTZMANN_CONSTANT_MEV_PER_K

if TYPE_CHECKING:
    from kmcpy.simulator.config import Configuration, RuntimeConfig

logger = logging.getLogger(__name__)


def _accepts_keyword(callable_obj, keyword: str) -> bool:
    """Return whether a callable accepts a specific keyword argument."""
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False
    return keyword in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


class CompositeLCEModel(BaseModel):
    """
    A composite model that combines a KRA LCE with a site-energy-difference model.
    
    This class combines one ``LocalClusterExpansion`` for ``E_KRA`` with a
    site-energy-difference contribution. A ``LocalClusterExpansion`` always uses
    ``compute(simulation_state=..., event=...)``; its meaning comes from the
    role it is passed into. As ``kra_model`` it returns ``E_KRA``. As
    ``site_model`` it returns the site-energy-difference contribution for the
    canonical event orientation. ``SiteEnergyModel`` exposes the same
    ``compute(event=..., simulation_state=...)`` interface and returns the
    signed event energy change, ``E_after_hop - E_before_hop``, in meV.
    
    The composite model provides:
    
    - compute_probability(): compute transition rate from an event
    
    Example::
    
        # Create individual models with parameters
        site_model = LocalClusterExpansion(...)
        site_model.load_parameters_from_file("site_parameters.json")
        
        kra_model = LocalClusterExpansion(...)
        kra_model.load_parameters_from_file("kra_parameters.json")
        
        # Combine them
        composite = CompositeLCEModel(site_model, kra_model)
        
        # Use the composite model with State (preferred)
        rate = composite.compute_probability(
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
            site_model: model for site-energy-difference calculations. A
                ``LocalClusterExpansion`` is evaluated with ``compute`` and
                interpreted in the event's canonical orientation. Other models
                must expose ``compute`` and return
                ``E_after_hop - E_before_hop`` in meV.
            kra_model: model for E_KRA calculations
        """
        if kra_model is not None and not isinstance(kra_model, LocalClusterExpansion):
            raise TypeError(f"KRA model must be a LocalClusterExpansion instance, got {type(kra_model)}")
        if site_model is not None and not callable(getattr(site_model, "compute", None)):
            raise TypeError(
                "Site model must expose "
                f"compute(event=..., simulation_state=...), got {type(site_model)}"
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
        
    def fit(self, *args, **kwargs):
        """Composite models are assembled from separately fitted LCE models."""
        raise NotImplementedError(
            "Fit LocalClusterExpansion models separately, build SiteEnergyModel "
            "objects separately when needed, then pass them to "
            "CompositeLCEModel(site_model=..., kra_model=...)."
        )

    def _compute_site_energy_difference(
        self,
        event: Event,
        simulation_state: State,
        direction: int,
    ) -> float:
        """Return ``E_after_hop - E_before_hop`` in meV."""
        if self.site_model is None:
            return 0.0
        value = self.site_model.compute(
            simulation_state=simulation_state,
            event=event,
        )
        if isinstance(self.site_model, LocalClusterExpansion):
            # LCE uses one evaluator. A site LCE returns the fitted
            # site-energy-difference contribution for the canonical event
            # orientation; the current occupation determines the event sign.
            value = direction * value
        return float(value)

    def initialize_state(
        self,
        *,
        simulation_state: State,
        event_lib=None,
        structure=None,
        config=None,
        active_site_order=None,
    ) -> None:
        """Initialize optional stateful submodel caches."""
        for model in (self.kra_model, self.site_model):
            initialize_state = getattr(model, "initialize_state", None)
            if callable(initialize_state):
                kwargs = {
                    "simulation_state": simulation_state,
                    "event_lib": event_lib,
                    "structure": structure,
                    "config": config,
                }
                if active_site_order is not None and _accepts_keyword(
                    initialize_state,
                    "active_site_order",
                ):
                    kwargs["active_site_order"] = active_site_order
                initialize_state(**kwargs)

        self._batch_evaluator = _LCEBatchRateEvaluator.build(
            self,
            event_lib=event_lib,
            simulation_state=simulation_state,
        )

    def apply_event(self, *, event: Event, simulation_state: State) -> None:
        """Commit an accepted event to optional stateful submodels."""
        for model in (self.kra_model, self.site_model):
            apply_event = getattr(model, "apply_event", None)
            if callable(apply_event):
                apply_event(event=event, simulation_state=simulation_state)

        evaluator = getattr(self, "_batch_evaluator", None)
        if evaluator is not None:
            evaluator.apply_event(event, simulation_state)

    def compute_probabilities(
        self,
        *,
        events,
        event_indices,
        runtime_config: "RuntimeConfig",
        simulation_state: State,
    ) -> np.ndarray:
        """Compute rates in Hz for ``events[i]`` for each ``i`` in ``event_indices``.

        When both submodels are plain ``LocalClusterExpansion`` objects and
        ``initialize_state`` received the event library, all rates are
        evaluated in one compiled call. The result is identical to calling
        ``compute_probability`` for each event, which is the fallback.
        """
        evaluator = getattr(self, "_batch_evaluator", None)
        if evaluator is not None and evaluator.can_evaluate(
            self, events, simulation_state
        ):
            return evaluator.compute(
                self,
                event_indices=event_indices,
                runtime_config=runtime_config,
                simulation_state=simulation_state,
            )
        return super().compute_probabilities(
            events=events,
            event_indices=event_indices,
            runtime_config=runtime_config,
            simulation_state=simulation_state,
        )

    def compute_probability(
        self,
        event: Event,
        runtime_config: "RuntimeConfig",
        simulation_state: State,
    ) -> float:
        """
        Compute the transition rate in Hz for a given event using the composite LCE model.

        This method calculates the transition rate for a migration event by:
        
        - Computing the site-energy difference (delta_e_site, meV) using the site model.
        - Computing the barrier energy (e_kra, meV) using the barrier LocalClusterExpansion model and its stored parameters.
        - Determining the direction of the event from the occupation vector in the State.
        - Calculating the effective barrier as: e_barrier = e_kra + delta_e_site / 2
        - Using the Arrhenius equation to compute the rate:
          rate = hop_available * v * np.exp(-e_barrier / (k * temperature))

        Args:
            event (Event): The migration event, containing mobile ion indices and local environment info.
            runtime_config (RuntimeConfig): Contains attempt frequency (v) and temperature (T).
            simulation_state (State): Contains the current occupation vector.

        Returns:
            float: The computed transition rate in Hz.
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
        # Compute signed site-energy difference in meV.
        delta_e_site = self._compute_site_energy_difference(
            event=event,
            simulation_state=simulation_state,
            direction=direction,
        )

        # Calculate effective barrier
        e_barrier = e_kra + delta_e_site / 2
        
        # Get temperature and attempt frequency from runtime configuration
        temperature = runtime_config.temperature
        v = runtime_config.attempt_frequency
        
        # Compute rate using Arrhenius equation
        rate = v * np.exp(-e_barrier / (k * temperature))
        
        return rate

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
                "Composite LCE site model must contain object key 'model'"
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
            "Build LocalClusterExpansion models separately, build SiteEnergyModel "
            "objects separately when needed, then pass them to "
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
                "Site model payload must include '@module' and '@class'"
            )
        model_cls = getattr(importlib.import_module(module_path), class_name)
        if not callable(getattr(model_cls, "from_dict", None)):
            raise ValueError(
                f"Site model class '{module_path}.{class_name}' "
                "must provide from_dict()."
            )
        model = model_cls.from_dict(payload)
        if not callable(getattr(model, "compute", None)):
            raise TypeError(
                f"Site model '{module_path}.{class_name}' must expose "
                "compute(event=..., simulation_state=...)."
            )
        return model

    @classmethod
    def from_file(cls, model_file: str) -> "CompositeLCEModel":
        """Create a CompositeLCEModel from a serialized model file."""
        from monty.serialization import loadfn

        logger.info("Loading composite model file from: %s", model_file)
        return cls.from_dict(loadfn(model_file, cls=None))


def _uses_default_lce_evaluation(model) -> bool:
    """Return whether ``model`` evaluates exactly like ``LocalClusterExpansion.compute``."""
    model_type = type(model)
    return (
        isinstance(model, LocalClusterExpansion)
        and model_type.compute is LocalClusterExpansion.compute
        and model_type._calculate_correlation is LocalClusterExpansion._calculate_correlation
        and hasattr(model, "keci")
        and hasattr(model, "empty_cluster")
        and hasattr(model, "cluster_site_indices")
    )


def _lce_kernel_inputs(model) -> tuple:
    """Return the arrays the batch kernel needs to evaluate one LCE submodel."""
    correlation_count = model._validate_keci_once()
    correlation_basis_indices = getattr(model, "correlation_basis_indices", None)
    site_basis_values = getattr(model, "site_basis_values", None)
    decorated = correlation_basis_indices is not None and site_basis_values is not None
    orbit_offsets, cluster_offsets, sites, basis = model._flat_cluster_indices(
        correlation_basis_indices if decorated else None
    )
    if not decorated:
        site_basis_values = _EMPTY_SITE_BASIS_VALUES
    return (
        decorated,
        correlation_count,
        orbit_offsets,
        cluster_offsets,
        sites,
        basis,
        np.asarray(site_basis_values, dtype=np.float64),
        np.asarray(model.keci, dtype=np.float64),
        float(model.empty_cluster),
    )


_EMPTY_SITE_BASIS_VALUES = np.zeros((1, 1, 1), dtype=np.float64)
_NO_SITE_MODEL_INPUTS = (
    False,
    0,
    np.zeros(1, dtype=np.int64),
    np.zeros(1, dtype=np.int64),
    np.zeros(0, dtype=np.int64),
    np.zeros(0, dtype=np.int64),
    _EMPTY_SITE_BASIS_VALUES,
    np.zeros(0, dtype=np.float64),
    0.0,
)


class _LCEBatchRateEvaluator:
    """Batched rate evaluation for a composite of plain LCE submodels.

    It keeps an ``int64`` copy of the active-site occupations, which is updated
    at the two hop endpoints in ``apply_event`` and fully resynchronized if
    the ``State`` object or its step counter changes unexpectedly.
    """

    def __init__(self, kra_model, site_model, events, simulation_state):
        self.kra_model = kra_model
        self.site_model = site_model
        self.events = events
        self.event_count = len(events)

        from_sites = np.empty(self.event_count, dtype=np.int64)
        to_sites = np.empty(self.event_count, dtype=np.int64)
        hop_codes = np.empty((self.event_count, 4), dtype=np.int64)
        env_offsets = np.zeros(self.event_count + 1, dtype=np.int64)
        env_sites = []
        for event_index, event in enumerate(events):
            from_site, to_site = event.mobile_ion_indices
            from_sites[event_index] = int(from_site)
            to_sites[event_index] = int(to_site)
            hop_codes[event_index] = getattr(
                event, "hop_state_codes", DEFAULT_HOP_STATE_CODES
            )
            env_sites.extend(int(site) for site in event.local_env_indices)
            env_offsets[event_index + 1] = len(env_sites)
        self.from_sites = from_sites
        self.to_sites = to_sites
        self.hop_codes = hop_codes
        self.env_offsets = env_offsets
        self.env_sites = np.asarray(env_sites, dtype=np.int64)
        self._sync_occupations(simulation_state)

    @classmethod
    def build(cls, model, *, event_lib, simulation_state):
        """Return an evaluator, or ``None`` when the model/events are not eligible."""
        if event_lib is None or simulation_state is None:
            return None
        if not _uses_default_lce_evaluation(model.kra_model):
            return None
        if model.site_model is not None and not _uses_default_lce_evaluation(
            model.site_model
        ):
            return None
        events = getattr(event_lib, "events", event_lib)
        try:
            if any(len(event.mobile_ion_indices) != 2 for event in events):
                return None
            return cls(model.kra_model, model.site_model, events, simulation_state)
        except (TypeError, ValueError):
            return None

    def _sync_occupations(self, simulation_state) -> None:
        self.state = simulation_state
        self.occupations = np.asarray(simulation_state.occupations, dtype=np.int64).copy()
        self.state_step = simulation_state.step

    def can_evaluate(self, model, events, simulation_state) -> bool:
        return (
            events is self.events
            and len(events) == self.event_count
            and model.kra_model is self.kra_model
            and model.site_model is self.site_model
            and simulation_state is not None
        )

    def apply_event(self, event, simulation_state) -> None:
        if simulation_state is not self.state:
            self._sync_occupations(simulation_state)
            return
        occupations = simulation_state.occupations
        for site in event.mobile_ion_indices:
            self.occupations[site] = occupations[site]
        self.state_step = simulation_state.step

    def compute(self, model, *, event_indices, runtime_config, simulation_state) -> np.ndarray:
        if simulation_state is not self.state or simulation_state.step != self.state_step:
            self._sync_occupations(simulation_state)
        site_inputs = (
            _lce_kernel_inputs(self.site_model)
            if self.site_model is not None
            else _NO_SITE_MODEL_INPUTS
        )
        rates = np.empty(len(event_indices), dtype=np.float64)
        _compute_composite_lce_rates(
            rates,
            np.asarray(event_indices, dtype=np.int64),
            self.occupations,
            self.from_sites,
            self.to_sites,
            self.hop_codes,
            self.env_offsets,
            self.env_sites,
            *_lce_kernel_inputs(self.kra_model),
            self.site_model is not None,
            *site_inputs,
            float(runtime_config.attempt_frequency),
            float(BOLTZMANN_CONSTANT_MEV_PER_K * runtime_config.temperature),
        )
        return rates


@nb.njit
def _lce_value(
    local_occupation,
    decorated,
    correlation_count,
    orbit_offsets,
    cluster_offsets,
    sites,
    basis,
    site_basis_values,
    keci,
    empty_cluster,
):
    corr = np.empty(correlation_count)
    if decorated:
        _calc_corr_decorated(
            corr,
            local_occupation,
            orbit_offsets,
            cluster_offsets,
            sites,
            basis,
            site_basis_values,
        )
    else:
        _calc_corr(corr, local_occupation, orbit_offsets, cluster_offsets, sites)
    return np.dot(corr, keci) + empty_cluster


@nb.njit
def _compute_composite_lce_rates(
    rates,
    event_indices,
    occupations,
    from_sites,
    to_sites,
    hop_codes,
    env_offsets,
    env_sites,
    kra_decorated,
    kra_correlation_count,
    kra_orbit_offsets,
    kra_cluster_offsets,
    kra_sites,
    kra_basis,
    kra_site_basis_values,
    kra_keci,
    kra_empty_cluster,
    has_site_model,
    site_decorated,
    site_correlation_count,
    site_orbit_offsets,
    site_cluster_offsets,
    site_sites,
    site_basis,
    site_site_basis_values,
    site_keci,
    site_empty_cluster,
    attempt_frequency,
    k_times_temperature,
):
    """Evaluate ``CompositeLCEModel.compute_probability`` for many events.

    The arithmetic mirrors the scalar path operation by operation so that
    batched and per-event rates are identical.
    """
    for position in range(len(event_indices)):
        event_index = event_indices[position]
        from_occ = occupations[from_sites[event_index]]
        to_occ = occupations[to_sites[event_index]]
        if from_occ == hop_codes[event_index, 0] and to_occ == hop_codes[event_index, 1]:
            direction = 1
        elif from_occ == hop_codes[event_index, 2] and to_occ == hop_codes[event_index, 3]:
            direction = -1
        else:
            rates[position] = 0.0
            continue

        start = env_offsets[event_index]
        stop = env_offsets[event_index + 1]
        local_occupation = np.empty(stop - start, dtype=np.int64)
        for offset in range(stop - start):
            local_occupation[offset] = occupations[env_sites[start + offset]]

        e_kra = _lce_value(
            local_occupation,
            kra_decorated,
            kra_correlation_count,
            kra_orbit_offsets,
            kra_cluster_offsets,
            kra_sites,
            kra_basis,
            kra_site_basis_values,
            kra_keci,
            kra_empty_cluster,
        )
        delta_e_site = 0.0
        if has_site_model:
            delta_e_site = direction * _lce_value(
                local_occupation,
                site_decorated,
                site_correlation_count,
                site_orbit_offsets,
                site_cluster_offsets,
                site_sites,
                site_basis,
                site_site_basis_values,
                site_keci,
                site_empty_cluster,
            )
        e_barrier = e_kra + delta_e_site / 2
        rates[position] = attempt_frequency * np.exp(-e_barrier / k_times_temperature)

