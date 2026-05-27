"""Site-energy difference adapters for composite KMC models."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Optional

from monty.serialization import loadfn

from kmcpy.models.base import BaseModel, MODEL_FILETYPE, require_model_type

logger = logging.getLogger(__name__)


class ExternalSiteEnergyModel(BaseModel):
    """Adapter for external site-energy difference evaluators.

    The wrapped callable must return the event energy change
    ``E_after_hop - E_before_hop`` for the current ``simulation_state`` and
    ``event``. kMCpy consumes this value in meV, so the adapter converts from
    ``eV`` when requested.

    The callable is resolved from a string reference such as
    ``"package.module:function"`` or ``"package.module.function"`` and is called
    as::

        callable(event=event, simulation_state=simulation_state, **kwargs)

    This intentionally keeps smol/CLEASE optional: project-specific adapters can
    live outside kMCpy while the composite model sees one small interface.
    """

    MODEL_TYPE = "external_site_energy"
    PAYLOAD_KEY = "external_site_energy"
    _UNIT_FACTORS_TO_MEV = {
        "mev": 1.0,
        "ev": 1000.0,
    }

    def __init__(
        self,
        callable_ref: str,
        units: str = "meV",
        kwargs: Optional[dict[str, Any]] = None,
        name: str = "ExternalSiteEnergyModel",
    ) -> None:
        super().__init__(name=name)
        if not isinstance(callable_ref, str) or not callable_ref.strip():
            raise ValueError("'callable_ref' must be a non-empty string")
        self.callable_ref = callable_ref.strip()
        self.units = self._normalize_units(units)
        self.kwargs = dict(kwargs or {})
        self._callable = None

    @classmethod
    def _normalize_units(cls, units: str) -> str:
        token = str(units).strip()
        if token.lower() not in cls._UNIT_FACTORS_TO_MEV:
            raise ValueError("ExternalSiteEnergyModel units must be 'meV' or 'eV'")
        return "meV" if token.lower() == "mev" else "eV"

    @property
    def unit_factor_to_mev(self) -> float:
        """Conversion factor from configured units to meV."""
        return self._UNIT_FACTORS_TO_MEV[self.units.lower()]

    def _resolve_callable(self):
        if self._callable is None:
            self._callable = resolve_callable_reference(self.callable_ref)
        return self._callable

    def compute_delta(self, event, simulation_state) -> float:
        """Return ``E_after_hop - E_before_hop`` in meV."""
        raw_value = self._resolve_callable()(
            event=event,
            simulation_state=simulation_state,
            **self.kwargs,
        )
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError(
                "External site-energy callable must return a numeric value"
            )
        return float(raw_value) * self.unit_factor_to_mev

    def compute(self, simulation_state, event) -> float:
        """Alias for ``compute_delta`` for BaseModel compatibility."""
        return self.compute_delta(event=event, simulation_state=simulation_state)

    def compute_probability(self, *args, **kwargs):
        raise NotImplementedError(
            "ExternalSiteEnergyModel only computes site-energy differences. "
            "Use it as CompositeLCEModel(site_model=...)."
        )

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "ExternalSiteEnergyModel delegates to an external callable and "
            "does not support fit()."
        )

    def build(self, *args, **kwargs):
        raise NotImplementedError(
            "ExternalSiteEnergyModel delegates to an external callable and "
            "does not support build()."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "name": self.name,
            "callable_ref": self.callable_ref,
            "units": self.units,
            "kwargs": dict(self.kwargs),
        }

    def to_model_file_dict(self) -> dict[str, Any]:
        return {
            "filetype": MODEL_FILETYPE,
            "model_type": self.MODEL_TYPE,
            self.PAYLOAD_KEY: self.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExternalSiteEnergyModel":
        if not isinstance(data, dict):
            raise ValueError("ExternalSiteEnergyModel payload must be a JSON object")
        if data.get("model_type") == cls.MODEL_TYPE and cls.PAYLOAD_KEY in data:
            data = data[cls.PAYLOAD_KEY]
        if not isinstance(data, dict):
            raise ValueError("ExternalSiteEnergyModel payload must be a JSON object")
        return cls(
            callable_ref=data.get("callable_ref") or data.get("callable"),
            units=data.get("units", "meV"),
            kwargs=data.get("kwargs"),
            name=data.get("name", "ExternalSiteEnergyModel"),
        )

    @classmethod
    def from_model_file_dict(
        cls, model_data: dict[str, Any]
    ) -> "ExternalSiteEnergyModel":
        data = require_model_type(model_data, cls.MODEL_TYPE)
        return cls.from_dict(data[cls.PAYLOAD_KEY])

    @classmethod
    def from_file(cls, filename: str) -> "ExternalSiteEnergyModel":
        data = loadfn(filename, cls=None)
        if isinstance(data, dict) and data.get("filetype") == MODEL_FILETYPE:
            return cls.from_model_file_dict(data)
        return cls.from_dict(data)

    def to(self, filename: str, indent: int = 2) -> None:
        from monty.serialization import dumpfn

        logger.info("Saving external site-energy model file to: %s", filename)
        dumpfn(self.to_model_file_dict(), filename, indent=indent)

    def __str__(self) -> str:
        return (
            f"ExternalSiteEnergyModel(callable_ref={self.callable_ref!r}, "
            f"units={self.units!r})"
        )

    def __repr__(self) -> str:
        return (
            "ExternalSiteEnergyModel("
            f"callable_ref={self.callable_ref!r}, units={self.units!r}, "
            f"kwargs={self.kwargs!r})"
        )


class ZeroSiteEnergyModel(BaseModel):
    """Site-energy model that always returns zero energy difference."""

    MODEL_TYPE = "zero_site_energy"
    PAYLOAD_KEY = "zero_site_energy"

    def __init__(self, name: str = "ZeroSiteEnergyModel") -> None:
        super().__init__(name=name)

    def compute_delta(self, event, simulation_state) -> float:
        return 0.0

    def compute(self, simulation_state, event) -> float:
        return 0.0

    def compute_probability(self, *args, **kwargs):
        raise NotImplementedError(
            "ZeroSiteEnergyModel only computes site-energy differences. "
            "Use it as CompositeLCEModel(site_model=...)."
        )

    def fit(self, *args, **kwargs):
        raise NotImplementedError("ZeroSiteEnergyModel does not support fit().")

    def build(self, *args, **kwargs):
        raise NotImplementedError("ZeroSiteEnergyModel does not support build().")

    def as_dict(self) -> dict[str, Any]:
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "name": self.name,
        }

    def to_model_file_dict(self) -> dict[str, Any]:
        return {
            "filetype": MODEL_FILETYPE,
            "model_type": self.MODEL_TYPE,
            self.PAYLOAD_KEY: self.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZeroSiteEnergyModel":
        if not isinstance(data, dict):
            raise ValueError("ZeroSiteEnergyModel payload must be a JSON object")
        return cls(name=data.get("name", "ZeroSiteEnergyModel"))

    @classmethod
    def from_file(cls, filename: str) -> "ZeroSiteEnergyModel":
        data = loadfn(filename, cls=None)
        if isinstance(data, dict) and data.get("filetype") == MODEL_FILETYPE:
            data = require_model_type(data, cls.MODEL_TYPE)[cls.PAYLOAD_KEY]
        return cls.from_dict(data)

    def to(self, filename: str, indent: int = 2) -> None:
        from monty.serialization import dumpfn

        logger.info("Saving zero site-energy model file to: %s", filename)
        dumpfn(self.to_model_file_dict(), filename, indent=indent)

    def __str__(self) -> str:
        return "ZeroSiteEnergyModel()"

    def __repr__(self) -> str:
        return "ZeroSiteEnergyModel()"


def resolve_callable_reference(callable_ref: str):
    """Resolve ``module:function`` or ``module.function`` references."""
    if ":" in callable_ref:
        module_path, attr_path = callable_ref.split(":", 1)
    else:
        module_path, _, attr_path = callable_ref.rpartition(".")
    if not module_path or not attr_path:
        raise ValueError(
            f"Invalid callable reference '{callable_ref}'. Use "
            "'package.module:function' or 'package.module.function'."
        )
    module = importlib.import_module(module_path)
    obj: Any = module
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    if not callable(obj):
        raise TypeError(f"Resolved object '{callable_ref}' is not callable")
    return obj


def constant_site_energy_delta(event, simulation_state, value: float = 0.0) -> float:
    """Small helper used by examples/tests to return a constant delta."""
    return float(value)
