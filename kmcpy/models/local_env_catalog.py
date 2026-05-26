#!/usr/bin/env python
"""Local-environment catalog for direct sparse lookup."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from kmcpy.event import Event
from kmcpy.models.base import BaseModel
from kmcpy.models.schema import MODEL_FILETYPE, require_model_type
from kmcpy.simulator.state import State

if TYPE_CHECKING:
    from kmcpy.simulator.config import Configuration, RuntimeConfig

logger = logging.getLogger(__name__)


def _normalize_index_sequence(values: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"'{field_name}' must be a list or tuple of integers")
    if not values:
        raise ValueError(f"'{field_name}' must be non-empty")

    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"'{field_name}' must contain integers only")
        normalized.append(int(value))
    return tuple(normalized)


def _canonical_site_indices(
    mobile_ion_indices: tuple[int, ...], local_env_indices: tuple[int, ...]
) -> tuple[int, ...]:
    canonical: list[int] = []
    seen: set[int] = set()
    for site_index in mobile_ion_indices + local_env_indices:
        if site_index in seen:
            continue
        seen.add(site_index)
        canonical.append(site_index)
    return tuple(canonical)


def _normalize_occupations(values: Any) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError("'occupations' must be a list or tuple of integers")
    if not values:
        raise ValueError("'occupations' must be non-empty")

    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("'occupations' must contain integers only")
        ivalue = int(value)
        if ivalue not in (-1, 1):
            raise ValueError("'occupations' must use Chebyshev values -1 or 1")
        normalized.append(ivalue)
    return tuple(normalized)


def _normalize_properties(properties: Any) -> dict[str, float]:
    if not isinstance(properties, dict) or not properties:
        raise ValueError("'properties' must be a non-empty object")

    normalized: dict[str, float] = {}
    for key, value in properties.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Property names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Property '{key}' must be a numeric value")
        normalized[key] = float(value)
    return normalized


@dataclass(frozen=True)
class LocalEnvCatalogEntry:
    """One catalog row keyed by event and canonical local occupations."""

    mobile_ion_indices: tuple[int, ...]
    local_env_indices: tuple[int, ...]
    occupations: tuple[int, ...]
    properties: dict[str, float]

    @property
    def canonical_site_indices(self) -> tuple[int, ...]:
        return _canonical_site_indices(
            self.mobile_ion_indices, self.local_env_indices
        )

    @property
    def canonical_key(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return (
            self.mobile_ion_indices,
            self.canonical_site_indices,
            self.occupations,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "mobile_ion_indices": list(self.mobile_ion_indices),
            "local_env_indices": list(self.local_env_indices),
            "occupations": list(self.occupations),
            "properties": dict(self.properties),
        }

    @classmethod
    def from_dict(cls, entry: dict[str, Any]) -> "LocalEnvCatalogEntry":
        if not isinstance(entry, dict):
            raise TypeError("Each local-environment catalog entry must be a JSON object")

        mobile_ion_indices = _normalize_index_sequence(
            entry.get("mobile_ion_indices"), "mobile_ion_indices"
        )
        local_env_indices = _normalize_index_sequence(
            entry.get("local_env_indices"), "local_env_indices"
        )
        occupations = _normalize_occupations(entry.get("occupations"))
        properties = _normalize_properties(entry.get("properties"))

        canonical_sites = _canonical_site_indices(mobile_ion_indices, local_env_indices)
        if len(occupations) != len(canonical_sites):
            raise ValueError(
                "Entry occupation length must match canonical site count "
                f"({len(canonical_sites)}), got {len(occupations)}"
            )

        return cls(
            mobile_ion_indices=mobile_ion_indices,
            local_env_indices=local_env_indices,
            occupations=occupations,
            properties=properties,
        )


class LocalEnvCatalog(BaseModel):
    """Direct lookup model for sparse, exact event+occupation data."""

    SUPPORTED_LOOKUP_KEY = "event_local_occupation"
    SUPPORTED_PROBABILITY_MODE = "barrier_arrhenius"
    BOLTZMANN_CONSTANT_MEV_PER_K = 8.617333262145e-2

    def __init__(
        self,
        entries: Optional[list[dict[str, Any] | LocalEnvCatalogEntry]] = None,
        name: str = "LocalEnvCatalog",
        lookup_key: str = SUPPORTED_LOOKUP_KEY,
        default_property: str = "barrier",
        probability_mode: str = SUPPORTED_PROBABILITY_MODE,
        probability_property: str = "barrier",
    ) -> None:
        super().__init__(name=name)
        self.name = name
        self.lookup_key = lookup_key
        self.default_property = default_property
        self.probability_mode = probability_mode
        self.probability_property = probability_property
        self.entries: list[LocalEnvCatalogEntry] = []
        self._table: dict[
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], dict[str, float]
        ] = {}

        if entries is not None:
            self.build(entries=entries)

    def fit(self, *args, **kwargs):
        """Local-environment catalogs are defined by explicit data, not by fitting."""
        raise NotImplementedError(
            "LocalEnvCatalog does not support fit(). Provide explicit table entries instead."
        )

    def _validate_modes(self) -> None:
        if self.lookup_key != self.SUPPORTED_LOOKUP_KEY:
            raise ValueError(
                f"Unsupported lookup_key '{self.lookup_key}'. "
                f"Expected '{self.SUPPORTED_LOOKUP_KEY}'."
            )
        if self.probability_mode != self.SUPPORTED_PROBABILITY_MODE:
            raise ValueError(
                f"Unsupported probability_mode '{self.probability_mode}'. "
                f"Expected '{self.SUPPORTED_PROBABILITY_MODE}'."
            )

    @classmethod
    def canonical_key_from_entry(
        cls, entry: dict[str, Any] | LocalEnvCatalogEntry
    ) -> tuple[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        dict[str, float],
    ]:
        normalized = (
            LocalEnvCatalogEntry.from_dict(entry)
            if isinstance(entry, dict)
            else entry
        )
        if not isinstance(normalized, LocalEnvCatalogEntry):
            raise TypeError("entry must be a dictionary or LocalEnvCatalogEntry")

        return (
            normalized.canonical_key,
            normalized.mobile_ion_indices,
            normalized.local_env_indices,
            normalized.occupations,
            dict(normalized.properties),
        )

    @classmethod
    def _canonical_key_from_event_state(
        cls, event: Event, occupations: list[int]
    ) -> tuple[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        tuple[int, ...],
        tuple[int, ...],
    ]:
        mobile_ion_indices = _normalize_index_sequence(
            event.mobile_ion_indices, "event.mobile_ion_indices"
        )
        local_env_indices = _normalize_index_sequence(
            event.local_env_indices, "event.local_env_indices"
        )
        canonical_site_indices = _canonical_site_indices(
            mobile_ion_indices, local_env_indices
        )

        try:
            occupation_pattern = tuple(
                int(occupations[site_index]) for site_index in canonical_site_indices
            )
        except IndexError as exc:
            raise IndexError(
                "Event site index is out of range for provided simulation occupations"
            ) from exc

        return (
            (mobile_ion_indices, canonical_site_indices, occupation_pattern),
            canonical_site_indices,
            occupation_pattern,
        )

    def add_entry(self, entry: dict[str, Any] | LocalEnvCatalogEntry) -> None:
        """Add one validated entry to the model."""
        self._validate_modes()

        normalized = (
            LocalEnvCatalogEntry.from_dict(entry)
            if isinstance(entry, dict)
            else entry
        )
        if not isinstance(normalized, LocalEnvCatalogEntry):
            raise TypeError("entry must be a dictionary or LocalEnvCatalogEntry")

        if self.default_property not in normalized.properties:
            raise ValueError(
                f"default_property '{self.default_property}' is not present in local-environment catalog entry"
            )
        if self.probability_property not in normalized.properties:
            raise ValueError(
                f"probability_property '{self.probability_property}' is not present in local-environment catalog entry"
            )

        canonical_key = normalized.canonical_key
        if canonical_key in self._table:
            raise ValueError(
                "Duplicate local-environment catalog canonical key detected: "
                f"mobile_ion_indices={normalized.mobile_ion_indices}, "
                f"canonical_sites={normalized.canonical_site_indices}, "
                f"occupations={normalized.occupations}"
            )

        self.entries.append(normalized)
        self._table[canonical_key] = dict(normalized.properties)

    def build(
        self,
        entries: list[dict[str, Any] | LocalEnvCatalogEntry],
        lookup_key: Optional[str] = None,
        default_property: Optional[str] = None,
        probability_mode: Optional[str] = None,
        probability_property: Optional[str] = None,
    ) -> None:
        """Mutating builder equivalent to ``from_entries(...)`` on an existing instance."""
        if lookup_key is not None:
            self.lookup_key = lookup_key
        if default_property is not None:
            self.default_property = default_property
        if probability_mode is not None:
            self.probability_mode = probability_mode
        if probability_property is not None:
            self.probability_property = probability_property

        self._validate_modes()

        if not isinstance(entries, list) or not entries:
            raise ValueError(
                "'entries' must be a non-empty list of local-environment catalog entries"
            )

        self.entries = []
        self._table = {}
        for entry in entries:
            self.add_entry(entry)

    @classmethod
    def from_entries(
        cls,
        entries: list[dict[str, Any] | LocalEnvCatalogEntry],
        name: str = "LocalEnvCatalog",
        lookup_key: str = SUPPORTED_LOOKUP_KEY,
        default_property: str = "barrier",
        probability_mode: str = SUPPORTED_PROBABILITY_MODE,
        probability_property: str = "barrier",
    ) -> "LocalEnvCatalog":
        """Construct a new local-environment catalog directly from entries."""
        return cls(
            entries=entries,
            name=name,
            lookup_key=lookup_key,
            default_property=default_property,
            probability_mode=probability_mode,
            probability_property=probability_property,
        )

    def _lookup_properties(
        self, simulation_state: State, event: Event
    ) -> dict[str, float]:
        if simulation_state is None:
            raise ValueError("simulation_state is required")
        if event is None:
            raise ValueError("event is required")

        (
            canonical_key,
            canonical_sites,
            occupation_pattern,
        ) = self._canonical_key_from_event_state(
            event=event, occupations=simulation_state.occupations
        )
        properties = self._table.get(canonical_key)
        if properties is None:
            raise KeyError(
                "No local-environment catalog entry found for event lookup: "
                f"mobile_ion_indices={tuple(event.mobile_ion_indices)}, "
                f"local_env_indices={tuple(event.local_env_indices)}, "
                f"canonical_sites={canonical_sites}, "
                f"occupations={occupation_pattern}"
            )
        return properties

    def compute(
        self,
        simulation_state: State,
        event: Event,
        property_name: Optional[str] = None,
    ) -> float:
        """Compute a catalog property value by exact event+occupation lookup."""
        properties = self._lookup_properties(
            simulation_state=simulation_state, event=event
        )
        selected_property = property_name or self.default_property
        if selected_property not in properties:
            raise KeyError(
                f"Property '{selected_property}' not found in matched local-environment catalog entry"
            )
        return float(properties[selected_property])

    def compute_probability(
        self,
        event: Event,
        runtime_config: "RuntimeConfig",
        simulation_state: State,
    ) -> float:
        """Compute event probability from a catalog barrier using Arrhenius equation."""
        if self.probability_mode != self.SUPPORTED_PROBABILITY_MODE:
            raise ValueError(
                f"Unsupported probability mode '{self.probability_mode}' for LocalEnvCatalog"
            )

        barrier = self.compute(
            simulation_state=simulation_state,
            event=event,
            property_name=self.probability_property,
        )
        from_site, to_site = event.mobile_ion_indices
        occupations = simulation_state.occupations
        direction = (occupations[to_site] - occupations[from_site]) / 2
        temperature = runtime_config.temperature
        attempt_frequency = runtime_config.attempt_frequency

        probability = abs(direction) * attempt_frequency * np.exp(
            -barrier / (self.BOLTZMANN_CONSTANT_MEV_PER_K * temperature)
        )
        return float(probability)

    def __str__(self) -> str:
        return (
            f"LocalEnvCatalog(name={self.name}, entries={len(self.entries)}, "
            f"default_property={self.default_property}, "
            f"probability_property={self.probability_property})"
        )

    def __repr__(self) -> str:
        return (
            "LocalEnvCatalog("
            f"name={self.name!r}, entries={len(self.entries)}, "
            f"lookup_key={self.lookup_key!r}, "
            f"default_property={self.default_property!r}, "
            f"probability_mode={self.probability_mode!r}, "
            f"probability_property={self.probability_property!r})"
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialize model payload."""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "name": self.name,
            "lookup_key": self.lookup_key,
            "default_property": self.default_property,
            "probability_mode": self.probability_mode,
            "probability_property": self.probability_property,
            "entries": [entry.as_dict() for entry in self.entries],
        }

    def to_model_file_dict(self) -> dict[str, Any]:
        """Serialize this local-environment catalog into the model-file format."""
        model_data = {
            "filetype": MODEL_FILETYPE,
            "model_type": "local_env_catalog",
            "local_env_catalog": self.as_dict(),
        }
        self.validate_model_file_dict(model_data)
        return model_data

    def to(self, filename: str, indent: int = 2) -> None:
        """Write this local-environment catalog as a serialized model file."""
        from monty.serialization import dumpfn

        dumpfn(self.to_model_file_dict(), filename, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LocalEnvCatalog":
        """Deserialize from in-memory payload."""
        if not isinstance(data, dict):
            raise ValueError("LocalEnvCatalog payload must be a JSON object")

        return cls(
            entries=data.get("entries"),
            name=data.get("name", "LocalEnvCatalog"),
            lookup_key=data.get("lookup_key", cls.SUPPORTED_LOOKUP_KEY),
            default_property=data.get("default_property", "barrier"),
            probability_mode=data.get(
                "probability_mode", cls.SUPPORTED_PROBABILITY_MODE
            ),
            probability_property=data.get("probability_property", "barrier"),
        )

    @classmethod
    def _from_entries_payload(
        cls,
        payload: dict[str, Any] | list[dict[str, Any]],
        name: str | None = None,
        lookup_key: str | None = None,
        default_property: str | None = None,
        probability_mode: str | None = None,
        probability_property: str | None = None,
    ) -> "LocalEnvCatalog":
        if isinstance(payload, list):
            entries = payload
            metadata: dict[str, Any] = {}
        elif isinstance(payload, dict):
            if "entries" not in payload:
                raise ValueError(
                    "Local-environment catalog entries JSON object must include key 'entries'"
                )
            entries = payload["entries"]
            metadata = payload
        else:
            raise ValueError(
                "Local-environment catalog entries file must contain a JSON list or object"
            )

        return cls.from_entries(
            entries=entries,
            name=name or metadata.get("name", "LocalEnvCatalog"),
            lookup_key=lookup_key
            or metadata.get("lookup_key", cls.SUPPORTED_LOOKUP_KEY),
            default_property=default_property
            or metadata.get("default_property", "barrier"),
            probability_mode=probability_mode
            or metadata.get("probability_mode", cls.SUPPORTED_PROBABILITY_MODE),
            probability_property=probability_property
            or metadata.get("probability_property", "barrier"),
        )

    @classmethod
    def validate_model_file_dict(cls, model_data: dict[str, Any]) -> None:
        """Validate a local-environment catalog-file payload."""
        data = require_model_type(model_data, "local_env_catalog")
        local_env_catalog_payload = data.get("local_env_catalog")
        if not isinstance(local_env_catalog_payload, dict):
            raise ValueError(
                "Local-environment catalog file is missing object key "
                "'local_env_catalog'"
            )
        cls.from_dict(local_env_catalog_payload)

    @classmethod
    def from_model_file_dict(cls, model_data: dict[str, Any]) -> "LocalEnvCatalog":
        """Create a LocalEnvCatalog from an in-memory model-file payload."""
        cls.validate_model_file_dict(model_data)
        return cls.from_dict(model_data["local_env_catalog"])

    @classmethod
    def from_file(
        cls,
        filename: str,
        name: str | None = None,
        lookup_key: str | None = None,
        default_property: str | None = None,
        probability_mode: str | None = None,
        probability_property: str | None = None,
    ) -> "LocalEnvCatalog":
        """Load from a model file, direct model payload, or raw entries file."""
        from monty.serialization import loadfn

        payload = loadfn(filename, cls=None)

        if isinstance(payload, dict) and "filetype" in payload:
            model = cls.from_model_file_dict(payload)
            if any(
                value is not None
                for value in (
                    name,
                    lookup_key,
                    default_property,
                    probability_mode,
                    probability_property,
                )
            ):
                return cls._from_entries_payload(
                    model.as_dict(),
                    name=name,
                    lookup_key=lookup_key,
                    default_property=default_property,
                    probability_mode=probability_mode,
                    probability_property=probability_property,
                )
            return model

        return cls._from_entries_payload(
            payload,
            name=name,
            lookup_key=lookup_key,
            default_property=default_property,
            probability_mode=probability_mode,
            probability_property=probability_property,
        )
