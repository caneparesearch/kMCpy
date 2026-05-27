"""Site-energy-difference adapters for composite KMC models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import json
import logging
from typing import Any, Optional

import numpy as np
from monty.serialization import loadfn

from kmcpy.models.base import BaseModel, MODEL_FILETYPE, require_model_type
from kmcpy.structure.active_site_index_map import ActiveSiteIndexMap

logger = logging.getLogger(__name__)

_UNIT_FACTORS_TO_MEV = {
    "mev": 1.0,
    "ev": 1000.0,
}


@dataclass(frozen=True)
class MappedOccupationChange:
    """One local occupation change in both kMCpy and external coordinates."""

    kmcpy_site: int
    external_site: int
    old_state: int
    new_state: int
    old_value: Any
    new_value: Any

    def as_flip(self) -> tuple[int, Any]:
        """Return ``(external_site, new_value)`` for smol-style APIs."""
        return (self.external_site, self.new_value)

    def as_system_change_tuple(self) -> tuple[int, Any, Any]:
        """Return ``(external_site, old_value, new_value)``."""
        return (self.external_site, self.old_value, self.new_value)


class CallableSiteEnergyModel(BaseModel):
    """Adapter for direct kMCpy-native site-energy-difference callables.

    The wrapped callable must return the event energy change
    ``E_after_hop - E_before_hop`` for the current ``simulation_state`` and
    ``event``. It receives kMCpy site indices and kMCpy occupation labels. Use
    :class:`MappedSiteEnergyModel` instead when an external code has a different
    site order, state encoding, or live runtime object.

    kMCpy consumes this value in meV, so the adapter converts from ``eV`` when
    requested.

    The callable is resolved from a string reference such as
    ``"package.module:function"`` or ``"package.module.function"`` and is called
    as::

        callable(event=event, simulation_state=simulation_state, **kwargs)
    """

    MODEL_TYPE = "callable_site_energy"
    PAYLOAD_KEY = "callable_site_energy"

    def __init__(
        self,
        callable_ref: str,
        units: str = "meV",
        kwargs: Optional[dict[str, Any]] = None,
        name: str = "CallableSiteEnergyModel",
    ) -> None:
        super().__init__(name=name)
        if not isinstance(callable_ref, str) or not callable_ref.strip():
            raise ValueError("'callable_ref' must be a non-empty string")
        self.callable_ref = callable_ref.strip()
        self.units = _normalize_energy_units(units)
        self.kwargs = dict(kwargs or {})
        self._callable = None

    @property
    def unit_factor_to_mev(self) -> float:
        """Conversion factor from configured units to meV."""
        return _unit_factor_to_mev(self.units)

    def _resolve_callable(self):
        if self._callable is None:
            self._callable = resolve_callable_reference(self.callable_ref)
        return self._callable

    def compute(self, event, simulation_state) -> float:
        """Return ``E_after_hop - E_before_hop`` in meV."""
        raw_value = self._resolve_callable()(
            event=event,
            simulation_state=simulation_state,
            **self.kwargs,
        )
        return _numeric_delta_to_mev(raw_value, self.unit_factor_to_mev)

    def as_dict(self) -> dict[str, Any]:
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "name": self.name,
            "callable_ref": self.callable_ref,
            "units": self.units,
            "kwargs": dict(self.kwargs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CallableSiteEnergyModel":
        if not isinstance(data, dict):
            raise ValueError("CallableSiteEnergyModel payload must be a JSON object")
        if data.get("model_type") == cls.MODEL_TYPE and cls.PAYLOAD_KEY in data:
            data = data[cls.PAYLOAD_KEY]
        if not isinstance(data, dict):
            raise ValueError("CallableSiteEnergyModel payload must be a JSON object")
        return cls(
            callable_ref=data.get("callable_ref") or data.get("callable"),
            units=data.get("units", "meV"),
            kwargs=data.get("kwargs"),
            name=data.get("name", "CallableSiteEnergyModel"),
        )

    @classmethod
    def from_file(cls, filename: str) -> "CallableSiteEnergyModel":
        data = loadfn(filename, cls=None)
        if isinstance(data, dict) and data.get("filetype") == MODEL_FILETYPE:
            data = require_model_type(data, cls.MODEL_TYPE).get(cls.PAYLOAD_KEY)
        return cls.from_dict(data)

    def to(self, filename: str, indent: int = 2) -> None:
        from monty.serialization import dumpfn

        logger.info(
            "Saving callable site-energy-difference adapter to: %s",
            filename,
        )
        dumpfn(self.as_dict(), filename, indent=indent)

    def __str__(self) -> str:
        return (
            f"CallableSiteEnergyModel(callable_ref={self.callable_ref!r}, "
            f"units={self.units!r})"
        )

    def __repr__(self) -> str:
        return (
            "CallableSiteEnergyModel("
            f"callable_ref={self.callable_ref!r}, units={self.units!r}, "
            f"kwargs={self.kwargs!r})"
        )


class MappedSiteEnergyModel(BaseModel):
    """Stateful site-energy-difference adapter with precomputed mappings.

    This adapter is intentionally external-code agnostic. kMCpy owns only the
    cheap, reusable translation from active-site occupations to an external
    occupation representation. The user supplies the actual runtime object and
    delta function for smol, CLEASE, ASE, or project-specific code.

    ``initialize_state`` validates the mapping once, builds the external
    occupation once, and caches active-site lookups. ``compute`` passes only
    the two endpoint changes for the proposed event to ``delta_fn``.
    ``apply_event`` updates only accepted endpoints and optionally calls
    ``apply_fn`` to keep a live external evaluator synchronized.

    ``delta_fn`` is called as::

        delta_fn(
            runtime=runtime,
            external_occupation=external_occupation,
            changes=changes,
            event=event,
            simulation_state=simulation_state,
            **delta_kwargs,
        )

    where ``changes`` is a list of :class:`MappedOccupationChange` objects.
    It must return ``E_after_hop - E_before_hop`` in ``units``.

    ``apply_fn`` is optional and is called before the cached external
    occupation is updated in place.
    """

    MODEL_TYPE = "mapped_site_energy"
    PAYLOAD_KEY = "mapped_site_energy"

    def __init__(
        self,
        delta_fn=None,
        delta_ref: str | None = None,
        delta_kwargs: Optional[dict[str, Any]] = None,
        apply_fn=None,
        apply_ref: str | None = None,
        apply_kwargs: Optional[dict[str, Any]] = None,
        runtime: Any = None,
        runtime_ref: str | None = None,
        runtime_kwargs: Optional[dict[str, Any]] = None,
        site_mapping: Mapping[Any, Any] | Sequence[Any] | None = None,
        state_mapping: Mapping[Any, Any] | Sequence[Any] | None = None,
        state_mapping_by_site: Mapping[Any, Any] | Sequence[Any] | None = None,
        initial_occupation: Sequence[Any] | np.ndarray | None = None,
        external_size: int | None = None,
        external_fill_value: Any = 0,
        external_dtype: str | None = None,
        active_site_index_map: ActiveSiteIndexMap | Mapping[str, Any] | None = None,
        kmcpy_site_order_hash: str | None = None,
        units: str = "eV",
        name: str = "MappedSiteEnergyModel",
    ) -> None:
        super().__init__(name=name)
        self.delta_fn = delta_fn
        self.delta_ref = _normalize_optional_ref(delta_ref)
        self.delta_kwargs = dict(delta_kwargs or {})
        self.apply_fn = apply_fn
        self.apply_ref = _normalize_optional_ref(apply_ref)
        self.apply_kwargs = dict(apply_kwargs or {})
        self.runtime = runtime
        self.runtime_ref = _normalize_optional_ref(runtime_ref)
        self.runtime_kwargs = dict(runtime_kwargs or {})
        self.site_mapping = _normalize_site_mapping(site_mapping)
        self.state_mapping = _normalize_state_mapping(state_mapping)
        self.state_mapping_by_site = _normalize_state_mapping_by_site(
            state_mapping_by_site
        )
        self.initial_occupation = _optional_array_copy(initial_occupation)
        self.external_size = (
            int(external_size) if external_size is not None else None
        )
        self.external_fill_value = external_fill_value
        self.external_dtype = external_dtype
        self.active_site_index_map = _normalize_active_site_index_map(
            active_site_index_map
        )
        normalized_site_order_hash = _normalize_optional_ref(kmcpy_site_order_hash)
        if (
            self.active_site_index_map is not None
            and normalized_site_order_hash is not None
            and normalized_site_order_hash != self.active_site_index_map.fingerprint
        ):
            raise ValueError(
                "MappedSiteEnergyModel kmcpy_site_order_hash does not match "
                "the active_site_index_map fingerprint."
            )
        self.kmcpy_site_order_hash = (
            self.active_site_index_map.fingerprint
            if self.active_site_index_map is not None
            else normalized_site_order_hash
        )
        self.units = _normalize_energy_units(units)

        self.external_occupation: np.ndarray | None = None
        self._site_lookup: np.ndarray | None = None
        self._delta_callable = None
        self._apply_callable = None

    @property
    def unit_factor_to_mev(self) -> float:
        """Conversion factor from configured units to meV."""
        return _unit_factor_to_mev(self.units)

    @property
    def external_site_order_hash(self) -> str:
        """Order-sensitive hash of the active-site to external-site mapping."""
        return _external_site_order_hash(
            site_mapping=self.site_mapping,
            external_size=self.external_size,
            initial_occupation=self.initial_occupation,
        )

    def _resolve_runtime(self):
        if self.runtime is None and self.runtime_ref is not None:
            self.runtime = resolve_callable_reference(self.runtime_ref)(
                **self.runtime_kwargs
            )
        return self.runtime

    def _resolve_delta_fn(self):
        if self.delta_fn is not None:
            return self.delta_fn
        if self._delta_callable is None:
            if self.delta_ref is None:
                raise RuntimeError(
                    "MappedSiteEnergyModel requires delta_fn or delta_ref "
                    "before compute() can run"
                )
            self._delta_callable = resolve_callable_reference(self.delta_ref)
        return self._delta_callable

    def _resolve_apply_fn(self):
        if self.apply_fn is not None:
            return self.apply_fn
        if self.apply_ref is None:
            return None
        if self._apply_callable is None:
            self._apply_callable = resolve_callable_reference(self.apply_ref)
        return self._apply_callable

    def initialize_state(
        self,
        *,
        simulation_state,
        event_lib=None,
        structure=None,
        config=None,
        active_site_index_map=None,
    ) -> None:
        """Build and validate external occupation caches once."""
        occupations = list(simulation_state.occupations)
        self._set_active_site_index_map(active_site_index_map)
        self._validate_kmcpy_site_order(len(occupations))
        self._site_lookup = _build_site_lookup(
            n_sites=len(occupations),
            site_mapping=self.site_mapping,
            external_size=self.external_size,
            initial_occupation=self.initial_occupation,
        )
        self.external_occupation = self._build_external_occupation(occupations)
        self._validate_event_mappings(event_lib, occupations)
        self._resolve_runtime()

    def _set_active_site_index_map(self, active_site_index_map) -> None:
        if active_site_index_map is None:
            return
        normalized = _normalize_active_site_index_map(active_site_index_map)
        if (
            self.kmcpy_site_order_hash is not None
            and normalized.fingerprint != self.kmcpy_site_order_hash
        ):
            raise ValueError(
                "MappedSiteEnergyModel active-site order hash does not "
                "match the current kMCpy active-site index map."
            )
        self.active_site_index_map = normalized
        self.kmcpy_site_order_hash = normalized.fingerprint

    def _validate_kmcpy_site_order(self, occupation_count: int) -> None:
        if self.active_site_index_map is None:
            return
        if self.active_site_index_map.active_site_count != int(occupation_count):
            raise ValueError(
                "MappedSiteEnergyModel active-site index map contains "
                f"{self.active_site_index_map.active_site_count} active sites, "
                f"but the simulation state contains {occupation_count} occupations."
            )

    def _ensure_initialized(self, simulation_state) -> None:
        if self.external_occupation is None or self._site_lookup is None:
            self.initialize_state(simulation_state=simulation_state)

    def _build_external_occupation(self, occupations: Sequence[int]) -> np.ndarray:
        mapped_values = [
            self._external_value(site, int(state))
            for site, state in enumerate(occupations)
        ]
        dtype = _resolve_external_dtype(
            self.external_dtype,
            mapped_values=mapped_values,
            fill_value=self.external_fill_value,
            initial_occupation=self.initial_occupation,
        )

        if self.initial_occupation is not None:
            external = np.asarray(self.initial_occupation, dtype=dtype).copy()
        else:
            external = np.full(
                _external_size_from_lookup(
                    self._site_lookup,
                    fallback=len(occupations),
                    external_size=self.external_size,
                ),
                self.external_fill_value,
                dtype=dtype,
            )

        for kmcpy_site, mapped_value in enumerate(mapped_values):
            external[int(self._site_lookup[kmcpy_site])] = mapped_value
        return external

    def _external_site(self, kmcpy_site: int) -> int:
        if self._site_lookup is None:
            raise RuntimeError("MappedSiteEnergyModel has not been initialized")
        return int(self._site_lookup[int(kmcpy_site)])

    def _external_value(self, kmcpy_site: int, state_value: int):
        return _external_state_value(
            kmcpy_site,
            state_value,
            state_mapping=self.state_mapping,
            state_mapping_by_site=self.state_mapping_by_site,
        )

    def _changes_from_pre_state(self, event, occupations) -> list[MappedOccupationChange]:
        from_site, to_site = (int(site) for site in event.mobile_ion_indices)
        from_state = int(occupations[from_site])
        to_state = int(occupations[to_site])
        return self._mapped_changes(
            (
                (from_site, from_state, to_state),
                (to_site, to_state, from_state),
            )
        )

    def _changes_from_post_state(self, event, occupations) -> list[MappedOccupationChange]:
        from_site, to_site = (int(site) for site in event.mobile_ion_indices)
        from_state = int(occupations[from_site])
        to_state = int(occupations[to_site])
        return self._mapped_changes(
            (
                (from_site, to_state, from_state),
                (to_site, from_state, to_state),
            )
        )

    def _mapped_changes(
        self, changes: Sequence[tuple[int, int, int]]
    ) -> list[MappedOccupationChange]:
        mapped = []
        for kmcpy_site, old_state, new_state in changes:
            external_site = self._external_site(kmcpy_site)
            old_value = self._external_value(kmcpy_site, old_state)
            new_value = self._external_value(kmcpy_site, new_state)
            if old_value == new_value:
                continue
            mapped.append(
                MappedOccupationChange(
                    kmcpy_site=int(kmcpy_site),
                    external_site=external_site,
                    old_state=int(old_state),
                    new_state=int(new_state),
                    old_value=old_value,
                    new_value=new_value,
                )
            )
        return mapped

    def _validate_event_mappings(self, event_lib, occupations: Sequence[int]) -> None:
        events = getattr(event_lib, "events", None)
        if events is None:
            return
        for event_index, event in enumerate(events):
            try:
                self._changes_from_pre_state(event, occupations)
            except (IndexError, KeyError, ValueError) as exc:
                raise ValueError(
                    "MappedSiteEnergyModel occupation mapping is incompatible "
                    f"with event {event_index}"
                ) from exc

    def compute(self, event, simulation_state) -> float:
        """Return ``E_after_hop - E_before_hop`` in meV."""
        self._ensure_initialized(simulation_state)
        changes = self._changes_from_pre_state(event, simulation_state.occupations)
        if not changes:
            return 0.0
        raw_value = self._resolve_delta_fn()(
            runtime=self._resolve_runtime(),
            external_occupation=self.external_occupation,
            changes=changes,
            event=event,
            simulation_state=simulation_state,
            **self.delta_kwargs,
        )
        return _numeric_delta_to_mev(raw_value, self.unit_factor_to_mev)

    def apply_event(self, *, event, simulation_state) -> None:
        """Commit one accepted event to the external runtime and cache."""
        self._ensure_initialized(simulation_state)
        changes = self._changes_from_post_state(event, simulation_state.occupations)
        if not changes:
            return

        apply_fn = self._resolve_apply_fn()
        if apply_fn is not None:
            apply_fn(
                runtime=self._resolve_runtime(),
                external_occupation=self.external_occupation,
                changes=changes,
                event=event,
                simulation_state=simulation_state,
                **self.apply_kwargs,
            )

        for change in changes:
            self.external_occupation[change.external_site] = change.new_value

    def as_dict(self) -> dict[str, Any]:
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "name": self.name,
            "delta_ref": self.delta_ref,
            "delta_kwargs": dict(self.delta_kwargs),
            "apply_ref": self.apply_ref,
            "apply_kwargs": dict(self.apply_kwargs),
            "runtime_ref": self.runtime_ref,
            "runtime_kwargs": dict(self.runtime_kwargs),
            "site_mapping": self.site_mapping,
            "state_mapping": self.state_mapping,
            "state_mapping_by_site": self.state_mapping_by_site,
            "initial_occupation": _optional_array_payload(self.initial_occupation),
            "external_size": self.external_size,
            "external_fill_value": self.external_fill_value,
            "external_dtype": self.external_dtype,
            "active_site_index_map": (
                self.active_site_index_map.as_dict()
                if self.active_site_index_map is not None
                else None
            ),
            "kmcpy_site_order_hash": self.kmcpy_site_order_hash,
            "external_site_order_hash": self.external_site_order_hash,
            "units": self.units,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MappedSiteEnergyModel":
        if not isinstance(data, dict):
            raise ValueError("MappedSiteEnergyModel payload must be a JSON object")
        if data.get("model_type") == cls.MODEL_TYPE and cls.PAYLOAD_KEY in data:
            data = data[cls.PAYLOAD_KEY]
        model = cls(
            delta_ref=data.get("delta_ref"),
            delta_kwargs=data.get("delta_kwargs"),
            apply_ref=data.get("apply_ref"),
            apply_kwargs=data.get("apply_kwargs"),
            runtime_ref=data.get("runtime_ref"),
            runtime_kwargs=data.get("runtime_kwargs"),
            site_mapping=data.get("site_mapping"),
            state_mapping=data.get("state_mapping"),
            state_mapping_by_site=data.get("state_mapping_by_site"),
            initial_occupation=data.get("initial_occupation"),
            external_size=data.get("external_size"),
            external_fill_value=data.get("external_fill_value", 0),
            external_dtype=data.get("external_dtype"),
            active_site_index_map=data.get("active_site_index_map"),
            kmcpy_site_order_hash=data.get("kmcpy_site_order_hash"),
            units=data.get("units", "eV"),
            name=data.get("name", "MappedSiteEnergyModel"),
        )
        stored_external_hash = data.get("external_site_order_hash")
        if (
            stored_external_hash is not None
            and str(stored_external_hash) != model.external_site_order_hash
        ):
            raise ValueError(
                "MappedSiteEnergyModel external_site_order_hash does not "
                "match its site_mapping/external_size metadata."
            )
        return model

    @classmethod
    def from_file(cls, filename: str) -> "MappedSiteEnergyModel":
        data = loadfn(filename, cls=None)
        if isinstance(data, dict) and data.get("filetype") == MODEL_FILETYPE:
            data = require_model_type(data, cls.MODEL_TYPE).get(cls.PAYLOAD_KEY)
        return cls.from_dict(data)

    def to(self, filename: str, indent: int = 2) -> None:
        from monty.serialization import dumpfn

        logger.info(
            "Saving mapped site-energy-difference adapter to: %s",
            filename,
        )
        dumpfn(self.as_dict(), filename, indent=indent)

    def __str__(self) -> str:
        return (
            "MappedSiteEnergyModel("
            f"delta_ref={self.delta_ref!r}, units={self.units!r})"
        )

    def __repr__(self) -> str:
        return (
            "MappedSiteEnergyModel("
            f"delta_ref={self.delta_ref!r}, runtime_ref={self.runtime_ref!r}, "
            f"units={self.units!r})"
        )


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


def constant_site_energy_difference(
    event,
    simulation_state,
    value: float = 0.0,
) -> float:
    """Small helper used by examples/tests to return a constant difference."""
    return float(value)


def _normalize_energy_units(units: str) -> str:
    token = str(units).strip()
    if token.lower() not in _UNIT_FACTORS_TO_MEV:
        raise ValueError("Site-energy-difference units must be 'meV' or 'eV'")
    return "meV" if token.lower() == "mev" else "eV"


def _unit_factor_to_mev(units: str) -> float:
    return _UNIT_FACTORS_TO_MEV[str(units).lower()]


def _numeric_delta_to_mev(value, unit_factor_to_mev: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(
            "Site-energy-difference callable must return a numeric value"
        )
    return float(value) * unit_factor_to_mev


def _normalize_optional_ref(ref: str | None) -> str | None:
    if ref is None:
        return None
    token = str(ref).strip()
    return token or None


def _normalize_site_mapping(
    site_mapping: Mapping[Any, Any] | Sequence[Any] | None,
) -> dict[int, int] | None:
    if site_mapping is None:
        return None
    if isinstance(site_mapping, Mapping):
        return {int(key): int(value) for key, value in site_mapping.items()}
    return {index: int(value) for index, value in enumerate(site_mapping)}


def _normalize_state_mapping(
    state_mapping: Mapping[Any, Any] | Sequence[Any] | None,
) -> dict[int, Any] | None:
    if state_mapping is None:
        return None
    if isinstance(state_mapping, Mapping):
        return {int(key): value for key, value in state_mapping.items()}
    return {index: value for index, value in enumerate(state_mapping)}


def _normalize_state_mapping_by_site(
    state_mapping_by_site: Mapping[Any, Any] | Sequence[Any] | None,
) -> dict[int, dict[int, Any]] | None:
    if state_mapping_by_site is None:
        return None
    if isinstance(state_mapping_by_site, Mapping):
        return {
            int(site): _normalize_state_mapping(mapping) or {}
            for site, mapping in state_mapping_by_site.items()
        }
    return {
        site: _normalize_state_mapping(mapping) or {}
        for site, mapping in enumerate(state_mapping_by_site)
        if mapping is not None
    }


def _normalize_active_site_index_map(
    active_site_index_map: ActiveSiteIndexMap | Mapping[str, Any] | None,
) -> ActiveSiteIndexMap | None:
    if active_site_index_map is None:
        return None
    if isinstance(active_site_index_map, ActiveSiteIndexMap):
        return active_site_index_map
    if isinstance(active_site_index_map, Mapping):
        return ActiveSiteIndexMap.from_dict(active_site_index_map)
    raise TypeError(
        "active_site_index_map must be an ActiveSiteIndexMap, a serialized "
        "mapping, or None"
    )


def _external_site_order_hash(
    *,
    site_mapping: dict[int, int] | None,
    external_size: int | None,
    initial_occupation: np.ndarray | None,
) -> str:
    initial_length = (
        int(len(initial_occupation)) if initial_occupation is not None else None
    )
    payload = {
        "format": "kmcpy.mapped_site_energy.external_site_order.v1",
        "site_mapping": (
            [
                [int(site), int(external_site)]
                for site, external_site in sorted(site_mapping.items())
            ]
            if site_mapping is not None
            else None
        ),
        "external_size": int(external_size) if external_size is not None else None,
        "initial_occupation_length": initial_length,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _build_site_lookup(
    *,
    n_sites: int,
    site_mapping: dict[int, int] | None,
    external_size: int | None,
    initial_occupation: np.ndarray | None,
) -> np.ndarray:
    if site_mapping is None:
        lookup = np.arange(int(n_sites), dtype=np.int64)
    else:
        missing = [site for site in range(int(n_sites)) if site not in site_mapping]
        if missing:
            raise ValueError(
                "site_mapping does not cover all kMCpy active sites; "
                f"missing first site {missing[0]}"
            )
        lookup = np.array([site_mapping[site] for site in range(int(n_sites))])

    if np.any(lookup < 0):
        raise ValueError("External site indices must be nonnegative")
    upper_bound = external_size
    if initial_occupation is not None:
        upper_bound = len(initial_occupation)
    if upper_bound is not None and len(lookup) and int(np.max(lookup)) >= upper_bound:
        raise ValueError(
            "site_mapping points outside the external occupation length "
            f"({int(np.max(lookup))} >= {upper_bound})"
        )
    return lookup


def _external_size_from_lookup(
    lookup: np.ndarray,
    *,
    fallback: int,
    external_size: int | None,
) -> int:
    if external_size is not None:
        return int(external_size)
    if lookup is None or len(lookup) == 0:
        return int(fallback)
    return int(np.max(lookup)) + 1


def _external_state_value(
    kmcpy_site: int,
    state_value: int,
    *,
    state_mapping: dict[int, Any] | None,
    state_mapping_by_site: dict[int, dict[int, Any]] | None,
):
    site = int(kmcpy_site)
    state = int(state_value)
    mapping = None
    if state_mapping_by_site is not None:
        mapping = state_mapping_by_site.get(site)
    if mapping is None:
        mapping = state_mapping
    if mapping is None:
        return state
    try:
        return mapping[state]
    except KeyError as exc:
        raise ValueError(
            f"No external state mapping is defined for kMCpy site {site}, "
            f"state {state}"
        ) from exc


def _resolve_external_dtype(
    external_dtype: str | None,
    *,
    mapped_values: Sequence[Any],
    fill_value: Any,
    initial_occupation: np.ndarray | None,
):
    if external_dtype is not None:
        return np.dtype(external_dtype)
    values = list(mapped_values)
    if initial_occupation is not None:
        return initial_occupation.dtype
    if values and all(isinstance(value, (int, np.integer)) for value in values):
        return np.int64
    if values and all(isinstance(value, (int, float, np.number)) for value in values):
        return float
    if isinstance(fill_value, (int, np.integer)) and not values:
        return np.int64
    return object


def _optional_array_copy(value):
    if value is None:
        return None
    return np.asarray(value).copy()


def _optional_array_payload(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)
