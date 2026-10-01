"""Site-energy-difference models for composite KMC simulations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
from typing import Any, Optional

import numpy as np

from kmcpy.callables import (
    call_with_supported_keywords,
    resolve_callable_reference,
    supported_keyword_names,
)
from kmcpy.models.base import BaseModel
from kmcpy.models.external_site_mapping import ExternalSiteMapping
from kmcpy.structure.active_site_order import ActiveSiteOrder

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


class SiteEnergyModel(BaseModel):
    """Site-energy-difference model with optional external-site mapping.

    The model returns ``E_after_hop - E_before_hop`` for a proposed event. If
    ``site_mapping`` and state mappings are omitted, the callable receives kMCpy
    active-site indices and occupation labels. If an external code has a
    different site order or state encoding, provide mappings once before KMC
    starts; per-event evaluation then touches only the two event endpoints.

    ``initialize_state`` validates the mapping once, builds the external
    occupation once, and caches site/state mapping dictionaries as lookup
    arrays. ``compute`` passes only the two endpoint changes for the proposed
    event to ``compute_fn``.
    ``apply_event`` updates only accepted endpoints and optionally calls
    ``apply_fn`` to keep a live external evaluator synchronized.

    ``compute_fn`` is called as::

        compute_fn(
            runtime=runtime,
            external_occupation=external_occupation,
            changes=changes,
            event=event,
            simulation_state=simulation_state,
            **compute_kwargs,
        )

    where ``changes`` is a list of :class:`MappedOccupationChange` objects.
    Simple callables may also accept only the subset they need, such as
    ``event`` and ``simulation_state``. It must return
    ``E_after_hop - E_before_hop`` in ``units``.

    ``apply_fn`` is optional and is called before the cached external
    occupation is updated in place.
    """

    MODEL_TYPE = "site_energy"
    PAYLOAD_KEY = "site_energy"

    def __init__(
        self,
        compute_fn=None,
        compute_ref: str | None = None,
        compute_kwargs: Optional[dict[str, Any]] = None,
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
        active_site_order: ActiveSiteOrder | Mapping[str, Any] | None = None,
        active_site_order_hash: str | None = None,
        units: str = "eV",
        name: str = "SiteEnergyModel",
    ) -> None:
        super().__init__(name=name)
        self.compute_fn = compute_fn
        self.compute_ref = _normalize_optional_ref(compute_ref)
        self.compute_kwargs = dict(compute_kwargs or {})
        self.apply_fn = apply_fn
        self.apply_ref = _normalize_optional_ref(apply_ref)
        self.apply_kwargs = dict(apply_kwargs or {})
        self.runtime = runtime
        self.runtime_ref = _normalize_optional_ref(runtime_ref)
        self.runtime_kwargs = dict(runtime_kwargs or {})
        self.external_mapping = ExternalSiteMapping(
            site_mapping=site_mapping,
            state_mapping=state_mapping,
            state_mapping_by_site=state_mapping_by_site,
            initial_occupation=initial_occupation,
            external_size=external_size,
            external_fill_value=external_fill_value,
            external_dtype=external_dtype,
        )
        self.active_site_order = _normalize_active_site_order(
            active_site_order
        )
        normalized_site_order_hash = _normalize_optional_ref(active_site_order_hash)
        if (
            self.active_site_order is not None
            and normalized_site_order_hash is not None
            and normalized_site_order_hash != self.active_site_order.fingerprint
        ):
            raise ValueError(
                "SiteEnergyModel active_site_order_hash does not match "
                "the active_site_order fingerprint."
            )
        self.active_site_order_hash = (
            self.active_site_order.fingerprint
            if self.active_site_order is not None
            else normalized_site_order_hash
        )
        self.units = _normalize_energy_units(units)

        self.external_occupation: np.ndarray | None = None
        self._compute_callable = None
        self._apply_callable = None

    @property
    def unit_factor_to_mev(self) -> float:
        """Conversion factor from configured units to meV."""
        return _unit_factor_to_mev(self.units)

    @property
    def external_site_order_hash(self) -> str:
        """Order-sensitive hash of the active-site to external-site mapping."""
        return self.external_mapping.order_hash

    def _resolve_runtime(self):
        if self.runtime is None and self.runtime_ref is not None:
            self.runtime = resolve_callable_reference(self.runtime_ref)(
                **self.runtime_kwargs
            )
        return self.runtime

    def _resolve_compute_fn(self):
        if self.compute_fn is not None:
            return self.compute_fn
        if self._compute_callable is None:
            if self.compute_ref is None:
                raise RuntimeError(
                    "SiteEnergyModel requires compute_fn or compute_ref "
                    "before compute() can run"
                )
            self._compute_callable = resolve_callable_reference(self.compute_ref)
        return self._compute_callable

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
        active_site_order=None,
    ) -> None:
        """Build and validate external occupation caches once."""
        occupations = list(simulation_state.occupations)
        self._set_active_site_order(active_site_order)
        self._validate_kmcpy_site_order(len(occupations))
        self.external_mapping.prepare(len(occupations))
        self.external_occupation = self.external_mapping.build_external_occupation(
            occupations
        )
        self._validate_event_mappings(event_lib, occupations)
        self._resolve_runtime()

    def _set_active_site_order(self, active_site_order) -> None:
        if active_site_order is None:
            return
        normalized = _normalize_active_site_order(active_site_order)
        if (
            self.active_site_order_hash is not None
            and normalized.fingerprint != self.active_site_order_hash
        ):
            raise ValueError(
                "SiteEnergyModel active-site order hash does not "
                "match the current kMCpy active-site order."
            )
        self.active_site_order = normalized
        self.active_site_order_hash = normalized.fingerprint

    def _validate_kmcpy_site_order(self, occupation_count: int) -> None:
        if self.active_site_order is None:
            return
        if self.active_site_order.active_site_count != int(occupation_count):
            raise ValueError(
                "SiteEnergyModel active-site order contains "
                f"{self.active_site_order.active_site_count} active sites, "
                f"but the simulation state contains {occupation_count} occupations."
            )

    def _ensure_initialized(self, simulation_state) -> None:
        if self.external_occupation is None or not self.external_mapping.is_prepared:
            self.initialize_state(simulation_state=simulation_state)

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
        mapping = self.external_mapping
        mapped = []
        for kmcpy_site, old_state, new_state in changes:
            external_site = mapping.external_site(kmcpy_site)
            old_value = mapping.external_value(kmcpy_site, old_state)
            new_value = mapping.external_value(kmcpy_site, new_state)
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
                    "SiteEnergyModel occupation mapping is incompatible "
                    f"with event {event_index}"
                ) from exc

    def compute(self, event, simulation_state) -> float:
        """Return ``E_after_hop - E_before_hop`` in meV."""
        self._ensure_initialized(simulation_state)
        changes = self._changes_from_pre_state(event, simulation_state.occupations)
        if not changes:
            return 0.0
        compute_fn = self._resolve_compute_fn()
        accepted = supported_keyword_names(compute_fn)
        if accepted is not None:
            unused_kwargs = sorted(set(self.compute_kwargs) - accepted)
            if unused_kwargs:
                raise TypeError(
                    "SiteEnergyModel compute_kwargs contains keys not accepted by "
                    f"{compute_fn}: {unused_kwargs}"
                )
        raw_value = call_with_supported_keywords(
            compute_fn,
            {
                "runtime": self._resolve_runtime(),
                "external_occupation": self.external_occupation,
                "changes": changes,
                "event": event,
                "simulation_state": simulation_state,
                **self.compute_kwargs,
            },
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
            "compute_ref": self.compute_ref,
            "compute_kwargs": dict(self.compute_kwargs),
            "apply_ref": self.apply_ref,
            "apply_kwargs": dict(self.apply_kwargs),
            "runtime_ref": self.runtime_ref,
            "runtime_kwargs": dict(self.runtime_kwargs),
            **self.external_mapping.as_dict(),
            "active_site_order": (
                self.active_site_order.as_dict()
                if self.active_site_order is not None
                else None
            ),
            "active_site_order_hash": self.active_site_order_hash,
            "external_site_order_hash": self.external_site_order_hash,
            "units": self.units,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SiteEnergyModel":
        if not isinstance(data, dict):
            raise ValueError("SiteEnergyModel payload must be a JSON object")
        data = cls._unwrap_model_file(data)
        model = cls(
            compute_ref=data.get("compute_ref"),
            compute_kwargs=data.get("compute_kwargs"),
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
            active_site_order=data.get("active_site_order"),
            active_site_order_hash=data.get("active_site_order_hash"),
            units=data.get("units", "eV"),
            name=data.get("name", "SiteEnergyModel"),
        )
        stored_external_hash = data.get("external_site_order_hash")
        if (
            stored_external_hash is not None
            and str(stored_external_hash) != model.external_site_order_hash
        ):
            raise ValueError(
                "SiteEnergyModel external_site_order_hash does not "
                "match its site_mapping/external_size metadata."
            )
        return model

    def __str__(self) -> str:
        return (
            "SiteEnergyModel("
            f"compute_ref={self.compute_ref!r}, units={self.units!r})"
        )

    def __repr__(self) -> str:
        return (
            "SiteEnergyModel("
            f"compute_ref={self.compute_ref!r}, runtime_ref={self.runtime_ref!r}, "
            f"units={self.units!r})"
        )


def constant_site_energy_difference(
    event=None,
    simulation_state=None,
    value: float = 0.0,
    **kwargs,
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


def _normalize_active_site_order(
    active_site_order: ActiveSiteOrder | Mapping[str, Any] | None,
) -> ActiveSiteOrder | None:
    if active_site_order is None:
        return None
    if isinstance(active_site_order, ActiveSiteOrder):
        return active_site_order
    if isinstance(active_site_order, Mapping):
        return ActiveSiteOrder.from_dict(active_site_order)
    raise TypeError(
        "active_site_order must be an ActiveSiteOrder, a serialized "
        "mapping, or None"
    )
