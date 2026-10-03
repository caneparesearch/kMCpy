#!/usr/bin/env python
"""Direct local-barrier rules for KMC event rates.

``LocalBarrierModel`` is the lightweight alternative to fitting a local cluster
expansion. It is useful when the migration barrier can be written directly as a
small set of ordered rules:

* use a constant fallback barrier for every hop;
* count selected occupation states in an event local environment;
* count chemical species after mapping occupation states to species labels;
* match a short wildcard occupation pattern; or
* match an exact event/local-occupation entry.

The model works with the same compact KMC state used by the simulator. State
values are nonnegative integer indices. For binary models, ``occupied`` and
``template`` are aliases for state ``0`` and ``vacant`` and ``mismatch`` are
aliases for state ``1``. Multicomponent rules can use explicit state indices
such as ``2`` or ``3``. Rule order is significant: the first matching rule
supplies the selected property, usually ``barrier`` in meV. If no rule matches,
``default_properties`` are used when present. ``compute_probability`` returns an
event rate in Hz using temperature in K and attempt frequency in Hz.

Minimal setup::

    from kmcpy.models import LocalBarrierModel

    model = LocalBarrierModel.constant_barrier(300.0)
    model.to("model.json")

Rule-based setup::

    model = LocalBarrierModel(default_barrier=300.0)
    model.add_state_count_rule(
        name="crowded",
        sites="local_env",
        state="occupied",
        min_count=3,
        barrier=450.0,
    )

The saved ``model.json`` can be referenced by ``model_file`` in a simulation
configuration. ``BaseModel.from_config`` dispatches to this class when the model
file carries this class' Monty metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, TYPE_CHECKING

import numpy as np

from kmcpy.event import Event, event_direction
from kmcpy.models.base import BaseModel
from kmcpy.simulator.state import State
from kmcpy.units import BOLTZMANN_CONSTANT_MEV_PER_K as K_B_MEV_PER_K

if TYPE_CHECKING:
    from kmcpy.simulator.config import RuntimeConfig

logger = logging.getLogger(__name__)


STATE_VALUE_ALIASES = {
    "occupied": 0,
    "match": 0,
    "template": 0,
    "vacant": 1,
    "vacancy": 1,
    "mismatch": 1,
    "other": 1,
}

SITE_SELECTORS = {
    "canonical",
    "local_env",
    "mobile_ion",
    "from",
    "to",
    "all",
}

RULE_TYPES = {
    "constant",
    "exact",
    "pattern",
    "state_count",
    "species_count",
}

COUNT_KEYS = ("count", "min_count", "max_count")


def _normalize_index_sequence(
    values: Any, field_name: str, allow_empty: bool = False
) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"'{field_name}' must be a list or tuple of integers")
    if not values and not allow_empty:
        raise ValueError(f"'{field_name}' must be non-empty")

    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"'{field_name}' must contain integers only")
        normalized.append(int(value))
    return tuple(normalized)


def _normalize_state_value(value: Any, field_name: str = "state") -> int:
    message = (
        f"'{field_name}' must be a nonnegative integer state index or one of "
        f"{sorted(STATE_VALUE_ALIASES)}"
    )
    if isinstance(value, bool):
        raise TypeError(message)
    if isinstance(value, int):
        if value < 0:
            raise ValueError(message)
        return int(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in STATE_VALUE_ALIASES:
            return STATE_VALUE_ALIASES[token]
        try:
            state_value = int(token)
        except ValueError:
            raise ValueError(message) from None
        if state_value < 0:
            raise ValueError(message)
        return state_value
    raise ValueError(message)


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


def _occupations_at(occupations: Sequence[int], sites: tuple[int, ...]) -> tuple[int, ...]:
    try:
        return tuple(int(occupations[site_index]) for site_index in sites)
    except IndexError as exc:
        raise IndexError(
            "Rule site index is out of range for provided simulation occupations"
        ) from exc


@dataclass(frozen=True)
class EventSites:
    """Validated site indices of one event.

    ``canonical`` is ``mobile_ion_indices`` followed by ``local_env_indices``
    with duplicates removed; exact rules store occupations in this order.
    """

    mobile_ion_indices: tuple[int, ...]
    local_env_indices: tuple[int, ...]
    canonical: tuple[int, ...]

    @classmethod
    def from_indices(
        cls,
        mobile_ion_indices: Any,
        local_env_indices: Any,
        field_prefix: str = "",
    ) -> "EventSites":
        mobile = _normalize_index_sequence(
            mobile_ion_indices, f"{field_prefix}mobile_ion_indices"
        )
        local_env = _normalize_index_sequence(
            local_env_indices, f"{field_prefix}local_env_indices", allow_empty=True
        )
        return cls(mobile, local_env, tuple(dict.fromkeys(mobile + local_env)))

    @classmethod
    def from_event(cls, event: Event) -> "EventSites":
        return cls.from_indices(
            event.mobile_ion_indices, event.local_env_indices, field_prefix="event."
        )

    def select(self, sites: str | tuple[int, ...], site_count: int) -> tuple[int, ...]:
        """Return the site indices chosen by a rule's ``sites`` selector."""
        if isinstance(sites, tuple):
            return sites
        if sites == "canonical":
            return self.canonical
        if sites == "local_env":
            return self.local_env_indices
        if sites == "mobile_ion":
            return self.mobile_ion_indices
        if sites == "from":
            return (self.mobile_ion_indices[0],)
        if sites == "to":
            return (self.mobile_ion_indices[1],)
        if sites == "all":
            return tuple(range(site_count))
        raise ValueError(f"Unsupported sites selector '{sites}'")


@dataclass(frozen=True)
class BarrierRule:
    """One validated ``LocalBarrierModel`` rule.

    Build rules from plain dictionaries with :meth:`from_dict`, which accepts
    the same keys as model files and ``LocalBarrierModel(rules=[...])``;
    :meth:`as_dict` writes them back. Fields that do not apply to a rule's
    ``type`` are ``None``.
    """

    name: str
    type: str
    properties: dict[str, float]
    mobile_ion_indices: tuple[int, ...] | None = None
    local_env_indices: tuple[int, ...] | None = None
    occupations: tuple[int, ...] | None = None
    pattern: tuple[int | str, ...] | None = None
    sites: str | tuple[int, ...] | None = None
    state: int | None = None
    species: tuple[str, ...] | None = None
    count: int | None = None
    min_count: int | None = None
    max_count: int | None = None

    @classmethod
    def from_dict(cls, rule: dict[str, Any], default_name: str = "rule") -> "BarrierRule":
        """Validate and normalize a rule dictionary."""
        if not isinstance(rule, dict):
            raise TypeError("Each local barrier rule must be a dictionary")

        rule_type = cls._parse_type(rule)
        fields: dict[str, Any] = {
            "name": str(rule.get("name") or default_name),
            "type": rule_type,
            "properties": cls._parse_properties(rule),
        }
        if "mobile_ion_indices" in rule:
            fields["mobile_ion_indices"] = _normalize_index_sequence(
                rule["mobile_ion_indices"], "mobile_ion_indices"
            )
        if "local_env_indices" in rule:
            fields["local_env_indices"] = _normalize_index_sequence(
                rule["local_env_indices"], "local_env_indices", allow_empty=True
            )

        if rule_type == "exact":
            event_sites = EventSites.from_indices(
                rule.get("mobile_ion_indices"), rule.get("local_env_indices")
            )
            occupations = cls._parse_occupations(rule.get("occupations"))
            if len(occupations) != len(event_sites.canonical):
                raise ValueError(
                    "Exact rule occupation length must match canonical site count "
                    f"({len(event_sites.canonical)}), got {len(occupations)}"
                )
            fields["mobile_ion_indices"] = event_sites.mobile_ion_indices
            fields["local_env_indices"] = event_sites.local_env_indices
            fields["occupations"] = occupations

        elif rule_type == "pattern":
            fields["sites"] = cls._parse_sites(rule.get("sites"), "canonical")
            fields["pattern"] = cls._parse_pattern(rule.get("pattern"))

        elif rule_type == "state_count":
            fields["sites"] = cls._parse_sites(rule.get("sites"), "local_env")
            if "occupation" in rule:
                fields["state"] = _normalize_state_value(rule["occupation"], "occupation")
            else:
                fields["state"] = _normalize_state_value(rule.get("state"), "state")
            fields.update(cls._parse_count_constraints(rule))

        elif rule_type == "species_count":
            fields["sites"] = cls._parse_sites(rule.get("sites"), "local_env")
            fields["species"] = cls._parse_species(rule.get("species"))
            fields.update(cls._parse_count_constraints(rule))

        return cls(**fields)

    def as_dict(self) -> dict[str, Any]:
        """Serialize to the model-file rule payload."""
        payload: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "properties": dict(self.properties),
        }
        for key in ("mobile_ion_indices", "local_env_indices", "occupations", "pattern"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = list(value)
        if self.sites is not None:
            payload["sites"] = list(self.sites) if isinstance(self.sites, tuple) else self.sites
        if self.state is not None:
            payload["state"] = self.state
        if self.species is not None:
            payload["species"] = self.species[0] if len(self.species) == 1 else list(self.species)
        for key in COUNT_KEYS:
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload

    @property
    def canonical_site_indices(self) -> tuple[int, ...] | None:
        """Canonical event sites of an exact rule, otherwise ``None``."""
        if self.type != "exact":
            return None
        return tuple(dict.fromkeys(self.mobile_ion_indices + self.local_env_indices))

    @property
    def exact_key(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Identity of an exact rule, used to reject duplicates."""
        return (self.mobile_ion_indices, self.canonical_site_indices, self.occupations)

    def matches(
        self,
        event_sites: EventSites,
        occupations: Sequence[int],
        species_for_site: Callable[[int, int], str],
    ) -> bool:
        """Return whether this rule applies to an event in the current state."""
        if (
            self.mobile_ion_indices is not None
            and self.mobile_ion_indices != event_sites.mobile_ion_indices
        ):
            return False
        if (
            self.local_env_indices is not None
            and self.local_env_indices != event_sites.local_env_indices
        ):
            return False

        if self.type == "constant":
            return True

        if self.type == "exact":
            if event_sites.canonical != self.canonical_site_indices:
                return False
            return _occupations_at(occupations, event_sites.canonical) == self.occupations

        sites = event_sites.select(self.sites, len(occupations))
        current = _occupations_at(occupations, sites)

        if self.type == "pattern":
            if len(current) != len(self.pattern):
                raise ValueError(
                    f"Pattern rule '{self.name}' has length {len(self.pattern)} "
                    f"but selected {len(current)} sites"
                )
            return all(
                expected == "*" or expected == actual
                for expected, actual in zip(self.pattern, current)
            )

        if self.type == "state_count":
            return self._count_matches(sum(1 for value in current if value == self.state))

        if self.type == "species_count":
            species = set(self.species)
            return self._count_matches(
                sum(
                    1
                    for site_index, occupation in zip(sites, current)
                    if species_for_site(site_index, occupation) in species
                )
            )

        raise ValueError(f"Unsupported rule type '{self.type}'")

    def _count_matches(self, count: int) -> bool:
        if self.count is not None and count != self.count:
            return False
        if self.min_count is not None and count < self.min_count:
            return False
        if self.max_count is not None and count > self.max_count:
            return False
        return True

    @staticmethod
    def _parse_type(rule: dict[str, Any]) -> str:
        rule_type = rule.get("type")
        if rule_type is None:
            if "occupations" in rule and "mobile_ion_indices" in rule:
                return "exact"
            if "pattern" in rule:
                return "pattern"
            if "species" in rule:
                return "species_count"
            if "state" in rule or "occupation" in rule:
                return "state_count"
            return "constant"

        if not isinstance(rule_type, str) or not rule_type.strip():
            raise ValueError("Rule 'type' must be a non-empty string")
        rule_type = rule_type.strip()
        if rule_type not in RULE_TYPES:
            raise ValueError(
                f"Unsupported local barrier rule type '{rule_type}'. "
                f"Supported types: {sorted(RULE_TYPES)}"
            )
        return rule_type

    @staticmethod
    def _parse_properties(rule: dict[str, Any]) -> dict[str, float]:
        properties = dict(rule.get("properties", {}))
        if "barrier" in rule:
            barrier = rule["barrier"]
            if isinstance(barrier, bool) or not isinstance(barrier, (int, float)):
                raise TypeError("'barrier' must be numeric")
            properties.setdefault("barrier", float(barrier))
        return _normalize_properties(properties)

    @staticmethod
    def _parse_occupations(values: Any) -> tuple[int, ...]:
        if not isinstance(values, (list, tuple)):
            raise TypeError("'occupations' must be a list or tuple")
        if not values:
            raise ValueError("'occupations' must be non-empty")
        return tuple(_normalize_state_value(value, "occupations") for value in values)

    @staticmethod
    def _parse_pattern(values: Any) -> tuple[int | str, ...]:
        if not isinstance(values, (list, tuple)):
            raise TypeError("'pattern' must be a list or tuple")
        if not values:
            raise ValueError("'pattern' must be non-empty")
        return tuple(
            "*"
            if isinstance(value, str) and value.strip() == "*"
            else _normalize_state_value(value, "pattern")
            for value in values
        )

    @staticmethod
    def _parse_sites(value: Any, default: str) -> str | tuple[int, ...]:
        if value is None:
            return default
        if isinstance(value, str):
            token = value.strip()
            if token not in SITE_SELECTORS:
                raise ValueError(
                    f"Unsupported sites selector '{value}'. "
                    f"Supported selectors: {sorted(SITE_SELECTORS)}"
                )
            return token
        return _normalize_index_sequence(value, "sites")

    @staticmethod
    def _parse_species(species: Any) -> tuple[str, ...]:
        if isinstance(species, str):
            return (species,)
        if isinstance(species, (list, tuple)) and species:
            for item in species:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError("'species' entries must be non-empty strings")
            return tuple(species)
        raise ValueError("'species' must be a string or non-empty list")

    @staticmethod
    def _parse_count_constraints(rule: dict[str, Any]) -> dict[str, int]:
        constraints: dict[str, int] = {}
        for key in COUNT_KEYS:
            if key not in rule:
                continue
            value = rule[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"'{key}' must be an integer")
            if value < 0:
                raise ValueError(f"'{key}' must be non-negative")
            constraints[key] = int(value)

        if not constraints:
            raise ValueError(
                "Count rules must provide at least one of "
                "'count', 'min_count', or 'max_count'"
            )
        if "count" in constraints and (
            "min_count" in constraints or "max_count" in constraints
        ):
            raise ValueError("'count' cannot be combined with min_count or max_count")
        return constraints


def _normalize_site_species(site_species: Any) -> dict[int, dict[int, str]]:
    if site_species is None:
        return {}
    if not isinstance(site_species, dict):
        raise TypeError("'site_species' must be a mapping")

    normalized: dict[int, dict[int, str]] = {}
    for site_key, state_mapping in site_species.items():
        try:
            site_index = int(site_key)
        except (TypeError, ValueError) as exc:
            raise TypeError("site_species keys must be site indices") from exc
        if not isinstance(state_mapping, dict):
            raise TypeError(
                "site_species values must map occupation states to species strings"
            )

        normalized_state_mapping: dict[int, str] = {}
        for state_key, species in state_mapping.items():
            state_value = _normalize_state_value(state_key, "site_species state")
            if not isinstance(species, str) or not species.strip():
                raise ValueError("site_species species values must be non-empty strings")
            normalized_state_mapping[state_value] = species
        normalized[site_index] = normalized_state_mapping
    return normalized


def _site_species_as_dict(site_species: dict[int, dict[int, str]]) -> dict[str, dict[str, str]]:
    return {
        str(site_index): {
            str(state_value): species
            for state_value, species in state_mapping.items()
        }
        for site_index, state_mapping in site_species.items()
    }


class LocalBarrierModel(BaseModel):
    """
    Choose migration barriers from ordered local-environment rules.

    ``LocalBarrierModel`` stores an ordered list of :class:`BarrierRule` objects,
    built from plain rule dictionaries, and evaluates them against a ``State``
    and ``Event``. Each rule returns a dictionary of
    numeric properties; by default ``compute`` returns the ``barrier`` property.
    ``compute_probability`` then evaluates the Arrhenius rate when the current
    endpoint states match a mobile-vacancy hop.

    Parameters:
        rules: Ordered rule dictionaries or ``BarrierRule`` objects. The first
            matching rule is used.
        name: Human-readable model name.
        default_properties: Property dictionary used when no rule matches.
        default_barrier: Shortcut for ``default_properties={"barrier": value}``.
        default_property: Property returned by ``compute`` when
            ``property_name`` is not supplied.
        probability_mode: Rate calculation mode. Currently only
            ``"barrier_arrhenius"`` is supported.
        probability_property: Property used as the barrier in
            ``compute_probability``.
        site_species: Mapping used by ``species_count`` rules. The shape is
            ``{site_index: {occupation_state: species}}``. For example,
            ``{10: {0: "P", 1: "Si", 2: "Al"}}`` means site 10 is counted as
            P, Si, or Al depending on its current state index.

    Supported rule types:
        ``constant``
            Always matches and returns its properties. In most cases,
            ``default_barrier`` is clearer than an explicit constant rule.
        ``state_count``
            Counts how many selected sites have the requested state index.
            Binary aliases such as ``occupied``/``0`` and ``vacant``/``1`` are
            accepted for convenience.
        ``species_count``
            Counts species labels after applying ``site_species``.
        ``pattern``
            Matches selected occupations against a pattern containing
            nonnegative state indices, binary state aliases, or ``"*"``
            wildcards.
        ``exact``
            Matches a specific event and exact occupation vector. This is the
            direct replacement for catalog-style local-environment tables.

    Site selectors:
        Rules can use ``sites="local_env"``, ``"mobile_ion"``,
        ``"canonical"``, ``"from"``, ``"to"``, ``"all"``, or an explicit list
        of active-site indices. ``canonical`` means ``event.mobile_ion_indices``
        followed by ``event.local_env_indices`` with duplicates removed.

    Examples:
        Constant barrier::

            model = LocalBarrierModel.constant_barrier(300.0)

        At least three occupied local-environment sites::

            model = LocalBarrierModel(default_barrier=300.0)
            model.add_state_count_rule(
                name="crowded",
                sites="local_env",
                state="occupied",
                min_count=3,
                barrier=450.0,
            )

        More than three Si sites in the local environment::

            model = LocalBarrierModel(
                default_barrier=300.0,
                site_species={
                    1: {0: "P", 1: "Si"},
                    2: {0: "Si", 1: "P"},
                    3: {0: "Si", 1: "P"},
                    4: {0: "Al", 1: "Si"},
                },
            )
            model.add_species_count_rule(
                name="si_rich",
                sites="local_env",
                species="Si",
                min_count=4,
                barrier=420.0,
            )

        Exact event/local-environment match::

            model = LocalBarrierModel.from_exact_entries([
                {
                    "mobile_ion_indices": [0, 1],
                    "local_env_indices": [1, 2, 3],
                    "occupations": [1, 0, 1, 0],
                    "properties": {"barrier": 250.0},
                }
            ])
    """

    MODEL_TYPE = "local_barrier"
    PAYLOAD_KEY = "local_barrier"
    SUPPORTED_PROBABILITY_MODE = "barrier_arrhenius"
    BOLTZMANN_CONSTANT_MEV_PER_K = K_B_MEV_PER_K

    def __init__(
        self,
        rules: Optional[list[dict[str, Any]]] = None,
        name: str = "LocalBarrierModel",
        default_properties: Optional[dict[str, float]] = None,
        default_barrier: Optional[float] = None,
        default_property: str = "barrier",
        probability_mode: str = SUPPORTED_PROBABILITY_MODE,
        probability_property: str = "barrier",
        site_species: Optional[dict[Any, Any]] = None,
    ) -> None:
        super().__init__(name=name)
        self.name = name
        self.default_property = default_property
        self.probability_mode = probability_mode
        self.probability_property = probability_property
        self.site_species = _normalize_site_species(site_species)
        self.default_properties = (
            _normalize_properties(default_properties)
            if default_properties is not None
            else None
        )
        if default_barrier is not None:
            if isinstance(default_barrier, bool) or not isinstance(
                default_barrier, (int, float)
            ):
                raise TypeError("'default_barrier' must be numeric")
            if self.default_properties is None:
                self.default_properties = {}
            self.default_properties.setdefault("barrier", float(default_barrier))
        self.rules: list[BarrierRule] = []
        self._exact_rule_keys: set[
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
        ] = set()

        if rules is not None:
            self.build(rules=rules)

    def fit(self, *args, **kwargs):
        """Local barrier rules are defined explicitly, not fitted."""
        raise NotImplementedError(
            "LocalBarrierModel does not support fit(). Provide rules or default_properties instead."
        )

    @classmethod
    def constant_barrier(
        cls,
        barrier: float,
        name: str = "ConstantBarrierModel",
        **kwargs,
    ) -> "LocalBarrierModel":
        """Construct a model that returns the same barrier for every event.

        This is the simplest setup for smoke tests, toy simulations, or models
        where all event rates share one activation barrier. The returned model
        has no rules; the barrier is stored in ``default_properties``.
        """
        return cls(name=name, default_barrier=barrier, **kwargs)

    @classmethod
    def entry_from_event_state(
        cls,
        event: Event,
        state: State,
        properties: dict[str, float],
        name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build an exact-match rule from a runtime event and state snapshot.

        The occupation vector is sampled in canonical event-site order:
        ``mobile_ion_indices`` first, then ``local_env_indices`` with duplicate
        site indices removed. Use this helper when turning a known event/state
        snapshot into an exact rule without manually constructing the
        occupation list.
        """
        event_sites = EventSites.from_event(event)
        try:
            occupations = [
                int(state.occupations[site_index]) for site_index in event_sites.canonical
            ]
        except IndexError as exc:
            raise IndexError(
                "Event site index is out of range for provided simulation occupations"
            ) from exc
        entry = {
            "type": "exact",
            "mobile_ion_indices": list(event_sites.mobile_ion_indices),
            "local_env_indices": list(event_sites.local_env_indices),
            "occupations": occupations,
            "properties": dict(properties),
        }
        if name is not None:
            entry["name"] = name
        return entry

    @classmethod
    def from_exact_entries(
        cls,
        entries: list[dict[str, Any] | Any],
        name: str = "LocalBarrierModel",
        default_properties: Optional[dict[str, float]] = None,
        default_barrier: Optional[float] = None,
        default_property: str = "barrier",
        probability_mode: str = SUPPORTED_PROBABILITY_MODE,
        probability_property: str = "barrier",
        site_species: Optional[dict[Any, Any]] = None,
    ) -> "LocalBarrierModel":
        """Construct from exact event/local-occupation entries.

        Each entry must contain ``mobile_ion_indices``, ``local_env_indices``,
        ``occupations``, and ``properties``. The ``occupations`` list is in
        canonical site order: mobile-ion sites first, then local-environment
        sites with duplicates removed. Duplicate exact entries are rejected.
        """
        rules: list[dict[str, Any]] = []
        for index, entry in enumerate(entries):
            payload = entry.as_dict() if hasattr(entry, "as_dict") else dict(entry)
            payload["type"] = "exact"
            payload.setdefault("name", f"exact_{index}")
            rules.append(payload)
        return cls(
            rules=rules,
            name=name,
            default_properties=default_properties,
            default_barrier=default_barrier,
            default_property=default_property,
            probability_mode=probability_mode,
            probability_property=probability_property,
            site_species=site_species,
        )

    def _validate_probability_mode(self) -> None:
        if self.probability_mode != self.SUPPORTED_PROBABILITY_MODE:
            raise ValueError(
                f"Unsupported probability mode '{self.probability_mode}' for LocalBarrierModel"
            )

    def add_rule(self, rule: dict[str, Any] | BarrierRule) -> None:
        """Add one local barrier rule to the end of the ordered rule list."""
        if not isinstance(rule, BarrierRule):
            rule = BarrierRule.from_dict(rule, default_name=f"rule_{len(self.rules)}")
        if rule.type == "exact":
            if rule.exact_key in self._exact_rule_keys:
                raise ValueError(
                    "Duplicate exact local-barrier rule detected: "
                    f"mobile_ion_indices={rule.mobile_ion_indices}, "
                    f"canonical_sites={rule.canonical_site_indices}, "
                    f"occupations={rule.occupations}"
                )
            self._exact_rule_keys.add(rule.exact_key)
        self.rules.append(rule)

    def _add_typed_rule(
        self,
        rule_type: str,
        barrier: Optional[float],
        properties: Optional[dict[str, float]],
        name: Optional[str],
        **fields: Any,
    ) -> str:
        """Add a rule built by one of the ``add_*_rule`` helpers; return its name."""
        rule = {
            "type": rule_type,
            "properties": dict(properties or {}),
            **{key: value for key, value in fields.items() if value is not None},
        }
        if barrier is not None:
            rule["barrier"] = barrier
        if name is not None:
            rule["name"] = name
        self.add_rule(rule)
        return self.rules[-1].name

    def add_exact_rule(
        self,
        mobile_ion_indices: list[int] | tuple[int, ...],
        local_env_indices: list[int] | tuple[int, ...],
        occupations: list[int] | tuple[int, ...],
        barrier: Optional[float] = None,
        properties: Optional[dict[str, float]] = None,
        name: Optional[str] = None,
    ) -> str:
        """Add an event-specific exact occupation rule.

        ``occupations`` must follow canonical site order for the supplied
        ``mobile_ion_indices`` and ``local_env_indices``. Use this rule type
        when the barrier is known only for one exact event/environment pattern.
        """
        return self._add_typed_rule(
            "exact",
            barrier,
            properties,
            name,
            mobile_ion_indices=list(mobile_ion_indices),
            local_env_indices=list(local_env_indices),
            occupations=list(occupations),
        )

    def add_state_count_rule(
        self,
        state: str | int,
        barrier: Optional[float] = None,
        properties: Optional[dict[str, float]] = None,
        name: Optional[str] = None,
        sites: str | list[int] | tuple[int, ...] = "local_env",
        count: Optional[int] = None,
        min_count: Optional[int] = None,
        max_count: Optional[int] = None,
    ) -> str:
        """Add a rule based on the number of sites in an occupation state.

        ``state`` accepts any nonnegative integer state index. The binary
        aliases ``"occupied"``/``0`` and ``"vacant"``/``1`` are also accepted.
        Supply exactly one of ``count`` or a ``min_count``/``max_count`` range.
        """
        return self._add_typed_rule(
            "state_count",
            barrier,
            properties,
            name,
            sites=sites,
            state=state,
            count=count,
            min_count=min_count,
            max_count=max_count,
        )

    def add_species_count_rule(
        self,
        species: str | list[str] | tuple[str, ...],
        barrier: Optional[float] = None,
        properties: Optional[dict[str, float]] = None,
        name: Optional[str] = None,
        sites: str | list[int] | tuple[int, ...] = "local_env",
        count: Optional[int] = None,
        min_count: Optional[int] = None,
        max_count: Optional[int] = None,
    ) -> str:
        """Add a rule based on the number of sites currently carrying a species.

        Species labels are looked up from ``site_species`` using each selected
        site index and current occupation value. This is appropriate for rules
        such as "use a higher barrier when at least four selected sites are Si".
        """
        return self._add_typed_rule(
            "species_count",
            barrier,
            properties,
            name,
            sites=sites,
            species=species,
            count=count,
            min_count=min_count,
            max_count=max_count,
        )

    def add_pattern_rule(
        self,
        pattern: list[int | str] | tuple[int | str, ...],
        barrier: Optional[float] = None,
        properties: Optional[dict[str, float]] = None,
        name: Optional[str] = None,
        sites: str | list[int] | tuple[int, ...] = "canonical",
    ) -> str:
        """Add a wildcard occupation pattern rule.

        Patterns can contain nonnegative integer state indices, binary state
        aliases, or ``"*"`` wildcards. The pattern length must match the number
        of selected sites.
        """
        return self._add_typed_rule(
            "pattern", barrier, properties, name, sites=sites, pattern=list(pattern)
        )

    def build(
        self,
        rules: Optional[list[dict[str, Any]]] = None,
        default_properties: Optional[dict[str, float]] = None,
        default_barrier: Optional[float] = None,
        default_property: Optional[str] = None,
        probability_mode: Optional[str] = None,
        probability_property: Optional[str] = None,
        site_species: Optional[dict[Any, Any]] = None,
    ) -> None:
        """Replace this model's rule table."""
        if default_properties is not None:
            self.default_properties = _normalize_properties(default_properties)
        if default_barrier is not None:
            if self.default_properties is None:
                self.default_properties = {}
            self.default_properties.setdefault("barrier", float(default_barrier))
        if default_property is not None:
            self.default_property = default_property
        if probability_mode is not None:
            self.probability_mode = probability_mode
        if probability_property is not None:
            self.probability_property = probability_property
        if site_species is not None:
            self.site_species = _normalize_site_species(site_species)

        self._validate_probability_mode()

        if rules is None:
            rules = []
        if not isinstance(rules, list):
            raise TypeError("'rules' must be a list")
        self.rules = []
        self._exact_rule_keys = set()
        for rule in rules:
            self.add_rule(rule)

    def _species_for_site(self, site_index: int, occupation: int) -> str:
        if site_index not in self.site_species:
            raise ValueError(
                "species_count rules require site_species for every counted site; "
                f"missing site {site_index}"
            )
        state_mapping = self.site_species[site_index]
        if occupation not in state_mapping:
            raise ValueError(
                "species_count rules require site_species entries for all "
                f"occupation states; missing site {site_index}, state {occupation}"
            )
        return state_mapping[occupation]

    def _matched_properties(self, simulation_state: State, event: Event) -> dict[str, float]:
        if simulation_state is None:
            raise ValueError("simulation_state is required")
        if event is None:
            raise ValueError("event is required")

        event_sites = EventSites.from_event(event)
        occupations = simulation_state.occupations
        for rule in self.rules:
            if rule.matches(event_sites, occupations, self._species_for_site):
                return rule.properties

        if self.default_properties is not None:
            return self.default_properties

        raise KeyError(
            "No local barrier rule matched and no default_properties were provided: "
            f"mobile_ion_indices={tuple(event.mobile_ion_indices)}, "
            f"local_env_indices={tuple(event.local_env_indices)}, "
            f"canonical_sites={event_sites.canonical}, "
            f"occupations={_occupations_at(occupations, event_sites.canonical)}"
        )

    def compute(
        self,
        simulation_state: State,
        event: Event,
        property_name: Optional[str] = None,
    ) -> float:
        """Compute a barrier/property value by local rule matching."""
        properties = self._matched_properties(
            simulation_state=simulation_state, event=event
        )
        selected_property = property_name or self.default_property
        if selected_property not in properties:
            raise KeyError(
                f"Property '{selected_property}' not found in matched local barrier rule"
            )
        return float(properties[selected_property])

    def compute_probability(
        self,
        event: Event,
        runtime_config: "RuntimeConfig",
        simulation_state: State,
    ) -> float:
        """Compute event rate in Hz from a selected meV barrier."""
        self._validate_probability_mode()
        barrier = self.compute(
            simulation_state=simulation_state,
            event=event,
            property_name=self.probability_property,
        )
        occupations = simulation_state.occupations
        hop_factor = 1.0 if event_direction(occupations, event) != 0 else 0.0
        temperature = runtime_config.temperature
        attempt_frequency = runtime_config.attempt_frequency

        rate = hop_factor * attempt_frequency * np.exp(
            -barrier / (self.BOLTZMANN_CONSTANT_MEV_PER_K * temperature)
        )
        return float(rate)

    def __str__(self) -> str:
        return (
            f"LocalBarrierModel(name={self.name}, rules={len(self.rules)}, "
            f"default_property={self.default_property}, "
            f"probability_property={self.probability_property})"
        )

    def __repr__(self) -> str:
        return (
            "LocalBarrierModel("
            f"name={self.name!r}, rules={len(self.rules)}, "
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
            "default_property": self.default_property,
            "probability_mode": self.probability_mode,
            "probability_property": self.probability_property,
            "default_properties": (
                dict(self.default_properties)
                if self.default_properties is not None
                else None
            ),
            "site_species": _site_species_as_dict(self.site_species),
            "rules": [rule.as_dict() for rule in self.rules],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LocalBarrierModel":
        """Deserialize from in-memory payload."""
        if not isinstance(data, dict):
            raise ValueError("LocalBarrierModel payload must be a JSON object")
        data = cls._unwrap_model_file(data)

        return cls(
            rules=data.get("rules", []),
            name=data.get("name", "LocalBarrierModel"),
            default_properties=data.get("default_properties"),
            default_barrier=data.get("default_barrier"),
            default_property=data.get("default_property", "barrier"),
            probability_mode=data.get(
                "probability_mode", cls.SUPPORTED_PROBABILITY_MODE
            ),
            probability_property=data.get("probability_property", "barrier"),
            site_species=data.get("site_species"),
        )
