"""Mapping from kMCpy active sites and states to an external occupation array.

External site-energy codes (smol, CLEASE, ASE, project code) keep their own
site order and occupation encoding. :class:`ExternalSiteMapping` translates a
kMCpy active-site index and state index into an external site index and
occupation value, using dense lookup arrays built once by :meth:`prepare`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_MISSING_STATE_VALUE = object()


@dataclass(frozen=True)
class _StateLookup:
    """Dense lookup array for kMCpy state index to external occupation value."""

    values: np.ndarray
    offset: int

    @classmethod
    def from_mapping(cls, state_mapping: dict[int, Any]) -> "_StateLookup":
        if not state_mapping:
            return cls(values=np.empty(0, dtype=object), offset=0)
        keys = [int(key) for key in state_mapping]
        min_state = min(keys)
        values = np.empty(max(keys) - min_state + 1, dtype=object)
        values.fill(_MISSING_STATE_VALUE)
        for state, external_value in state_mapping.items():
            values[int(state) - min_state] = external_value
        return cls(values=values, offset=min_state)

    def value(self, kmcpy_site: int, state_value: int) -> Any:
        array_index = int(state_value) - self.offset
        value = (
            self.values[array_index]
            if 0 <= array_index < len(self.values)
            else _MISSING_STATE_VALUE
        )
        if value is _MISSING_STATE_VALUE:
            raise ValueError(
                f"No external state mapping is defined for kMCpy site {kmcpy_site}, "
                f"state {state_value}"
            )
        return value


@dataclass
class ExternalSiteMapping:
    """How kMCpy active sites and state indices map onto an external occupation.

    Parameters:
        site_mapping: ``{kmcpy_site: external_site}`` or a sequence indexed by
            kMCpy site. ``None`` maps each active site to the same index.
        state_mapping: ``{state_index: external_value}`` (or a sequence) used for
            every site. ``None`` passes state indices through unchanged.
        state_mapping_by_site: Per-site state mappings that override
            ``state_mapping`` for the listed sites.
        initial_occupation: External occupation array to start from; sites not
            mapped from kMCpy keep these values.
        external_size: Length of the external occupation when
            ``initial_occupation`` is not given.
        external_fill_value: Value for unmapped external sites.
        external_dtype: NumPy dtype of the external occupation; inferred from
            the mapped values when omitted.
    """

    site_mapping: dict[int, int] | None = None
    state_mapping: dict[int, Any] | None = None
    state_mapping_by_site: dict[int, dict[int, Any]] | None = None
    initial_occupation: np.ndarray | None = None
    external_size: int | None = None
    external_fill_value: Any = 0
    external_dtype: str | None = None

    # Lookup arrays built by prepare().
    site_lookup: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    state_lookup: _StateLookup | None = field(default=None, init=False, repr=False, compare=False)
    state_lookup_by_site: tuple[_StateLookup | None, ...] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.site_mapping = self._int_mapping(self.site_mapping, int)
        self.state_mapping = self._int_mapping(self.state_mapping, lambda value: value)
        self.state_mapping_by_site = self._state_mapping_by_site(self.state_mapping_by_site)
        if self.initial_occupation is not None:
            self.initial_occupation = np.asarray(self.initial_occupation).copy()
        if self.external_size is not None:
            self.external_size = int(self.external_size)

    PAYLOAD_KEYS = (
        "site_mapping",
        "state_mapping",
        "state_mapping_by_site",
        "initial_occupation",
        "external_size",
        "external_fill_value",
        "external_dtype",
    )

    def as_dict(self) -> dict[str, Any]:
        """Return the mapping fields in model-payload form."""
        initial = self.initial_occupation
        return {
            "site_mapping": self.site_mapping,
            "state_mapping": self.state_mapping,
            "state_mapping_by_site": self.state_mapping_by_site,
            "initial_occupation": initial.tolist() if initial is not None else None,
            "external_size": self.external_size,
            "external_fill_value": self.external_fill_value,
            "external_dtype": self.external_dtype,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExternalSiteMapping":
        return cls(
            site_mapping=data.get("site_mapping"),
            state_mapping=data.get("state_mapping"),
            state_mapping_by_site=data.get("state_mapping_by_site"),
            initial_occupation=data.get("initial_occupation"),
            external_size=data.get("external_size"),
            external_fill_value=data.get("external_fill_value", 0),
            external_dtype=data.get("external_dtype"),
        )

    @property
    def order_hash(self) -> str:
        """Order-sensitive hash of the active-site to external-site mapping."""
        payload = {
            "format": "kmcpy.site_energy.external_site_order.v1",
            "site_mapping": (
                [
                    [int(site), int(external_site)]
                    for site, external_site in sorted(self.site_mapping.items())
                ]
                if self.site_mapping is not None
                else None
            ),
            "external_size": self.external_size,
            "initial_occupation_length": (
                int(len(self.initial_occupation))
                if self.initial_occupation is not None
                else None
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def is_prepared(self) -> bool:
        return self.site_lookup is not None

    def prepare(self, n_sites: int) -> None:
        """Build and validate the lookup arrays for ``n_sites`` active sites."""
        n_sites = int(n_sites)
        if self.site_mapping is None:
            lookup = np.arange(n_sites, dtype=np.int64)
        else:
            missing = [site for site in range(n_sites) if site not in self.site_mapping]
            if missing:
                raise ValueError(
                    "site_mapping does not cover all kMCpy active sites; "
                    f"missing first site {missing[0]}"
                )
            lookup = np.array([self.site_mapping[site] for site in range(n_sites)])

        if np.any(lookup < 0):
            raise ValueError("External site indices must be nonnegative")
        upper_bound = self.external_size
        if self.initial_occupation is not None:
            upper_bound = len(self.initial_occupation)
        if upper_bound is not None and len(lookup) and int(np.max(lookup)) >= upper_bound:
            raise ValueError(
                "site_mapping points outside the external occupation length "
                f"({int(np.max(lookup))} >= {upper_bound})"
            )

        self.site_lookup = lookup
        self.state_lookup = (
            _StateLookup.from_mapping(self.state_mapping)
            if self.state_mapping is not None
            else None
        )
        self.state_lookup_by_site = (
            tuple(
                _StateLookup.from_mapping(self.state_mapping_by_site[site])
                if site in self.state_mapping_by_site
                else self.state_lookup
                for site in range(n_sites)
            )
            if self.state_mapping_by_site is not None
            else None
        )

    def external_site(self, kmcpy_site: int) -> int:
        if self.site_lookup is None:
            raise RuntimeError("SiteEnergyModel has not been initialized")
        return int(self.site_lookup[int(kmcpy_site)])

    def external_value(self, kmcpy_site: int, state_value: int) -> Any:
        lookup = self.state_lookup
        if self.state_lookup_by_site is not None:
            try:
                lookup = self.state_lookup_by_site[int(kmcpy_site)]
            except IndexError as exc:
                raise ValueError(
                    f"kMCpy site {kmcpy_site} is outside the state lookup range"
                ) from exc
        if lookup is None:
            return int(state_value)
        return lookup.value(int(kmcpy_site), int(state_value))

    def build_external_occupation(self, occupations: Sequence[int]) -> np.ndarray:
        """Return the external occupation array for kMCpy ``occupations``."""
        mapped_values = [
            self.external_value(site, int(state))
            for site, state in enumerate(occupations)
        ]
        dtype = self._external_dtype(mapped_values)
        if self.initial_occupation is not None:
            external = np.asarray(self.initial_occupation, dtype=dtype).copy()
        else:
            if self.external_size is not None:
                size = self.external_size
            elif self.site_lookup is None or len(self.site_lookup) == 0:
                size = len(occupations)
            else:
                size = int(np.max(self.site_lookup)) + 1
            external = np.full(size, self.external_fill_value, dtype=dtype)

        for kmcpy_site, mapped_value in enumerate(mapped_values):
            external[int(self.site_lookup[kmcpy_site])] = mapped_value
        return external

    def _external_dtype(self, mapped_values: Sequence[Any]):
        if self.external_dtype is not None:
            return np.dtype(self.external_dtype)
        if self.initial_occupation is not None:
            dtype = self.initial_occupation.dtype
            if dtype.kind in "US":
                # Fixed-width strings: widen so that every mapped state value
                # fits, including values that only appear after later hops.
                text_values = [
                    value
                    for value in (*mapped_values, *self._all_state_values())
                    if isinstance(value, (str, bytes))
                ]
                if text_values:
                    dtype = np.result_type(dtype, np.asarray(text_values).dtype)
            return dtype
        values = list(mapped_values)
        if values and all(isinstance(value, (int, np.integer)) for value in values):
            return np.int64
        if values and all(isinstance(value, (int, float, np.number)) for value in values):
            return float
        if isinstance(self.external_fill_value, (int, np.integer)) and not values:
            return np.int64
        return object

    def _all_state_values(self) -> list[Any]:
        values = list((self.state_mapping or {}).values())
        for site_mapping in (self.state_mapping_by_site or {}).values():
            values.extend(site_mapping.values())
        return values

    @staticmethod
    def _int_mapping(mapping, convert_value) -> dict[int, Any] | None:
        if mapping is None:
            return None
        if isinstance(mapping, Mapping):
            return {int(key): convert_value(value) for key, value in mapping.items()}
        return {index: convert_value(value) for index, value in enumerate(mapping)}

    @classmethod
    def _state_mapping_by_site(cls, mapping) -> dict[int, dict[int, Any]] | None:
        if mapping is None:
            return None
        if isinstance(mapping, Mapping):
            return {
                int(site): cls._int_mapping(site_mapping, lambda value: value) or {}
                for site, site_mapping in mapping.items()
            }
        return {
            site: cls._int_mapping(site_mapping, lambda value: value) or {}
            for site, site_mapping in enumerate(mapping)
            if site_mapping is not None
        }
