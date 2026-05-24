"""Active-site index mapping utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from pymatgen.core import Species, Structure

from kmcpy.structure.vacancy import Vacancy


INDEX_MAP_FORMAT = "kmcpy.active_site_index_map.v1"
ORIGINAL_SITE_PROPERTY = "_kmcpy_original_site_index"
PRIMITIVE_SITE_PROPERTY = "_kmcpy_primitive_site_index"
PRIMITIVE_ACTIVE_SITE_PROPERTY = "_kmcpy_primitive_active_site_index"
ACTIVE_SITE_PROPERTY = "_kmcpy_active_site_index"


@dataclass(frozen=True)
class ActiveSiteIndexMap:
    """Map full template sites to compact mutable active-site indices."""

    primitive_site_count: int
    original_site_count: int
    primitive_active_indices: tuple[int, ...]
    active_to_original: tuple[int, ...]
    active_to_primitive: tuple[int, ...]
    fixed_original_indices: tuple[int, ...]
    supercell_shape: tuple[int, int, int]
    species_by_primitive_site: tuple[str, ...]
    allowed_species_by_primitive_site: tuple[tuple[str, ...], ...]
    fingerprint: str
    template_structure: Structure | None = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def from_lattice_structure(
        cls,
        lattice_structure,
        supercell_shape: Sequence[int] | None = None,
    ) -> "ActiveSiteIndexMap":
        """Build from a ``LatticeStructure`` instance."""
        return cls.from_structure_and_mapping(
            lattice_structure.template_structure,
            lattice_structure.site_mapping,
            supercell_shape=supercell_shape,
        )

    @classmethod
    def from_structure_and_mapping(
        cls,
        template_structure: Structure,
        site_mapping: Mapping[Any, Any],
        supercell_shape: Sequence[int] | None = None,
    ) -> "ActiveSiteIndexMap":
        """Build an active-site map from a full template and site mapping."""
        shape = _normalize_supercell_shape(supercell_shape)
        allowed_species = _allowed_species_by_site(
            template_structure, site_mapping
        )
        primitive_active_indices = tuple(
            index
            for index, allowed in enumerate(allowed_species)
            if len(allowed) > 1
        )
        if not primitive_active_indices:
            raise ValueError(
                "site_mapping does not define any mutable active sites"
            )

        primitive_active_lookup = {
            original_index: active_index
            for active_index, original_index in enumerate(primitive_active_indices)
        }
        full_structure = _make_supercell_with_properties(
            template_structure=template_structure,
            primitive_active_lookup=primitive_active_lookup,
            supercell_shape=shape,
        )

        active_to_original = []
        active_to_primitive = []
        fixed_original_indices = []
        for full_index, site in enumerate(full_structure):
            primitive_index = int(site.properties[PRIMITIVE_SITE_PROPERTY])
            if primitive_index in primitive_active_lookup:
                active_to_original.append(full_index)
                active_to_primitive.append(primitive_index)
            else:
                fixed_original_indices.append(full_index)

        species_by_primitive_site = tuple(
            _species_label(site.specie) for site in template_structure
        )
        allowed_species_by_primitive_site = tuple(
            tuple(_species_label(specie) for specie in allowed)
            for allowed in allowed_species
        )
        fingerprint = _fingerprint(
            {
                "format": INDEX_MAP_FORMAT,
                "primitive_site_count": len(template_structure),
                "original_site_count": len(full_structure),
                "primitive_active_indices": primitive_active_indices,
                "active_to_original": active_to_original,
                "active_to_primitive": active_to_primitive,
                "supercell_shape": shape,
                "species_by_primitive_site": species_by_primitive_site,
                "allowed_species_by_primitive_site": (
                    allowed_species_by_primitive_site
                ),
            }
        )
        return cls(
            primitive_site_count=len(template_structure),
            original_site_count=len(full_structure),
            primitive_active_indices=primitive_active_indices,
            active_to_original=tuple(active_to_original),
            active_to_primitive=tuple(active_to_primitive),
            fixed_original_indices=tuple(fixed_original_indices),
            supercell_shape=shape,
            species_by_primitive_site=species_by_primitive_site,
            allowed_species_by_primitive_site=allowed_species_by_primitive_site,
            fingerprint=fingerprint,
            template_structure=template_structure.copy(),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActiveSiteIndexMap":
        """Restore serialized index metadata."""
        if data.get("format") != INDEX_MAP_FORMAT:
            raise ValueError(
                f"Unsupported active-site index map format: {data.get('format')}"
            )
        return cls(
            primitive_site_count=int(data["primitive_site_count"]),
            original_site_count=int(data["original_site_count"]),
            primitive_active_indices=tuple(
                int(index) for index in data["primitive_active_indices"]
            ),
            active_to_original=tuple(
                int(index) for index in data["active_to_original"]
            ),
            active_to_primitive=tuple(
                int(index) for index in data["active_to_primitive"]
            ),
            fixed_original_indices=tuple(
                int(index) for index in data.get("fixed_original_indices", ())
            ),
            supercell_shape=tuple(int(value) for value in data["supercell_shape"]),
            species_by_primitive_site=tuple(data["species_by_primitive_site"]),
            allowed_species_by_primitive_site=tuple(
                tuple(site_species) for site_species in data[
                    "allowed_species_by_primitive_site"
                ]
            ),
            fingerprint=str(data["fingerprint"]),
            template_structure=None,
        )

    @property
    def active_site_count(self) -> int:
        return len(self.active_to_original)

    @property
    def original_to_active(self) -> dict[int, int]:
        return {
            int(original_index): active_index
            for active_index, original_index in enumerate(self.active_to_original)
        }

    @property
    def primitive_to_active(self) -> dict[int, int]:
        return {
            int(original_index): active_index
            for active_index, original_index in enumerate(self.primitive_active_indices)
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize map metadata without storing the full structure."""
        return {
            "format": INDEX_MAP_FORMAT,
            "primitive_site_count": self.primitive_site_count,
            "original_site_count": self.original_site_count,
            "primitive_active_indices": list(self.primitive_active_indices),
            "active_to_original": list(self.active_to_original),
            "active_to_primitive": list(self.active_to_primitive),
            "fixed_original_indices": list(self.fixed_original_indices),
            "supercell_shape": list(self.supercell_shape),
            "species_by_primitive_site": list(self.species_by_primitive_site),
            "allowed_species_by_primitive_site": [
                list(species) for species in self.allowed_species_by_primitive_site
            ],
            "fingerprint": self.fingerprint,
        }

    def assert_compatible(self, other: "ActiveSiteIndexMap | Mapping[str, Any]") -> None:
        """Raise if another map or metadata payload describes a different site space."""
        other_map = (
            ActiveSiteIndexMap.from_dict(other)
            if isinstance(other, Mapping)
            else other
        )
        if self.fingerprint != other_map.fingerprint:
            raise ValueError(
                "Active-site index metadata does not match the current "
                "site_mapping and structure."
            )

    def validate_active_indices(
        self,
        indices: Sequence[int],
        field_name: str = "indices",
    ) -> None:
        """Validate compact active-site indices."""
        invalid = [
            int(index)
            for index in indices
            if int(index) < 0 or int(index) >= self.active_site_count
        ]
        if invalid:
            raise IndexError(
                f"{field_name} contains indices outside the active-site range: {invalid}"
            )

    def select_active_values(self, values: Sequence[Any]) -> list[Any]:
        """Return compact active-site values from active or full-supercell input."""
        values = list(values)
        if len(values) == self.active_site_count:
            return values
        if len(values) == self.original_site_count:
            return [values[index] for index in self.active_to_original]
        raise ValueError(
            "Occupation length must match either the active-site count "
            f"({self.active_site_count}) or full site count ({self.original_site_count}); "
            f"got {len(values)}."
        )

    def full_structure_with_properties(self) -> Structure:
        """Return the full supercell with index-space site properties."""
        if self.template_structure is None:
            raise ValueError("Cannot build structures from serialized index metadata only")
        primitive_active_lookup = self.primitive_to_active
        return _make_supercell_with_properties(
            template_structure=self.template_structure,
            primitive_active_lookup=primitive_active_lookup,
            supercell_shape=self.supercell_shape,
        )

    def active_structure(self) -> Structure:
        """Return active sites only, ordered by compact active index."""
        full_structure = self.full_structure_with_properties()
        active_sites = [
            full_structure[original_index]
            for original_index in self.active_to_original
        ]
        active_structure = type(full_structure).from_sites(active_sites)
        active_structure.add_site_property(
            ACTIVE_SITE_PROPERTY, list(range(self.active_site_count))
        )
        return active_structure

    def filter_active_structure(
        self,
        structure: Structure,
        tol: float = 1e-2,
    ) -> Structure:
        """Filter a possibly full structure down to active sites by index metadata or position."""
        if (
            len(structure) == self.active_site_count
            and ACTIVE_SITE_PROPERTY in structure.site_properties
        ):
            return structure.copy()

        if len(structure) == self.original_site_count:
            active_sites = [structure[index] for index in self.active_to_original]
            active_structure = type(structure).from_sites(active_sites)
            active_structure.add_site_property(
                ACTIVE_SITE_PROPERTY, list(range(self.active_site_count))
            )
            return active_structure

        full_reference = self.full_structure_with_properties()
        original_to_active = self.original_to_active
        kept_by_active_index = {}
        for site in structure:
            distances = full_reference.lattice.get_all_distances(
                np.array([site.frac_coords]),
                full_reference.frac_coords,
            )[0]
            original_index = int(np.argmin(distances))
            if float(distances[original_index]) > tol:
                raise ValueError(
                    "Input structure contains a site that cannot be mapped to "
                    "the active-site template."
                )
            active_index = original_to_active.get(original_index)
            if active_index is None:
                continue
            if active_index in kept_by_active_index:
                raise ValueError(
                    "Input structure maps multiple sites to the same active site"
                )
            kept_by_active_index[active_index] = site

        active_sites = [
            kept_by_active_index[index]
            for index in sorted(kept_by_active_index)
        ]
        active_structure = type(structure).from_sites(active_sites)
        active_structure.add_site_property(
            ACTIVE_SITE_PROPERTY, sorted(kept_by_active_index)
        )
        return active_structure


def _normalize_supercell_shape(
    supercell_shape: Sequence[int] | None,
) -> tuple[int, int, int]:
    if supercell_shape is None:
        return (1, 1, 1)
    shape = tuple(int(value) for value in supercell_shape)
    if len(shape) != 3:
        raise ValueError("supercell_shape must contain three integers")
    if any(value <= 0 for value in shape):
        raise ValueError("supercell_shape values must be positive")
    return shape


def _make_supercell_with_properties(
    template_structure: Structure,
    primitive_active_lookup: Mapping[int, int],
    supercell_shape: tuple[int, int, int],
) -> Structure:
    from kmcpy.external.structure import StructureKMCpy

    base = template_structure.copy()
    primitive_indices = list(range(len(base)))
    base.add_site_property(PRIMITIVE_SITE_PROPERTY, primitive_indices)
    base.add_site_property(
        PRIMITIVE_ACTIVE_SITE_PROPERTY,
        [primitive_active_lookup.get(index, -1) for index in primitive_indices],
    )
    supercell = StructureKMCpy.from_sites(base.sites).make_kmc_supercell(
        supercell_shape
    )
    supercell.add_site_property(
        ORIGINAL_SITE_PROPERTY, list(range(len(supercell)))
    )
    active_indices = []
    for site in supercell:
        primitive_index = int(site.properties[PRIMITIVE_SITE_PROPERTY])
        active_indices.append(primitive_active_lookup.get(primitive_index, -1))
    supercell.add_site_property(ACTIVE_SITE_PROPERTY, active_indices)
    return supercell


def _allowed_species_by_site(
    template_structure: Structure,
    site_mapping: Mapping[Any, Any],
) -> list[tuple[Any, ...]]:
    entries = [
        (_normalize_species(key), _normalize_allowed_species(value))
        for key, value in site_mapping.items()
    ]
    allowed_species = []
    for index, site in enumerate(template_structure):
        matches = [
            allowed
            for key_species, allowed in entries
            if _species_equivalent(site.specie, key_species)
        ]
        if not matches:
            raise ValueError(
                "No site_mapping entry found for template site "
                f"{index} with species {site.species_string}."
            )
        allowed_species.append(matches[0])
    return allowed_species


def _normalize_allowed_species(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_species(item) for item in value)
    return (_normalize_species(value),)


def _normalize_species(value: Any) -> Any:
    if isinstance(value, Vacancy):
        return value
    if isinstance(value, str):
        if value == "X":
            return Vacancy()
        return Species(value)
    return value


def _species_equivalent(left: Any, right: Any) -> bool:
    return bool(_species_tokens(left).intersection(_species_tokens(right)))


def _species_tokens(specie: Any) -> set[str]:
    if isinstance(specie, Vacancy) or specie == "X":
        return {"X", "Vacancy"}
    tokens = {str(specie)}
    symbol = getattr(specie, "symbol", None)
    if symbol is not None:
        tokens.add(str(symbol))
    element = getattr(specie, "element", None)
    if element is not None:
        tokens.add(str(element))
    return tokens


def _species_label(specie: Any) -> str:
    if isinstance(specie, Vacancy) or specie == "X":
        return "X"
    symbol = getattr(specie, "symbol", None)
    if symbol is not None:
        return str(symbol)
    return str(specie)


def _fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
