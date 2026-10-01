"""Enumerate local site states and NEB endpoints.

:class:`LocalEnvironmentEnumerator` holds the active-site view of one
``LatticeStructure`` and implements the enumeration steps. The module-level
functions ``enumerate_local_environments``, ``generate_neb_endpoint_pair``, and
``enumerate_neb_endpoint_pairs`` are convenience wrappers around it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from pymatgen.core import PeriodicSite, Structure

from kmcpy.structure.basis import Occupation
from kmcpy.structure.lattice_structure import LatticeStructure
from kmcpy.structure.local_lattice_structure import resolve_center_site
from kmcpy.structure.local_site_order import LocalSiteOrder
from kmcpy.structure.species import (
    is_vacancy_species,
    species_equivalent,
    species_label,
    species_tokens,
)


@dataclass(frozen=True)
class LocalEnvironmentEnumeration:
    """One ordered local environment assignment."""

    structure: Structure
    full_occupation: Occupation
    local_occupation: Occupation
    local_site_indices: tuple[int, ...]
    variable_site_indices: tuple[int, ...]
    species_by_site: dict[int, str]
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NEBEndpointPair:
    """Initial and final structures for one mobile-ion hop."""

    initial: Structure
    final: Structure
    initial_occupation: Occupation
    final_occupation: Occupation
    mobile_ion_indices: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)


class LocalEnvironmentEnumerator:
    """Enumerate local environments and NEB endpoints for one lattice structure.

    All site indices are active-site indices of ``lattice_structure``; the
    active lattice and active-site order are built once per enumerator.
    """

    def __init__(self, lattice_structure: LatticeStructure):
        self.lattice_structure = lattice_structure
        self.active_site_order = lattice_structure.get_active_site_order()
        self.active_lattice_structure = lattice_structure.get_active_lattice_structure()

    def enumerate(
        self,
        center,
        cutoff: float,
        species_counts: Mapping[Any, int] | None = None,
        variable_species: Sequence[Any] | None = None,
        variable_site_indices: Sequence[int] | None = None,
        local_site_order=None,
        exclude_center_site=None,
        base_structure: Structure | None = None,
        transformation: Any | None = None,
        return_ranked_list: bool | int = True,
        max_results: int = 10000,
        tol: float = 0.1,
        angle_tol: float = 5,
    ) -> list[LocalEnvironmentEnumeration]:
        """Enumerate local site assignments; see :func:`enumerate_local_environments`."""
        if max_results < 1:
            raise ValueError("max_results must be at least 1")

        local_site_indices = self.local_site_indices(
            center, cutoff, local_site_order, exclude_center_site
        )
        base_occupation = self._base_occupation(base_structure, tol, angle_tol)
        variable_sites = self._resolve_variable_site_indices(
            local_site_indices, variable_site_indices, variable_species
        )
        exact_counts = _canonical_counts(species_counts)
        if exact_counts is not None and sum(exact_counts.values()) != len(variable_sites):
            raise ValueError(
                "species_counts must sum to the number of variable sites for exact enumeration"
            )

        if transformation is not None:
            return self._enumerate_with_transformation(
                base_occupation=base_occupation,
                local_site_indices=local_site_indices,
                variable_site_indices=variable_sites,
                variable_species=variable_species,
                species_counts=exact_counts,
                transformation=transformation,
                return_ranked_list=return_ranked_list,
                max_results=max_results,
                tol=tol,
                angle_tol=angle_tol,
            )

        lattice = self.active_lattice_structure
        choices_by_site = {
            site_index: self._allowed_choices(site_index, variable_species)
            for site_index in variable_sites
        }
        results: list[LocalEnvironmentEnumeration] = []
        for choices in product(*(choices_by_site[index] for index in variable_sites)):
            species_by_site = {
                site_index: species_label(specie)
                for site_index, specie in zip(variable_sites, choices)
            }
            if not _matches_counts(species_by_site, exact_counts):
                continue

            occupation = base_occupation.copy()
            for site_index, specie in zip(variable_sites, choices):
                occupation[site_index] = lattice.occupation_value_for_species(
                    site_index,
                    specie,
                )
            results.append(
                self._build_enumeration(
                    occupation=occupation,
                    local_site_indices=local_site_indices,
                    variable_site_indices=variable_sites,
                    species_by_site=species_by_site,
                    metadata={"source": "cartesian_product"},
                )
            )
            if len(results) >= max_results:
                break

        return results

    def endpoint_pair(
        self,
        local_environment_enumeration: LocalEnvironmentEnumeration | Occupation | Sequence[int],
        mobile_ion_indices: Any | Sequence[int],
    ) -> NEBEndpointPair:
        """Build ordered initial/final structures; see :func:`generate_neb_endpoint_pair`."""
        lattice = self.active_lattice_structure
        from_site, to_site = _mobile_ion_indices(mobile_ion_indices)
        self.active_site_order.validate_active_indices(
            (from_site, to_site), field_name="mobile_ion_indices"
        )
        if from_site == to_site:
            raise ValueError("mobile_ion_indices must contain two distinct sites")

        initial_occupation = self._occupation_from(local_environment_enumeration)
        if len(initial_occupation) != self.active_site_order.active_site_count:
            raise ValueError("environment occupation length does not match active sites")

        from_species = self._first_allowed_species(from_site)
        to_species = self._first_allowed_species(to_site)
        if is_vacancy_species(from_species) or is_vacancy_species(to_species):
            raise ValueError("mobile-ion sites must use a real species as the first mapping")
        if not species_equivalent(from_species, to_species):
            raise ValueError("hop sites must have the same first allowed mobile species")

        initial_occupation = initial_occupation.copy()
        initial_occupation[from_site] = lattice.basis.match_value
        initial_occupation[to_site] = lattice.basis.mismatch_value
        final_occupation = initial_occupation.flip([from_site, to_site])

        initial, final = self._ordered_endpoint_structures(
            initial_occupation, final_occupation, from_site, to_site
        )
        return NEBEndpointPair(
            initial=initial,
            final=final,
            initial_occupation=initial_occupation,
            final_occupation=final_occupation,
            mobile_ion_indices=(from_site, to_site),
            metadata={"mobile_ion_indices": (from_site, to_site)},
        )

    def endpoint_pairs(
        self,
        mobile_ion_indices: Any | Sequence[int],
        cutoff: float,
        center=None,
        **enumerate_kwargs: Any,
    ) -> list[NEBEndpointPair]:
        """Enumerate environments and build one endpoint pair per environment.

        ``center`` defaults to the hop's first site; ``enumerate_kwargs`` are
        passed to :meth:`enumerate`.
        """
        resolved_mobile_ion_indices = _mobile_ion_indices(mobile_ion_indices)
        if center is None:
            center = resolved_mobile_ion_indices[0]

        endpoint_pairs = []
        for index, environment in enumerate(
            self.enumerate(center=center, cutoff=cutoff, **enumerate_kwargs)
        ):
            pair = self.endpoint_pair(environment, resolved_mobile_ion_indices)
            metadata = {
                **pair.metadata,
                "local_environment_index": index,
                "local_environment_label": environment.label,
                "local_site_indices": environment.local_site_indices,
                "variable_site_indices": environment.variable_site_indices,
                "species_by_site": dict(environment.species_by_site),
                "enumeration": dict(environment.metadata),
            }
            endpoint_pairs.append(replace(pair, metadata=metadata))
        return endpoint_pairs

    def local_site_indices(
        self,
        center,
        cutoff: float,
        local_site_order=None,
        exclude_center_site=None,
    ) -> tuple[int, ...]:
        """Return ordered active-site indices within ``cutoff`` of ``center``."""
        structure = self.active_lattice_structure.template_structure.copy()
        structure.remove_oxidation_states()
        order = LocalSiteOrder.resolve(local_site_order)
        if exclude_center_site is not None:
            order = order.with_exclude_center_site(exclude_center_site)

        center_site, center_index = resolve_center_site(structure, center)
        local_env_sites = order.order_local_env_sites(
            structure.get_sites_in_sphere(center_site.coords, cutoff, include_index=True),
            center_site,
            center_index,
        )
        return tuple(int(site_info[2]) for site_info in local_env_sites)

    def _base_occupation(
        self,
        base_structure: Structure | None,
        tol: float,
        angle_tol: float,
    ) -> Occupation:
        lattice = self.active_lattice_structure
        if base_structure is not None:
            active_base_structure = self.active_site_order.filter_active_structure(
                base_structure, tol=tol
            )
            return lattice.get_occ_from_structure(
                active_base_structure,
                tol=tol,
                angle_tol=angle_tol,
            )
        data = np.full(
            len(lattice.template_structure),
            lattice.basis.match_value,
            dtype=type(lattice.basis.match_value),
        )
        return Occupation(data, basis=lattice.basis, validate=False)

    def _resolve_variable_site_indices(
        self,
        local_site_indices: tuple[int, ...],
        variable_site_indices: Sequence[int] | None,
        variable_species: Sequence[Any] | None,
    ) -> tuple[int, ...]:
        local_sites = set(local_site_indices)
        if variable_site_indices is not None:
            resolved = tuple(int(index) for index in variable_site_indices)
            missing = [index for index in resolved if index not in local_sites]
            if missing:
                raise ValueError(
                    f"variable_site_indices must be inside the local environment: {missing}"
                )
            for index in resolved:
                self._allowed_choices(index, variable_species)
            return resolved

        resolved_sites = []
        for index in local_site_indices:
            allowed = self.active_lattice_structure.allowed_species[index]
            if allowed is None or len(allowed) <= 1:
                continue
            try:
                choices = self._allowed_choices(index, variable_species)
            except ValueError as exc:
                if variable_species is not None and "No variable species" in str(exc):
                    continue
                raise
            if len(choices) > 1:
                resolved_sites.append(index)
        return tuple(resolved_sites)

    def _allowed_choices(
        self,
        site_index: int,
        variable_species: Sequence[Any] | None = None,
    ) -> tuple[Any, ...]:
        allowed_species = self.active_lattice_structure.allowed_species
        if site_index < 0 or site_index >= len(allowed_species):
            raise IndexError(f"site index {site_index} is out of range")
        allowed = allowed_species[site_index]
        if not allowed:
            raise ValueError(f"No allowed species defined for site {site_index}")

        choices = tuple(allowed)
        if variable_species is not None:
            choices = tuple(
                specie
                for specie in choices
                if any(species_label(token) in species_tokens(specie) for token in variable_species)
            )
        if not choices:
            raise ValueError(f"No variable species are allowed at site {site_index}")
        return choices

    def _enumerate_with_transformation(
        self,
        base_occupation: Occupation,
        local_site_indices: tuple[int, ...],
        variable_site_indices: tuple[int, ...],
        variable_species: Sequence[Any] | None,
        species_counts: dict[str, int] | None,
        transformation: Any,
        return_ranked_list: bool | int,
        max_results: int,
        tol: float,
        angle_tol: float,
    ) -> list[LocalEnvironmentEnumeration]:
        disordered_structure = self._build_disordered_structure(
            base_occupation, variable_site_indices, variable_species
        )
        ranked_request = (
            return_ranked_list
            if isinstance(return_ranked_list, int) and not isinstance(return_ranked_list, bool)
            else max_results if return_ranked_list else False
        )
        transformed = transformation.apply_transformation(
            disordered_structure,
            return_ranked_list=ranked_request,
        )

        results = []
        for structure, metadata in _iter_transformed_structures(transformed):
            occupation = self.active_lattice_structure.get_occ_from_structure(
                structure,
                tol=tol,
                angle_tol=angle_tol,
            )
            species_by_site = self._species_by_site(occupation, variable_site_indices)
            if not _matches_counts(species_by_site, species_counts):
                continue
            result_metadata = {"source": "transformation"}
            result_metadata.update(metadata)
            results.append(
                self._build_enumeration(
                    occupation=occupation,
                    local_site_indices=local_site_indices,
                    variable_site_indices=variable_site_indices,
                    species_by_site=species_by_site,
                    metadata=result_metadata,
                )
            )
            if len(results) >= max_results:
                break
        return results

    def _build_disordered_structure(
        self,
        base_occupation: Occupation,
        variable_site_indices: tuple[int, ...],
        variable_species: Sequence[Any] | None,
    ) -> Structure:
        lattice = self.active_lattice_structure
        variable_sites = set(variable_site_indices)
        species_entries = []
        frac_coords = []
        for site_index, template_site in enumerate(lattice.template_structure):
            if site_index in variable_sites:
                entry = _partial_species_entry(
                    self._allowed_choices(site_index, variable_species)
                )
            else:
                specie = lattice.species_for_occupation_value(
                    site_index,
                    base_occupation[site_index],
                )
                if is_vacancy_species(specie):
                    continue
                entry = specie
            species_entries.append(entry)
            frac_coords.append(template_site.frac_coords)
        return Structure(
            lattice.template_structure.lattice,
            species_entries,
            frac_coords,
            coords_are_cartesian=False,
        )

    def _build_enumeration(
        self,
        occupation: Occupation,
        local_site_indices: tuple[int, ...],
        variable_site_indices: tuple[int, ...],
        species_by_site: dict[int, str],
        metadata: dict[str, Any],
    ) -> LocalEnvironmentEnumeration:
        return LocalEnvironmentEnumeration(
            structure=self._structure_from_occupation(occupation),
            full_occupation=occupation,
            local_occupation=occupation[list(local_site_indices)],
            local_site_indices=local_site_indices,
            variable_site_indices=variable_site_indices,
            species_by_site=dict(species_by_site),
            label=_environment_label(species_by_site),
            metadata=metadata,
        )

    def _ordered_endpoint_structures(
        self,
        initial_occupation: Occupation,
        final_occupation: Occupation,
        from_site: int,
        to_site: int,
    ) -> tuple[Structure, Structure]:
        lattice = self.active_lattice_structure
        full_structure = self.active_site_order.full_structure_with_properties()
        original_to_active = self.active_site_order.original_to_active
        original_to_site = self.active_site_order.active_to_original[to_site]

        initial_sites = []
        final_sites = []
        for original_site_index, template_site in enumerate(full_structure):
            active_site_index = original_to_active.get(original_site_index)
            if active_site_index is None:
                initial_sites.append(_periodic_site_from_site(template_site, template_site.specie))
                final_sites.append(_periodic_site_from_site(template_site, template_site.specie))
                continue

            initial_species = lattice.species_for_occupation_value(
                active_site_index,
                initial_occupation[active_site_index],
            )
            if is_vacancy_species(initial_species):
                continue

            initial_sites.append(_periodic_site_from_site(template_site, initial_species))
            if active_site_index == from_site:
                final_template_site = full_structure[original_to_site]
                final_species = lattice.species_for_occupation_value(
                    to_site, final_occupation[to_site]
                )
            else:
                final_template_site = template_site
                final_species = lattice.species_for_occupation_value(
                    active_site_index,
                    final_occupation[active_site_index],
                )
            if is_vacancy_species(final_species):
                raise ValueError("final endpoint lost an initially occupied non-hop site")
            final_sites.append(_periodic_site_from_site(final_template_site, final_species))

        if len(initial_sites) != len(final_sites):
            raise ValueError("initial and final endpoints have different site counts")
        return Structure.from_sites(initial_sites), Structure.from_sites(final_sites)

    def _structure_from_occupation(self, occupation: Occupation) -> Structure:
        full_structure = self.active_site_order.full_structure_with_properties()
        original_to_active = self.active_site_order.original_to_active
        sites = []
        for original_site_index, template_site in enumerate(full_structure):
            active_site_index = original_to_active.get(original_site_index)
            if active_site_index is None:
                sites.append(_periodic_site_from_site(template_site, template_site.specie))
                continue
            species = self.active_lattice_structure.species_for_occupation_value(
                active_site_index, occupation[active_site_index]
            )
            if is_vacancy_species(species):
                continue
            sites.append(_periodic_site_from_site(template_site, species))
        return Structure.from_sites(sites)

    def _occupation_from(
        self,
        local_environment_enumeration: LocalEnvironmentEnumeration | Occupation | Sequence[int],
    ) -> Occupation:
        if isinstance(local_environment_enumeration, LocalEnvironmentEnumeration):
            return local_environment_enumeration.full_occupation.copy()
        values = (
            local_environment_enumeration.data
            if isinstance(local_environment_enumeration, Occupation)
            else local_environment_enumeration
        )
        return Occupation(
            self.active_site_order.select_active_values(values),
            basis=self.active_lattice_structure.basis,
            validate=True,
        )

    def _species_by_site(
        self,
        occupation: Occupation,
        site_indices: Sequence[int],
    ) -> dict[int, str]:
        lattice = self.active_lattice_structure
        return {
            int(site_index): species_label(
                lattice.species_for_occupation_value(int(site_index), occupation[int(site_index)])
            )
            for site_index in site_indices
        }

    def _first_allowed_species(self, site_index: int) -> Any:
        allowed = self.active_lattice_structure.allowed_species[site_index]
        if not allowed:
            raise ValueError(f"No allowed species defined for site {site_index}")
        return allowed[0]


def enumerate_local_environments(
    lattice_structure: LatticeStructure,
    center,
    cutoff: float,
    species_counts: Mapping[Any, int] | None = None,
    variable_species: Sequence[Any] | None = None,
    variable_site_indices: Sequence[int] | None = None,
    local_site_order=None,
    exclude_center_site=None,
    base_structure: Structure | None = None,
    transformation: Any | None = None,
    return_ranked_list: bool | int = True,
    max_results: int = 10000,
    tol: float = 0.1,
    angle_tol: float = 5,
) -> list[LocalEnvironmentEnumeration]:
    """Enumerate local site assignments from a lattice model.

    The default path is a deterministic Cartesian product over allowed species
    on selected local sites. If ``transformation`` is provided, it is called as a
    pymatgen transformation object and its ordered structures are normalized into
    the same result type.
    """
    if max_results < 1:
        raise ValueError("max_results must be at least 1")
    return LocalEnvironmentEnumerator(lattice_structure).enumerate(
        center=center,
        cutoff=cutoff,
        species_counts=species_counts,
        variable_species=variable_species,
        variable_site_indices=variable_site_indices,
        local_site_order=local_site_order,
        exclude_center_site=exclude_center_site,
        base_structure=base_structure,
        transformation=transformation,
        return_ranked_list=return_ranked_list,
        max_results=max_results,
        tol=tol,
        angle_tol=angle_tol,
    )


def generate_neb_endpoint_pair(
    lattice_structure: LatticeStructure,
    local_environment_enumeration: LocalEnvironmentEnumeration | Occupation | Sequence[int],
    mobile_ion_indices: Any | Sequence[int],
) -> NEBEndpointPair:
    """Generate ordered initial and final endpoint structures for one hop."""
    return LocalEnvironmentEnumerator(lattice_structure).endpoint_pair(
        local_environment_enumeration, mobile_ion_indices
    )


def enumerate_neb_endpoint_pairs(
    lattice_structure: LatticeStructure,
    mobile_ion_indices: Any | Sequence[int],
    cutoff: float,
    center=None,
    species_counts: Mapping[Any, int] | None = None,
    variable_species: Sequence[Any] | None = None,
    variable_site_indices: Sequence[int] | None = None,
    local_site_order=None,
    exclude_center_site=None,
    base_structure: Structure | None = None,
    transformation: Any | None = None,
    return_ranked_list: bool | int = True,
    max_results: int = 10000,
    tol: float = 0.1,
    angle_tol: float = 5,
) -> list[NEBEndpointPair]:
    """Enumerate local environments and build NEB endpoint pairs for one hop."""
    resolved_mobile_ion_indices = _mobile_ion_indices(mobile_ion_indices)
    if max_results < 1:
        raise ValueError("max_results must be at least 1")
    return LocalEnvironmentEnumerator(lattice_structure).endpoint_pairs(
        resolved_mobile_ion_indices,
        cutoff=cutoff,
        center=center,
        species_counts=species_counts,
        variable_species=variable_species,
        variable_site_indices=variable_site_indices,
        local_site_order=local_site_order,
        exclude_center_site=exclude_center_site,
        base_structure=base_structure,
        transformation=transformation,
        return_ranked_list=return_ranked_list,
        max_results=max_results,
        tol=tol,
        angle_tol=angle_tol,
    )


def _partial_species_entry(choices: Sequence[Any]) -> dict[Any, float]:
    real_species = [specie for specie in choices if not is_vacancy_species(specie)]
    if not real_species:
        raise ValueError("A disordered transformation site cannot contain only vacancy")
    occupancy = 1.0 / len(choices)
    return {specie: occupancy for specie in real_species}


def _iter_transformed_structures(transformed: Any) -> Iterable[tuple[Structure, dict[str, Any]]]:
    if isinstance(transformed, Structure):
        yield transformed, {}
        return
    if isinstance(transformed, list):
        for entry in transformed:
            if isinstance(entry, Structure):
                yield entry, {}
            elif isinstance(entry, dict) and isinstance(entry.get("structure"), Structure):
                metadata = {key: value for key, value in entry.items() if key != "structure"}
                yield entry["structure"], metadata
            else:
                raise TypeError("Unsupported transformation result entry")
        return
    raise TypeError("Unsupported transformation result")


def _periodic_site_from_site(template_site, specie: Any) -> PeriodicSite:
    return PeriodicSite(
        species=specie,
        coords=template_site.frac_coords,
        lattice=template_site.lattice,
        coords_are_cartesian=False,
        properties=dict(template_site.properties),
    )


def _mobile_ion_indices(mobile_ion_indices: Any | Sequence[int]) -> tuple[int, int]:
    indices = getattr(mobile_ion_indices, "mobile_ion_indices", mobile_ion_indices)
    if len(indices) != 2:
        raise ValueError("mobile_ion_indices must contain exactly two site indices")
    return int(indices[0]), int(indices[1])


def _matches_counts(
    species_by_site: Mapping[int, str],
    species_counts: dict[str, int] | None,
) -> bool:
    if species_counts is None:
        return True
    return Counter(species_by_site.values()) == species_counts


def _canonical_counts(species_counts: Mapping[Any, int] | None) -> dict[str, int] | None:
    if species_counts is None:
        return None
    return {species_label(key): int(value) for key, value in species_counts.items()}


def _environment_label(species_by_site: Mapping[int, str]) -> str:
    if not species_by_site:
        return "base"
    return "|".join(
        f"{site_index}:{species_by_site[site_index]}"
        for site_index in sorted(species_by_site)
    )
