"""Utilities for enumerating local site states and NEB endpoints."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from pymatgen.core import DummySpecies, PeriodicSite, Species, Structure

from kmcpy.structure.basis import Occupation
from kmcpy.structure.lattice_structure import LatticeStructure
from kmcpy.structure.local_site_ordering import LocalSiteOrderingConvention
from kmcpy.structure.vacancy import Vacancy


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


def enumerate_local_environments(
    lattice_structure: LatticeStructure,
    center,
    cutoff: float,
    species_counts: Mapping[Any, int] | None = None,
    variable_species: Sequence[Any] | None = None,
    variable_site_indices: Sequence[int] | None = None,
    exclude_species: Sequence[str] | None = None,
    ordering_convention=None,
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
    if exclude_species:
        raise ValueError(
            "exclude_species is no longer supported; encode fixed sites in "
            "site_mapping with a single allowed species."
        )

    active_lattice_structure, active_site_index_map = _active_lattice_context(
        lattice_structure
    )
    local_site_indices = _local_site_indices(
        active_lattice_structure,
        center=center,
        cutoff=cutoff,
        exclude_species=None,
        ordering_convention=ordering_convention,
        exclude_center_site=exclude_center_site,
    )
    base_occupation = _base_occupation(
        active_lattice_structure,
        active_site_index_map,
        base_structure,
        tol,
        angle_tol,
    )
    variable_sites = _resolve_variable_site_indices(
        active_lattice_structure,
        local_site_indices,
        variable_site_indices,
        variable_species,
    )
    exact_counts = _canonical_counts(species_counts)
    if exact_counts is not None and sum(exact_counts.values()) != len(variable_sites):
        raise ValueError(
            "species_counts must sum to the number of variable sites for exact enumeration"
        )

    if transformation is not None:
        return _enumerate_with_transformation(
            lattice_structure=lattice_structure,
            active_lattice_structure=active_lattice_structure,
            active_site_index_map=active_site_index_map,
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

    choices_by_site = {
        site_index: _allowed_choices(
            active_lattice_structure,
            site_index,
            variable_species=variable_species,
        )
        for site_index in variable_sites
    }
    results: list[LocalEnvironmentEnumeration] = []
    for choices in product(*(choices_by_site[index] for index in variable_sites)):
        species_by_site = {
            site_index: _species_label(specie)
            for site_index, specie in zip(variable_sites, choices)
        }
        if not _matches_counts(species_by_site, exact_counts):
            continue

        occupation = base_occupation.copy()
        for site_index, specie in zip(variable_sites, choices):
            occupation[site_index] = _occupation_value_for_species(
                active_lattice_structure,
                site_index,
                specie,
            )
        results.append(
            _build_enumeration(
                lattice_structure=lattice_structure,
                active_lattice_structure=active_lattice_structure,
                active_site_index_map=active_site_index_map,
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


def generate_neb_endpoint_pair(
    lattice_structure: LatticeStructure,
    local_environment_enumeration: LocalEnvironmentEnumeration | Occupation | Sequence[int],
    mobile_ion_indices: Any | Sequence[int],
) -> NEBEndpointPair:
    """Generate ordered initial and final endpoint structures for one hop."""
    active_lattice_structure, active_site_index_map = _active_lattice_context(
        lattice_structure
    )
    from_site, to_site = _mobile_ion_indices(mobile_ion_indices)
    active_site_index_map.validate_active_indices(
        (from_site, to_site), field_name="mobile_ion_indices"
    )
    if from_site == to_site:
        raise ValueError("mobile_ion_indices must contain two distinct sites")

    initial_occupation = _occupation_from_local_environment_enumeration(
        local_environment_enumeration,
        active_lattice_structure,
        active_site_index_map,
    )
    if len(initial_occupation) != active_site_index_map.active_site_count:
        raise ValueError("environment occupation length does not match active sites")

    from_species = _first_allowed_species(active_lattice_structure, from_site)
    to_species = _first_allowed_species(active_lattice_structure, to_site)
    if _is_vacancy(from_species) or _is_vacancy(to_species):
        raise ValueError("mobile-ion sites must use a real species as the first mapping")
    if not _species_equivalent(from_species, to_species):
        raise ValueError("hop sites must have the same first allowed mobile species")

    initial_occupation = initial_occupation.copy()
    initial_occupation[from_site] = active_lattice_structure.basis.match_value
    initial_occupation[to_site] = active_lattice_structure.basis.mismatch_value
    final_occupation = initial_occupation.flip([from_site, to_site])

    initial, final = _ordered_endpoint_structures(
        lattice_structure,
        active_lattice_structure,
        active_site_index_map,
        initial_occupation,
        final_occupation,
        from_site,
        to_site,
    )
    return NEBEndpointPair(
        initial=initial,
        final=final,
        initial_occupation=initial_occupation,
        final_occupation=final_occupation,
        mobile_ion_indices=(from_site, to_site),
        metadata={"mobile_ion_indices": (from_site, to_site)},
    )


def enumerate_neb_endpoint_pairs(
    lattice_structure: LatticeStructure,
    mobile_ion_indices: Any | Sequence[int],
    cutoff: float,
    center=None,
    species_counts: Mapping[Any, int] | None = None,
    variable_species: Sequence[Any] | None = None,
    variable_site_indices: Sequence[int] | None = None,
    exclude_species: Sequence[str] | None = None,
    ordering_convention=None,
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
    if center is None:
        center = resolved_mobile_ion_indices[0]

    local_environment_enumerations = enumerate_local_environments(
        lattice_structure=lattice_structure,
        center=center,
        cutoff=cutoff,
        species_counts=species_counts,
        variable_species=variable_species,
        variable_site_indices=variable_site_indices,
        exclude_species=exclude_species,
        ordering_convention=ordering_convention,
        exclude_center_site=exclude_center_site,
        base_structure=base_structure,
        transformation=transformation,
        return_ranked_list=return_ranked_list,
        max_results=max_results,
        tol=tol,
        angle_tol=angle_tol,
    )

    endpoint_pairs = []
    for index, local_environment_enumeration in enumerate(
        local_environment_enumerations
    ):
        pair = generate_neb_endpoint_pair(
            lattice_structure,
            local_environment_enumeration,
            resolved_mobile_ion_indices,
        )
        metadata = {
            **pair.metadata,
            "local_environment_index": index,
            "local_environment_label": local_environment_enumeration.label,
            "local_site_indices": local_environment_enumeration.local_site_indices,
            "variable_site_indices": (
                local_environment_enumeration.variable_site_indices
            ),
            "species_by_site": dict(local_environment_enumeration.species_by_site),
            "enumeration": dict(local_environment_enumeration.metadata),
        }
        endpoint_pairs.append(replace(pair, metadata=metadata))
    return endpoint_pairs


def _active_lattice_context(lattice_structure: LatticeStructure):
    active_site_index_map = lattice_structure.get_active_site_index_map()
    active_lattice_structure = lattice_structure.get_active_lattice_structure()
    return active_lattice_structure, active_site_index_map


def _local_site_indices(
    lattice_structure: LatticeStructure,
    center,
    cutoff: float,
    exclude_species: Sequence[str] | None,
    ordering_convention,
    exclude_center_site,
) -> tuple[int, ...]:
    structure = lattice_structure.template_structure.copy()
    structure.remove_oxidation_states()
    ordering = LocalSiteOrderingConvention.resolve(ordering_convention)
    if exclude_center_site is not None:
        ordering = ordering.with_exclude_center_site(exclude_center_site)

    if isinstance(center, int):
        center_site = structure[center]
        center_index = center
    elif isinstance(center, (list, tuple, np.ndarray)):
        center_site = PeriodicSite(
            species=DummySpecies("X"),
            coords=center,
            coords_are_cartesian=False,
            lattice=structure.lattice.copy(),
        )
        center_index = None
    else:
        raise ValueError("Center must be an index or a list of fractional coordinates.")

    excluded = set(_normalize_exclude_species(exclude_species))
    local_env_sites = structure.get_sites_in_sphere(
        center_site.coords,
        cutoff,
        include_index=True,
    )
    if excluded:
        local_env_sites = [
            site_info
            for site_info in local_env_sites
            if site_info[0].species_string not in excluded
            and str(site_info[0].specie) not in excluded
        ]
    if ordering.exclude_center_site:
        local_env_sites = [
            site_info
            for site_info in local_env_sites
            if not _is_center_site(site_info, center_site, center_index, ordering)
        ]
    local_env_sites = ordering.sort_local_env_sites(local_env_sites)
    return tuple(int(site_info[2]) for site_info in local_env_sites)


def _is_center_site(
    site_info,
    center_site: PeriodicSite,
    center_index: int | None,
    ordering: LocalSiteOrderingConvention,
) -> bool:
    site = site_info[0]
    site_index = site_info[2]
    if center_index is not None and int(site_index) == int(center_index):
        return True
    return np.linalg.norm(site.coords - center_site.coords) <= ordering.center_match_tolerance


def _normalize_exclude_species(exclude_species) -> list[str]:
    tokens = []
    for species in exclude_species or []:
        token = str(species)
        tokens.append(token)
        try:
            parsed_species = Species(token)
        except Exception:
            continue
        tokens.append(str(parsed_species.symbol))
        tokens.append(str(parsed_species.element))
    return list(dict.fromkeys(tokens))


def _base_occupation(
    lattice_structure: LatticeStructure,
    active_site_index_map,
    base_structure: Structure | None,
    tol: float,
    angle_tol: float,
) -> Occupation:
    if base_structure is not None:
        active_base_structure = active_site_index_map.filter_active_structure(
            base_structure, tol=tol
        )
        return lattice_structure.get_occ_from_structure(
            active_base_structure,
            tol=tol,
            angle_tol=angle_tol,
        )
    data = np.full(
        len(lattice_structure.template_structure),
        lattice_structure.basis.match_value,
        dtype=type(lattice_structure.basis.match_value),
    )
    return Occupation(data, basis=lattice_structure.basis, validate=False)


def _resolve_variable_site_indices(
    lattice_structure: LatticeStructure,
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
            _allowed_choices(lattice_structure, index, variable_species)
        return resolved

    resolved_sites = []
    for index in local_site_indices:
        allowed = lattice_structure.allowed_species[index]
        if allowed is None or len(allowed) <= 1:
            continue
        try:
            choices = _allowed_choices(lattice_structure, index, variable_species)
        except ValueError as exc:
            if variable_species is not None and "No variable species" in str(exc):
                continue
            raise
        if len(choices) > 1:
            resolved_sites.append(index)
    return tuple(resolved_sites)


def _allowed_choices(
    lattice_structure: LatticeStructure,
    site_index: int,
    variable_species: Sequence[Any] | None = None,
) -> tuple[Any, ...]:
    _validate_site_index(lattice_structure, site_index)
    allowed = lattice_structure.allowed_species[site_index]
    if not allowed:
        raise ValueError(f"No allowed species defined for site {site_index}")
    if len(allowed) > 2:
        raise ValueError(
            "Only binary site mappings are supported by kmcpy occupations"
        )

    choices = tuple(allowed)
    if variable_species is not None:
        choices = tuple(
            specie
            for specie in choices
            if any(_species_matches_token(specie, token) for token in variable_species)
        )
    if not choices:
        raise ValueError(f"No variable species are allowed at site {site_index}")
    return choices


def _enumerate_with_transformation(
    lattice_structure: LatticeStructure,
    active_lattice_structure: LatticeStructure,
    active_site_index_map,
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
    disordered_structure = _build_disordered_structure(
        active_lattice_structure,
        base_occupation,
        variable_site_indices,
        variable_species,
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
        occupation = active_lattice_structure.get_occ_from_structure(
            structure,
            tol=tol,
            angle_tol=angle_tol,
        )
        species_by_site = _species_by_site(
            active_lattice_structure,
            occupation,
            variable_site_indices,
        )
        if not _matches_counts(species_by_site, species_counts):
            continue
        result_metadata = {"source": "transformation"}
        result_metadata.update(metadata)
        results.append(
            _build_enumeration(
                lattice_structure=lattice_structure,
                active_lattice_structure=active_lattice_structure,
                active_site_index_map=active_site_index_map,
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
    lattice_structure: LatticeStructure,
    base_occupation: Occupation,
    variable_site_indices: tuple[int, ...],
    variable_species: Sequence[Any] | None,
) -> Structure:
    variable_sites = set(variable_site_indices)
    species_entries = []
    frac_coords = []
    for site_index, template_site in enumerate(lattice_structure.template_structure):
        if site_index in variable_sites:
            choices = _allowed_choices(
                lattice_structure,
                site_index,
                variable_species=variable_species,
            )
            entry = _partial_species_entry(choices)
        else:
            specie = _species_for_occupation(
                lattice_structure,
                site_index,
                base_occupation[site_index],
            )
            if _is_vacancy(specie):
                continue
            entry = specie
        species_entries.append(entry)
        frac_coords.append(template_site.frac_coords)
    return Structure(
        lattice_structure.template_structure.lattice,
        species_entries,
        frac_coords,
        coords_are_cartesian=False,
    )


def _partial_species_entry(choices: Sequence[Any]) -> dict[Any, float]:
    real_species = [specie for specie in choices if not _is_vacancy(specie)]
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


def _build_enumeration(
    lattice_structure: LatticeStructure,
    active_lattice_structure: LatticeStructure,
    active_site_index_map,
    occupation: Occupation,
    local_site_indices: tuple[int, ...],
    variable_site_indices: tuple[int, ...],
    species_by_site: dict[int, str],
    metadata: dict[str, Any],
) -> LocalEnvironmentEnumeration:
    local_occupation = occupation[list(local_site_indices)]
    return LocalEnvironmentEnumeration(
        structure=_structure_from_active_occupation(
            lattice_structure, active_lattice_structure, active_site_index_map, occupation
        ),
        full_occupation=occupation,
        local_occupation=local_occupation,
        local_site_indices=local_site_indices,
        variable_site_indices=variable_site_indices,
        species_by_site=dict(species_by_site),
        label=_environment_label(species_by_site),
        metadata=metadata,
    )


def _ordered_endpoint_structures(
    lattice_structure: LatticeStructure,
    active_lattice_structure: LatticeStructure,
    active_site_index_map,
    initial_occupation: Occupation,
    final_occupation: Occupation,
    from_site: int,
    to_site: int,
) -> tuple[Structure, Structure]:
    full_structure = active_site_index_map.full_structure_with_properties()
    original_to_active = active_site_index_map.original_to_active
    original_from_site = active_site_index_map.active_to_original[from_site]
    original_to_site = active_site_index_map.active_to_original[to_site]

    initial_sites = []
    final_sites = []
    for original_site_index, template_site in enumerate(full_structure):
        active_site_index = original_to_active.get(original_site_index)
        if active_site_index is None:
            initial_sites.append(_periodic_site_from_site(template_site, template_site.specie))
            final_sites.append(_periodic_site_from_site(template_site, template_site.specie))
            continue

        initial_species = _species_for_occupation(
            active_lattice_structure,
            active_site_index,
            initial_occupation[active_site_index],
        )
        if _is_vacancy(initial_species):
            continue

        initial_sites.append(_periodic_site_from_site(template_site, initial_species))
        if active_site_index == from_site:
            final_template_site = full_structure[original_to_site]
            final_species = _species_for_occupation(
                active_lattice_structure, to_site, final_occupation[to_site]
            )
        else:
            final_template_site = template_site
            final_species = _species_for_occupation(
                active_lattice_structure,
                active_site_index,
                final_occupation[active_site_index],
            )
        if _is_vacancy(final_species):
            raise ValueError("final endpoint lost an initially occupied non-hop site")
        final_sites.append(_periodic_site_from_site(final_template_site, final_species))

    if len(initial_sites) != len(final_sites):
        raise ValueError("initial and final endpoints have different site counts")
    return Structure.from_sites(initial_sites), Structure.from_sites(final_sites)


def _structure_from_active_occupation(
    lattice_structure: LatticeStructure,
    active_lattice_structure: LatticeStructure,
    active_site_index_map,
    occupation: Occupation,
) -> Structure:
    full_structure = active_site_index_map.full_structure_with_properties()
    original_to_active = active_site_index_map.original_to_active
    sites = []
    for original_site_index, template_site in enumerate(full_structure):
        active_site_index = original_to_active.get(original_site_index)
        if active_site_index is None:
            sites.append(_periodic_site_from_site(template_site, template_site.specie))
            continue
        species = _species_for_occupation(
            active_lattice_structure, active_site_index, occupation[active_site_index]
        )
        if _is_vacancy(species):
            continue
        sites.append(_periodic_site_from_site(template_site, species))
    return Structure.from_sites(sites)


def _periodic_site_from_site(template_site, specie: Any) -> PeriodicSite:
    return PeriodicSite(
        species=specie,
        coords=template_site.frac_coords,
        lattice=template_site.lattice,
        coords_are_cartesian=False,
        properties=dict(template_site.properties),
    )

def _periodic_site_from_template(
    lattice_structure: LatticeStructure,
    site_index: int,
    specie: Any,
) -> PeriodicSite:
    template_site = lattice_structure.template_structure[site_index]
    return PeriodicSite(
        species=specie,
        coords=template_site.frac_coords,
        lattice=template_site.lattice,
        coords_are_cartesian=False,
        properties=dict(template_site.properties),
    )


def _occupation_from_local_environment_enumeration(
    local_environment_enumeration: LocalEnvironmentEnumeration | Occupation | Sequence[int],
    lattice_structure: LatticeStructure,
    active_site_index_map,
) -> Occupation:
    if isinstance(local_environment_enumeration, LocalEnvironmentEnumeration):
        return local_environment_enumeration.full_occupation.copy()
    if isinstance(local_environment_enumeration, Occupation):
        values = active_site_index_map.select_active_values(
            local_environment_enumeration.array
        )
        return Occupation(values, basis=lattice_structure.basis, validate=True)
    values = active_site_index_map.select_active_values(local_environment_enumeration)
    return Occupation(values, basis=lattice_structure.basis, validate=True)


def _mobile_ion_indices(mobile_ion_indices: Any | Sequence[int]) -> tuple[int, int]:
    indices = getattr(mobile_ion_indices, "mobile_ion_indices", mobile_ion_indices)
    if len(indices) != 2:
        raise ValueError("mobile_ion_indices must contain exactly two site indices")
    return int(indices[0]), int(indices[1])


def _species_by_site(
    lattice_structure: LatticeStructure,
    occupation: Occupation,
    site_indices: Sequence[int],
) -> dict[int, str]:
    return {
        int(site_index): _species_label(
            _species_for_occupation(lattice_structure, int(site_index), occupation[int(site_index)])
        )
        for site_index in site_indices
    }


def _species_for_occupation(
    lattice_structure: LatticeStructure,
    site_index: int,
    value: int,
) -> Any:
    allowed = lattice_structure.allowed_species[site_index]
    if not allowed:
        raise ValueError(f"No allowed species defined for site {site_index}")
    if value == lattice_structure.basis.match_value:
        return allowed[0]
    if len(allowed) < 2:
        return allowed[0]
    if value == lattice_structure.basis.mismatch_value:
        return allowed[1]
    raise ValueError(f"Unsupported occupation value {value} at site {site_index}")


def _occupation_value_for_species(
    lattice_structure: LatticeStructure,
    site_index: int,
    specie: Any,
) -> int:
    allowed = lattice_structure.allowed_species[site_index]
    if _species_equivalent(specie, allowed[0]):
        return lattice_structure.basis.match_value
    if len(allowed) > 1 and _species_equivalent(specie, allowed[1]):
        return lattice_structure.basis.mismatch_value
    raise ValueError(f"Species {_species_label(specie)} is not allowed at site {site_index}")


def _first_allowed_species(lattice_structure: LatticeStructure, site_index: int) -> Any:
    allowed = lattice_structure.allowed_species[site_index]
    if not allowed:
        raise ValueError(f"No allowed species defined for site {site_index}")
    return allowed[0]


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
    return {_species_token_label(key): int(value) for key, value in species_counts.items()}


def _environment_label(species_by_site: Mapping[int, str]) -> str:
    if not species_by_site:
        return "base"
    return "|".join(
        f"{site_index}:{species_by_site[site_index]}"
        for site_index in sorted(species_by_site)
    )


def _validate_site_index(lattice_structure: LatticeStructure, site_index: int) -> None:
    if site_index < 0 or site_index >= len(lattice_structure.template_structure):
        raise IndexError(f"site index {site_index} is out of range")


def _species_equivalent(left: Any, right: Any) -> bool:
    return any(_species_matches_token(left, token) for token in _species_tokens(right))


def _species_matches_token(specie: Any, token: Any) -> bool:
    return _species_token_label(token) in _species_tokens(specie)


def _species_label(specie: Any) -> str:
    if _is_vacancy(specie):
        return "X"
    return _species_token_label(specie)


def _species_token_label(token: Any) -> str:
    if _is_vacancy(token):
        return "X"
    if isinstance(token, str):
        return token
    symbol = getattr(token, "symbol", None)
    if symbol is not None:
        return str(symbol)
    return str(token)


def _species_tokens(specie: Any) -> set[str]:
    if _is_vacancy(specie):
        return {"X", "Vacancy"}
    tokens = {str(specie)}
    symbol = getattr(specie, "symbol", None)
    if symbol is not None:
        tokens.add(str(symbol))
    element = getattr(specie, "element", None)
    if element is not None:
        tokens.add(str(element))
    return tokens


def _is_vacancy(specie: Any) -> bool:
    return (
        isinstance(specie, Vacancy)
        or specie == "X"
        or getattr(specie, "symbol", None) == "X"
    )
