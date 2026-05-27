"""Structure helpers for kMC site metadata and supercell indexing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import warnings

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.sites import PeriodicSite
from pymatgen.util.coord import lattice_points_in_supercell


def structure_from_sites(
    sites: Sequence[PeriodicSite],
    charge: float | None = None,
    validate_proximity: bool = False,
    to_unit_cell: bool = False,
    properties: Mapping[str, Sequence] | None = None,
) -> Structure:
    """Create a pymatgen Structure from sites while preserving site properties."""
    if len(sites) < 1:
        raise ValueError("At least one site is required to construct a Structure")

    prop_keys: list[str] = []
    props = {key: list(value) for key, value in (properties or {}).items()}
    lattice = sites[0].lattice
    for index, site in enumerate(sites):
        if site.lattice != lattice:
            raise ValueError("Sites must belong to the same lattice")
        for key, value in site.properties.items():
            if key not in prop_keys:
                prop_keys.append(key)
                props[key] = [None] * len(sites)
            props[key][index] = value

    for key, values in props.items():
        if any(value is None for value in values):
            warnings.warn(
                f"Not all sites have property {key}. Missing values are set to None.",
                UserWarning,
                stacklevel=2,
            )

    return Structure(
        lattice,
        [site.species for site in sites],
        [site.frac_coords for site in sites],
        charge=charge,
        site_properties=props,
        validate_proximity=validate_proximity,
        to_unit_cell=to_unit_cell,
    )


def find_site_by_wyckoff_sequence_and_label(
    structure: Structure,
    wyckoff_sequence: int = 0,
    label: str = "Na2",
):
    """Return the first site matching a Wyckoff sequence and CIF label."""
    if "wyckoff_sequence" not in structure.site_properties:
        raise ValueError(
            "wyckoff_sequence not found in site properties; load the structure "
            "with kmcpy.io.cif.load_labeled_structure_from_cif()."
        )
    for site in structure:
        if (
            site.properties["wyckoff_sequence"] == wyckoff_sequence
            and site.properties["label"] == label
        ):
            return site
    raise ValueError(
        "No site found for wyckoff_sequence="
        f"{wyckoff_sequence!r}, label={label!r}."
    )


def find_site_by_wyckoff_sequence_label_and_supercell(
    structure: Structure,
    wyckoff_sequence: int = 0,
    label: str = "Na2",
    supercell: tuple[int, int, int] = (0, 1, 1),
    return_index: bool = True,
):
    """Find a site or site index by Wyckoff sequence, label, and supercell image."""
    if "wyckoff_sequence" not in structure.site_properties:
        raise ValueError(
            "wyckoff_sequence not found in site properties; load the structure "
            "with kmcpy.io.cif.load_labeled_structure_from_cif()."
        )
    for index, site in enumerate(structure):
        if (
            site.properties["wyckoff_sequence"] == wyckoff_sequence
            and site.properties["label"] == label
            and site.properties["supercell"] == supercell
        ):
            return index if return_index else site
    raise ValueError(
        "No site found for wyckoff_sequence="
        f"{wyckoff_sequence!r}, label={label!r}, supercell={supercell!r}."
    )


def make_kmc_supercell(
    structure: Structure,
    scaling_matrix=(1, 2, 3),
    to_unit_cell: bool = True,
) -> Structure:
    """Build a supercell and annotate each site with its supercell image."""
    scale_matrix = np.array(scaling_matrix, int)
    if scale_matrix.shape != (3, 3):
        scale_matrix = np.array(scale_matrix * np.eye(3), int)
    new_lattice = Lattice(np.dot(scale_matrix, structure.lattice.matrix))
    translations = lattice_points_in_supercell(scale_matrix)

    new_sites = []
    for site in structure:
        for translation in translations:
            properties = site.properties.copy()
            properties["supercell"] = tuple(
                int(value)
                for value in np.rint(np.dot(translation, scale_matrix)).astype(int)
            )
            new_sites.append(
                PeriodicSite(
                    site.species,
                    site.coords + new_lattice.get_cartesian_coords(translation),
                    new_lattice,
                    properties=properties,
                    coords_are_cartesian=True,
                    to_unit_cell=True,
                    skip_checks=True,
                )
            )

    charge = getattr(structure, "charge", None)
    new_charge = charge * np.linalg.det(scale_matrix) if charge else None
    supercell = structure_from_sites(
        new_sites,
        charge=new_charge,
        to_unit_cell=to_unit_cell,
    )

    if to_unit_cell:
        for site in supercell:
            site.to_unit_cell(in_place=True)

    return supercell


def kmc_info_key(
    supercell: tuple[int, int, int] = (1, 2, 3),
    label: str = "Na2",
    wyckoff_sequence: int = 2,
) -> tuple[int, int, int, str, int]:
    """Build the historical key from supercell, label, and Wyckoff sequence."""
    return (
        int(supercell[0]),
        int(supercell[1]),
        int(supercell[2]),
        label,
        int(wyckoff_sequence),
    )


def site_index_key(
    supercell: tuple[int, int, int] = (1, 2, 3),
    label: str = "Na2",
    local_index: int = 2,
) -> tuple[int, int, int, str, int]:
    """Build a unique active-site key from supercell, label, and local index."""
    return (
        int(supercell[0]),
        int(supercell[1]),
        int(supercell[2]),
        label,
        int(local_index),
    )


def build_site_index(
    structure: Structure,
    skip_check: bool = True,
) -> dict[tuple[int, int, int, str, int], int]:
    """Map ``site_index_key`` values to structure site indices."""
    for prerequisite in ["supercell", "label", "local_index"]:
        if prerequisite not in structure.site_properties:
            raise KeyError(
                "Expected site properties 'supercell', 'label', and 'local_index'. "
                "Use load_labeled_structure_from_cif() followed by make_kmc_supercell()."
            )

    index_by_key: dict[tuple[int, int, int, str, int], int] = {}
    for site_index, site in enumerate(structure):
        key = site_index_key(
            supercell=site.properties["supercell"],
            label=site.properties["label"],
            local_index=site.properties["local_index"],
        )
        if not skip_check and key in index_by_key:
            raise KeyError(
                "Duplicate site identifier in kMC site index: "
                f"{key!r}."
            )
        index_by_key[key] = site_index
    return index_by_key


__all__ = [
    "build_site_index",
    "find_site_by_wyckoff_sequence_and_label",
    "find_site_by_wyckoff_sequence_label_and_supercell",
    "kmc_info_key",
    "make_kmc_supercell",
    "site_index_key",
    "structure_from_sites",
]
