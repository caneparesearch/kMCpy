"""Neighbor-list helpers for kMC local-environment metadata."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from monty.serialization import loadfn
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.core import Structure

_DATA_DIR = Path(__file__).resolve().parent


def prepare_cutoff_neighbor_lookup(
    cutoff_dict: dict[tuple[str, str], float] | None,
) -> dict[str, Any]:
    """Precompute lookup tables for fixed-distance neighbor searches."""
    cutoff_dict = cutoff_dict or {}
    max_dist = 0.0
    lookup_dict: dict[str, dict[str, float]] = defaultdict(dict)
    for (sp1, sp2), distance in cutoff_dict.items():
        lookup_dict[sp1][sp2] = distance
        lookup_dict[sp2][sp1] = distance
        max_dist = max(float(distance), max_dist)
    return {"lookup_dict": lookup_dict, "max_dist": max_dist}


def _neighbor_base_info(structure: Structure, neighbor) -> dict[str, Any]:
    return {
        "site": neighbor,
        "image": NearNeighbors._get_image(structure, neighbor),
        "weight": neighbor.nn_distance,
        "site_index": NearNeighbors._get_original_site(structure, neighbor),
    }


def _copy_kmc_site_metadata(
    structure: Structure,
    neighbor,
    info: dict[str, Any],
) -> None:
    if "wyckoff_sequence" not in structure.site_properties:
        return
    info["wyckoff_sequence"] = neighbor.properties["wyckoff_sequence"]
    info["local_index"] = neighbor.properties["local_index"]
    info["label"] = neighbor.properties["label"]
    if "supercell" in structure.site_properties:
        info["supercell"] = neighbor.properties["supercell"]


def get_cutoff_neighbor_info(
    structure: Structure,
    n: int,
    cutoff_lookup: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return fixed-cutoff neighbor dictionaries with kMC site metadata."""
    site = structure[n]
    neighbors = structure.get_neighbors(site, cutoff_lookup["max_dist"])
    lookup_dict = cutoff_lookup["lookup_dict"]

    neighbor_info = []
    for neighbor in neighbors:
        distance = neighbor.nn_distance
        cutoff = lookup_dict.get(site.species_string, {}).get(
            neighbor.species_string,
            0.0,
        )
        if distance < cutoff:
            info = _neighbor_base_info(structure, neighbor)
            _copy_kmc_site_metadata(structure, neighbor, info)
            neighbor_info.append(info)
    return neighbor_info


def prepare_range_cutoff_neighbor_lookup(
    cutoff_dict: dict[tuple[str, str], tuple[float, float]] | None,
) -> dict[str, Any]:
    """Precompute lookup tables for range-cutoff neighbor searches."""
    cutoff_dict = cutoff_dict or {}
    max_dist = 0.0
    min_dist = 1e3
    lookup_dict_max: dict[str, dict[str, float]] = defaultdict(dict)
    lookup_dict_min: dict[str, dict[str, float]] = defaultdict(dict)
    for (sp1, sp2), (dist_min, dist_max) in cutoff_dict.items():
        lookup_dict_max[sp1][sp2] = dist_max
        lookup_dict_max[sp2][sp1] = dist_max
        lookup_dict_min[sp1][sp2] = dist_min
        lookup_dict_min[sp2][sp1] = dist_min
        max_dist = max(float(dist_max), max_dist)
        min_dist = min(float(dist_min), min_dist)
    return {
        "lookup_dict_max": lookup_dict_max,
        "lookup_dict_min": lookup_dict_min,
        "max_dist": max_dist,
        "min_dist": min_dist,
    }


def get_range_cutoff_neighbor_info(
    structure: Structure,
    n: int,
    cutoff_lookup: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return neighbor dictionaries for species-pair distance ranges."""
    site = structure[n]
    neighbors = structure.get_neighbors(site, cutoff_lookup["max_dist"])
    lookup_max = cutoff_lookup["lookup_dict_max"]
    lookup_min = cutoff_lookup["lookup_dict_min"]

    neighbor_info = []
    for neighbor in neighbors:
        distance = neighbor.nn_distance
        cutoff_max = lookup_max.get(site.species_string, {}).get(
            neighbor.species_string,
            0.0,
        )
        cutoff_min = lookup_min.get(site.species_string, {}).get(
            neighbor.species_string,
            0.0,
        )
        if cutoff_min < distance < cutoff_max:
            info = _neighbor_base_info(structure, neighbor)
            _copy_kmc_site_metadata(structure, neighbor, info)
            neighbor_info.append(info)
    return neighbor_info


def prepare_range_cutoff_neighbor_lookup_from_preset(preset: str) -> dict[str, Any]:
    """Load a named range-cutoff preset."""
    if preset == "vesta_2019":
        cutoff_dict = loadfn(_DATA_DIR / "vesta_cutoffs.yaml")
        return prepare_range_cutoff_neighbor_lookup(cutoff_dict)
    raise ValueError(f"Unrecognized preset: {preset}")


__all__ = [
    "get_cutoff_neighbor_info",
    "get_range_cutoff_neighbor_info",
    "prepare_cutoff_neighbor_lookup",
    "prepare_range_cutoff_neighbor_lookup",
    "prepare_range_cutoff_neighbor_lookup_from_preset",
]
