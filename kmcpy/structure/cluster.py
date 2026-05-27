"""Finite clusters of sites and deterministic cluster matching."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import hashlib
import itertools
import json
import logging
from typing import Any, Sequence

from monty.json import MSONable
import numpy as np
from pymatgen.core.structure import Molecule

logger = logging.getLogger(__name__)


def _species_string(site: Any) -> str:
    species_string = getattr(site, "species_string", None)
    if species_string is not None:
        return str(species_string)
    specie = getattr(site, "specie", None)
    if specie is not None:
        return str(specie)
    species = getattr(site, "species", None)
    if species is not None:
        return str(species)
    return str(site)


def _site_coords(site: Any) -> np.ndarray:
    coords = getattr(site, "coords", None)
    if coords is None:
        coords = site
    return np.asarray(coords, dtype=float)


def _site_distance(site_a: Any, site_b: Any) -> float:
    distance = getattr(site_a, "distance", None)
    if distance is not None:
        try:
            return float(distance(site_b, jimage=[0, 0, 0]))
        except TypeError:
            return float(distance(site_b))
    return float(np.linalg.norm(_site_coords(site_a) - _site_coords(site_b)))


def _cluster_type(n_sites: int) -> str:
    return {
        1: "point",
        2: "pair",
        3: "triplet",
        4: "quadruplet",
    }.get(n_sites, f"{n_sites}-site")


def _json_safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if key == "site":
            continue
        if isinstance(value, np.ndarray):
            safe[key] = value.tolist()
        elif isinstance(value, tuple):
            safe[key] = list(value)
        elif hasattr(value, "as_dict"):
            safe[key] = value.as_dict()
        elif isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


class Orbit(MSONable):
    """A group of equivalent clusters."""

    def __init__(self):
        self.clusters = []
        self.multiplicity = 0

    def attach_cluster(self, cluster):
        self.clusters.append(cluster)
        self.multiplicity += 1

    def get_cluster_function(self, occupancy):
        """Return the orbit-averaged cluster function."""
        return (1 / self.multiplicity) * sum(
            cluster.get_cluster_function(occupancy) for cluster in self.clusters
        )

    def __str__(self):
        try:
            for i, cluster in enumerate(self.clusters):
                logger.info(
                    "Cluster[%d]: %5s\t%10s\t%8.3f\t%8.3f\t%5s\t%5d",
                    i,
                    cluster.type,
                    str(cluster.site_indices),
                    cluster.max_length,
                    cluster.min_length,
                    cluster.sym,
                    self.multiplicity,
                )
        except TypeError:
            logger.info("No cluster in this orbit!")

    def to_xyz(self, fname):
        self.clusters[0].to_xyz(fname)

    def show_representative_cluster(self):
        logger.info(
            "{0:5s}\t{1:10s}\t{2:8.3f}\t{3:8.3f}\t{4:5s}\t{5:5d}".format(
                self.clusters[0].type,
                str(self.clusters[0].site_indices),
                self.clusters[0].max_length,
                self.clusters[0].min_length,
                self.clusters[0].sym,
                self.multiplicity,
            )
        )

    @property
    def cluster_site_indices(self):
        """Cluster site-index sets included in this orbit."""
        return tuple(
            tuple(int(site_index) for site_index in cluster.site_indices)
            for cluster in self.clusters
        )

    @property
    def fingerprint(self):
        """Stable identity for the orbit term associated with one ECI."""
        cluster_records = []
        for cluster in self.clusters:
            cluster_records.append(
                {
                    "site_indices": [int(index) for index in cluster.site_indices],
                    "labels": [[str(species), str(role)] for species, role in cluster.labels],
                    "distances": np.round(cluster.distance_matrix, 8).tolist(),
                }
            )
        cluster_records = sorted(
            cluster_records,
            key=lambda record: json.dumps(record, sort_keys=True, separators=(",", ":")),
        )
        payload = {
            "clusters": cluster_records,
            "multiplicity": int(self.multiplicity),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def as_dict(self):
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "clusters": [cluster.as_dict() for cluster in self.clusters],
            "multiplicity": self.multiplicity,
            "cluster_site_indices": [list(indices) for indices in self.cluster_site_indices],
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Orbit":
        """Create an orbit from a Monty-style payload."""
        if not isinstance(data, dict):
            raise ValueError("Orbit.from_dict expects a dictionary")
        orbit = cls()
        orbit.clusters = [
            Cluster.from_dict(cluster_data)
            for cluster_data in data.get("clusters", [])
        ]
        orbit.multiplicity = int(data.get("multiplicity", len(orbit.clusters)))
        return orbit


class Cluster(MSONable):
    """A finite labeled cluster of sites.

    The cluster stores species, site indices, optional per-site metadata, and a
    distance matrix. It is used both by local cluster expansion terms and by
    event local-environment matching.
    """

    def __init__(
        self,
        site_indices: Sequence[int] | None = None,
        sites: Sequence[Any] | None = None,
        *,
        species: Sequence[str] | None = None,
        coords: Sequence[Sequence[float]] | None = None,
        roles: Sequence[str] | None = None,
        metadata: Sequence[dict[str, Any]] | None = None,
        distance_matrix: Sequence[Sequence[float]] | None = None,
        sym: str | None = None,
        analyze_symmetry: bool = False,
    ) -> None:
        if sites is not None:
            species = tuple(_species_string(site) for site in sites)
            coords = tuple(_site_coords(site) for site in sites)
            distance_matrix = (
                self._build_distance_matrix_from_sites(sites)
                if distance_matrix is None
                else distance_matrix
            )
        if species is None or coords is None:
            raise ValueError("Cluster requires either sites or species and coords")

        species = tuple(str(value) for value in species)
        coords_array = np.asarray(coords, dtype=float)
        if len(species) == 0 and coords_array.size == 0:
            coords_array = coords_array.reshape((0, 3))
        if coords_array.shape != (len(species), 3):
            raise ValueError("coords must have shape (number_of_sites, 3)")

        if site_indices is None:
            normalized_site_indices = tuple(range(len(species)))
        else:
            normalized_site_indices = tuple(int(value) for value in site_indices)
        if len(normalized_site_indices) != len(species):
            raise ValueError("site_indices must have one entry per site")

        if roles is None:
            roles = tuple("" for _ in species)
        else:
            roles = tuple(str(value) for value in roles)
        if len(roles) != len(species):
            raise ValueError("roles must have one entry per site")

        if metadata is None:
            metadata = tuple({} for _ in species)
        else:
            metadata = tuple(dict(value) for value in metadata)
        if len(metadata) != len(species):
            raise ValueError("metadata must have one entry per site")

        if distance_matrix is None:
            distance_matrix_array = self._build_distance_matrix(coords_array)
        else:
            distance_matrix_array = np.asarray(distance_matrix, dtype=float)
            if distance_matrix_array.shape != (len(species), len(species)):
                raise ValueError(
                    "distance_matrix must have shape "
                    "(number_of_sites, number_of_sites)"
                )

        self.species = species
        self.coords = coords_array
        self.site_indices = tuple(normalized_site_indices)
        self.roles = tuple(roles)
        self.metadata = tuple(metadata)
        self.distance_matrix = distance_matrix_array
        self.type = _cluster_type(len(species))
        self.structure = Molecule(species, coords_array)
        self.max_length, self.min_length, self.bond_distances = (
            self.get_bond_distances()
        )
        self.sym = sym if sym is not None else ""
        if analyze_symmetry:
            self.sym = self.point_group_symbol

    @classmethod
    def from_sites(
        cls,
        sites: Sequence[Any],
        site_indices: Sequence[int] | None = None,
        roles: Sequence[str] | None = None,
        metadata: Sequence[dict[str, Any]] | None = None,
        analyze_symmetry: bool = False,
    ) -> "Cluster":
        """Build a cluster from pymatgen sites or coordinate-like objects."""
        return cls(
            site_indices=site_indices,
            sites=sites,
            roles=roles,
            metadata=metadata,
            analyze_symmetry=analyze_symmetry,
        )

    @classmethod
    def from_neighbor_info(
        cls,
        neighbor_info: Sequence[dict[str, Any]],
        roles: Sequence[str] | None = None,
        analyze_symmetry: bool = False,
    ) -> "Cluster":
        """Build a cluster from pymatgen near-neighbor dictionaries."""
        sites = [neighbor["site"] for neighbor in neighbor_info]
        site_indices = [
            int(neighbor.get("site_index", neighbor.get("local_index", index)))
            for index, neighbor in enumerate(neighbor_info)
        ]
        metadata = [dict(neighbor) for neighbor in neighbor_info]
        return cls.from_sites(
            sites,
            site_indices=site_indices,
            roles=roles,
            metadata=metadata,
            analyze_symmetry=analyze_symmetry,
        )

    @property
    def labels(self) -> tuple[tuple[str, str], ...]:
        """Species and role labels used as node colors during matching."""
        return tuple(zip(self.species, self.roles))

    @property
    def signature(self) -> tuple[tuple[tuple[str, str], int], ...]:
        """Order-independent label-count signature."""
        counts: dict[tuple[str, str], int] = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return tuple(sorted(counts.items(), key=lambda item: item[0]))

    @property
    def fingerprint(self) -> str:
        """Stable cluster fingerprint with rounded distances."""
        payload = {
            "labels": self.labels,
            "distances": np.round(self.distance_matrix, 8).tolist(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def point_group_symbol(self) -> str:
        """Return the cluster point-group symbol, computed on demand."""
        from pymatgen.symmetry.analyzer import PointGroupAnalyzer

        return PointGroupAnalyzer(self.structure).sch_symbol

    @staticmethod
    def _build_distance_matrix(coords: np.ndarray) -> np.ndarray:
        n_sites = len(coords)
        distance_matrix = np.zeros((n_sites, n_sites), dtype=float)
        for i in range(n_sites):
            for j in range(n_sites):
                distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
        return distance_matrix

    @staticmethod
    def _build_distance_matrix_from_sites(sites: Sequence[Any]) -> np.ndarray:
        n_sites = len(sites)
        distance_matrix = np.zeros((n_sites, n_sites), dtype=float)
        for i in range(n_sites):
            for j in range(n_sites):
                distance_matrix[i, j] = _site_distance(sites[i], sites[j])
        return distance_matrix

    def get_bond_distances(self):
        """Return max, min, and sorted pair distances."""
        if len(self.distance_matrix) <= 1:
            return 0, 0, []
        pairs = combinations(range(len(self.distance_matrix)), 2)
        bond_distances = np.array(
            [self.distance_matrix[i, j] for i, j in pairs],
            dtype=float,
        )
        bond_distances.sort()
        return max(bond_distances), min(bond_distances), bond_distances

    def get_cluster_function(self, occupation):
        """Return the occupation product for this cluster."""
        return np.prod([occupation[i] for i in self.site_indices])

    def to_xyz(self, fname):
        """Write the cluster structure as XYZ."""
        structure = self.structure.copy()
        structure.remove_oxidation_states()
        structure.to(filename=fname, fmt="xyz")

    def __eq__(self, other):
        if not isinstance(other, Cluster):
            return False
        if len(self.site_indices) != len(other.site_indices):
            return False
        try:
            ClusterMatcher(self).match(other)
        except ValueError:
            return False
        return True

    def __str__(self):
        return (
            f"Cluster(type={self.type}, site_indices={self.site_indices}, "
            f"max_length={self.max_length:.3f}, min_length={self.min_length:.3f})"
        )

    def as_dict(self):
        """Serialize the cluster."""
        bond_distances = self.bond_distances
        if not isinstance(bond_distances, list):
            bond_distances = bond_distances.tolist()
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "site_indices": list(self.site_indices),
            "type": self.type,
            "structure": self.structure.as_dict(),
            "sym": self.sym,
            "max_length": self.max_length,
            "min_length": self.min_length,
            "bond_distances": bond_distances,
            "roles": list(self.roles),
            "metadata": [_json_safe_metadata(item) for item in self.metadata],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Cluster":
        """Create a cluster from a Monty-style payload."""
        if not isinstance(data, dict):
            raise ValueError("Cluster.from_dict expects a dictionary")
        cluster = cls(
            site_indices=data.get("site_indices", []),
            sites=Molecule.from_dict(data.get("structure", {})),
            roles=data.get("roles"),
            metadata=data.get("metadata"),
            sym=data.get("sym", ""),
        )
        cluster.max_length = data.get("max_length", cluster.max_length)
        cluster.min_length = data.get("min_length", cluster.min_length)
        cluster.bond_distances = data.get("bond_distances", cluster.bond_distances)
        return cluster


@dataclass(frozen=True)
class ClusterMatch:
    """Permutation returned by matching a candidate cluster to a reference."""

    reference_to_candidate: tuple[int, ...]
    candidate_to_reference: tuple[int, ...]
    ordered_candidate_indices: tuple[int, ...]
    fingerprint: str
    is_unique: bool
    score: float = 0.0


class ClusterMatcher:
    """Match candidate clusters to a reference cluster order."""

    def __init__(
        self,
        reference: Cluster,
        rtol: float = 1e-3,
        atol: float = 1e-3,
    ) -> None:
        self.reference = reference
        self.rtol = rtol
        self.atol = atol

    def match(
        self,
        candidate: Cluster,
        *,
        require_unique: bool = False,
        find_nearest_if_fail: bool = False,
    ) -> ClusterMatch:
        """Return the permutation that orders candidate sites like the reference."""
        self._validate_signature(candidate)

        permutations = self._matching_permutations(candidate, stop_after=2)
        if permutations:
            is_unique = len(permutations) == 1
            if require_unique and not is_unique:
                raise ValueError("Cluster match is ambiguous")
            return self._build_match(candidate, permutations[0], is_unique, score=0.0)

        if find_nearest_if_fail:
            permutation, score = self._nearest_permutation(candidate)
            return self._build_match(candidate, permutation, True, score=score)

        raise ValueError("No matching cluster permutation found")

    def ordered_candidate_indices(
        self,
        candidate: Cluster,
        *,
        require_unique: bool = False,
        find_nearest_if_fail: bool = False,
    ) -> tuple[int, ...]:
        """Return candidate site indices ordered in the reference convention."""
        return self.match(
            candidate,
            require_unique=require_unique,
            find_nearest_if_fail=find_nearest_if_fail,
        ).ordered_candidate_indices

    def _validate_signature(self, candidate: Cluster) -> None:
        if self.reference.signature != candidate.signature:
            raise ValueError(
                "Cluster signatures do not match: "
                f"{self.reference.signature} vs {candidate.signature}"
            )

    def _matching_permutations(
        self,
        candidate: Cluster,
        *,
        stop_after: int | None,
    ) -> list[tuple[int, ...]]:
        candidate_by_label: dict[tuple[str, str], list[int]] = {}
        for index, label in enumerate(candidate.labels):
            candidate_by_label.setdefault(label, []).append(index)

        reference_order = self._reference_assignment_order(candidate_by_label)
        reference_to_candidate: list[int | None] = [None] * len(self.reference.species)
        used_candidates: set[int] = set()
        matches: list[tuple[int, ...]] = []

        def backtrack(order_position: int) -> None:
            if stop_after is not None and len(matches) >= stop_after:
                return
            if order_position == len(reference_order):
                matches.append(tuple(int(value) for value in reference_to_candidate))
                return

            ref_index = reference_order[order_position]
            label = self.reference.labels[ref_index]
            for candidate_index in candidate_by_label[label]:
                if candidate_index in used_candidates:
                    continue
                if not self._is_partial_distance_match(
                    candidate,
                    ref_index,
                    candidate_index,
                    reference_to_candidate,
                ):
                    continue
                reference_to_candidate[ref_index] = candidate_index
                used_candidates.add(candidate_index)
                backtrack(order_position + 1)
                used_candidates.remove(candidate_index)
                reference_to_candidate[ref_index] = None

        backtrack(0)
        return matches

    def _reference_assignment_order(
        self,
        candidate_by_label: dict[tuple[str, str], list[int]],
    ) -> list[int]:
        label_counts = {
            label: len(indices) for label, indices in candidate_by_label.items()
        }

        def key(index: int) -> tuple[int, tuple[float, ...], int]:
            row = tuple(np.round(np.sort(self.reference.distance_matrix[index]), 8))
            return (label_counts[self.reference.labels[index]], row, index)

        return sorted(range(len(self.reference.species)), key=key)

    def _is_partial_distance_match(
        self,
        candidate: Cluster,
        ref_index: int,
        candidate_index: int,
        reference_to_candidate: Sequence[int | None],
    ) -> bool:
        for other_ref_index, other_candidate_index in enumerate(reference_to_candidate):
            if other_candidate_index is None:
                continue
            if not np.isclose(
                self.reference.distance_matrix[ref_index, other_ref_index],
                candidate.distance_matrix[candidate_index, other_candidate_index],
                rtol=self.rtol,
                atol=self.atol,
            ):
                return False
        return True

    def _nearest_permutation(
        self,
        candidate: Cluster,
    ) -> tuple[tuple[int, ...], float]:
        candidate_by_label: dict[tuple[str, str], list[int]] = {}
        for index, label in enumerate(candidate.labels):
            candidate_by_label.setdefault(label, []).append(index)

        label_groups = []
        for label in sorted(candidate_by_label):
            reference_indices = [
                index
                for index, ref_label in enumerate(self.reference.labels)
                if ref_label == label
            ]
            candidate_permutations = itertools.permutations(candidate_by_label[label])
            label_groups.append((reference_indices, candidate_permutations))

        best_permutation: list[int] | None = None
        best_score = float("inf")
        for group_permutations in itertools.product(
            *(group[1] for group in label_groups)
        ):
            reference_to_candidate = [0] * len(self.reference.species)
            for (reference_indices, _), candidate_indices in zip(
                label_groups,
                group_permutations,
            ):
                for reference_index, candidate_index in zip(
                    reference_indices,
                    candidate_indices,
                ):
                    reference_to_candidate[reference_index] = candidate_index
            score = float(
                np.sum(
                    np.abs(
                        self.reference.distance_matrix
                        - candidate.distance_matrix[
                            np.ix_(reference_to_candidate, reference_to_candidate)
                        ]
                    )
                )
            )
            if score < best_score:
                best_score = score
                best_permutation = reference_to_candidate

        if best_permutation is None:
            raise ValueError("Could not find a nearest cluster permutation")
        return tuple(best_permutation), best_score

    def _build_match(
        self,
        candidate: Cluster,
        reference_to_candidate: tuple[int, ...],
        is_unique: bool,
        *,
        score: float,
    ) -> ClusterMatch:
        candidate_to_reference = [-1] * len(reference_to_candidate)
        for reference_index, candidate_index in enumerate(reference_to_candidate):
            candidate_to_reference[candidate_index] = reference_index

        ordered_candidate_indices = tuple(
            candidate.site_indices[candidate_index]
            for candidate_index in reference_to_candidate
        )
        ordered_candidate = Cluster(
            site_indices=ordered_candidate_indices,
            species=tuple(candidate.species[index] for index in reference_to_candidate),
            coords=tuple(candidate.coords[index] for index in reference_to_candidate),
            roles=tuple(candidate.roles[index] for index in reference_to_candidate),
            metadata=tuple(
                candidate.metadata[index] for index in reference_to_candidate
            ),
            distance_matrix=candidate.distance_matrix[
                np.ix_(reference_to_candidate, reference_to_candidate)
            ],
        )
        return ClusterMatch(
            reference_to_candidate=tuple(reference_to_candidate),
            candidate_to_reference=tuple(candidate_to_reference),
            ordered_candidate_indices=ordered_candidate_indices,
            fingerprint=ordered_candidate.fingerprint,
            is_unique=is_unique,
            score=score,
        )


def match_clusters(
    reference: Cluster,
    candidate: Cluster,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    require_unique: bool = False,
    find_nearest_if_fail: bool = False,
) -> ClusterMatch:
    """Match a candidate cluster to a reference cluster."""
    matcher = ClusterMatcher(reference, rtol=rtol, atol=atol)
    return matcher.match(
        candidate,
        require_unique=require_unique,
        find_nearest_if_fail=find_nearest_if_fail,
    )
