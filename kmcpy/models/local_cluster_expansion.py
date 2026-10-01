#!/usr/bin/env python
"""
This module provides classes and functions to build a Local Cluster Expansion (LCE) model for kinetic Monte Carlo (KMC) simulations, particularly for ionic conductors such as NaSICON materials. The main class, `LocalClusterExpansion`, reads a crystal structure file (e.g., CIF format), processes the structure to define a local migration unit, and generates clusters (points, pairs, triplets, quadruplets) within a specified cutoff. The clusters are grouped into orbits based on symmetry, and the resulting model can be serialized to JSON for use in KMC simulations.
"""
from itertools import combinations, product
from typing import Optional, Sequence
from pymatgen.core import Structure
import numpy as np
import json
import hashlib
import logging
import warnings
from kmcpy.models.base import BaseModel
from kmcpy.models.fitting.fitter import LCEFitter
from kmcpy.models.fitting.registry import register_fitter
from kmcpy.models.lce_kernels import (
    EMPTY_SITE_BASIS_VALUES,
    LCEKernelInputs,
    correlation,
    decorated_correlation,
    flatten_cluster_indices,
)
from kmcpy.event import Event
from kmcpy.simulator.state import State
from kmcpy.structure.local_lattice_structure import LocalLatticeStructure
from kmcpy.structure.cluster import Cluster, Orbit
from kmcpy.structure.local_site_order import (
    LocalSiteOrder,
    ordered_site_hash,
    ordered_site_signature,
)

logger = logging.getLogger(__name__) 
logging.getLogger('pymatgen').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

class LocalClusterExpansion(BaseModel):
    """Local cluster expansion of a scalar (E_KRA or a site-energy term) around a hop.

    Create the model with :meth:`build` from a ``LocalLatticeStructure`` or load
    it with :meth:`from_dict`/:meth:`from_file`, then attach fitted parameters
    with :meth:`set_parameters` or :meth:`load_parameters_from_file`. Fields that
    have not been set yet are ``None``.
    """

    def __init__(self, name: str = "LocalClusterExpansion"):
        super().__init__(name=name)
        self.name = name

        # Local environment (from build() or from_dict()).
        self.local_lattice_structure: LocalLatticeStructure | None = None  # build() only
        self.center_site = None
        self.local_env_structure = None
        self.basis = None
        self.local_site_order: LocalSiteOrder = LocalSiteOrder.resolve(None)
        self.local_environment_signature: list[dict] | None = None
        self.local_environment_hash: str | None = None

        # Clusters and correlation-vector terms.
        self.clusters: list[Cluster] | None = None  # build() only
        self.orbits: list[Orbit] | None = None
        self.orbit_fingerprints: list[str] | None = None
        self.basis_site_state_counts: list[int] | None = None
        self.cluster_site_indices = None  # numba typed List: [feature][cluster][site]
        self.correlation_basis_indices = None  # same shape, decorated features only
        self.site_basis_values: np.ndarray | None = None
        self.correlation_feature_metadata: list[dict] | None = None
        self.correlation_fingerprints: list[str] | None = None

        # Fitted parameters (from set_parameters()).
        self.keci: list[float] | None = None
        self.empty_cluster: float | None = None
        self.parameter_orbit_fingerprints: list[str] | None = None
        self.parameter_local_environment_hash: str | None = None
        self.parameter_local_site_order = None
        self._parameters = None

        # compute() caches; see _validate_keci_once() and _flat_cluster_indices().
        self._keci_cache = None
        self._flat_cluster_indices_cache = None

    def fit(self, *args, **kwargs):
        """Fit parameters and include this model's local-environment metadata."""
        orbit_fingerprints = self.orbit_fingerprints
        if orbit_fingerprints is None and self.orbits is not None:
            orbit_fingerprints = self.get_orbit_fingerprints()
        if orbit_fingerprints is not None:
            kwargs.setdefault("orbit_fingerprints", orbit_fingerprints)

        if self.local_environment_hash is not None:
            kwargs.setdefault("local_environment_hash", str(self.local_environment_hash))

        return super().fit(*args, **kwargs)

    def build(self, local_lattice_structure:LocalLatticeStructure, 
        cutoff_cluster: list = [6, 6, 6], **kwargs) -> None:
        """
        Build the LocalClusterExpansion model from a Structure object.

        The local-environment center is already defined by
        ``local_lattice_structure``. If that center was a structure site, it may
        be included or excluded according to the local site order. If the center
        was a fractional-coordinate point, it is only a geometric origin and is
        not itself an occupation-vector site.

        Args:
            local_lattice_structure (LocalLatticeStructure): Local environment object containing the structure and center site.
            cutoff_cluster (list, optional): Cutoff distances for clusters [pair, triplet, quadruplet]. Defaults to [6, 6, 6].

        Notes:
            If the next-step KMC is not based on the same LCE object generated in this step, be careful with two things:
            1) The Ekra generated in this step can be transferred to the KMC, provided the orbits are arranged in the same way as here.
            2) The cluster_site_indices must correspond exactly to the input structure used in the KMC step, which may require reconstruction of an LCE object using the same KMC-input structure.
        """
        logger.info("Initializing LocalClusterExpansion ...")
        self.name = "LocalClusterExpansion"
        self.local_lattice_structure = local_lattice_structure
        self.center_site = local_lattice_structure.center_site
        self.local_env_structure = local_lattice_structure.structure
        self.basis = local_lattice_structure.basis
        self.local_site_order = local_lattice_structure.local_site_order
        self.local_environment_signature = (
            local_lattice_structure.get_ordered_site_signature()
        )
        self.local_environment_hash = local_lattice_structure.get_ordered_site_hash()

        # List all possible point, pair and triplet clusters
        atom_index_list = np.arange(0, len(local_lattice_structure.structure))

        cluster_indexes = (
            list(combinations(atom_index_list, 1))
            + list(combinations(atom_index_list, 2))
            + list(combinations(atom_index_list, 3))
            + list(combinations(atom_index_list, 4))
        )

        logger.info(f"{len(cluster_indexes)} clusters will be generated ...")

        self.clusters = self.build_clusters(
            local_lattice_structure.structure, cluster_indexes, [10] + cutoff_cluster
        )

        self.orbits = self.build_orbits(self.clusters)
        self._configure_correlation_terms(local_lattice_structure)
        self.orbit_fingerprints = self.get_orbit_fingerprints()
        
                
        logger.info(
            "Type\tIndex\tmax_length\tmin_length\tPoint Group\tMultiplicity"
        )
        for orbit in self.orbits:
            orbit.show_representative_cluster()

    # Keys written by as_dict(); from_dict() decodes exactly these.
    PAYLOAD_KEYS = frozenset({
        "name",
        "basis",
        "orbits",
        "orbit_fingerprints",
        "cluster_site_indices",
        "center_site",
        "migration_unit_structure",
        "basis_site_state_counts",
        "correlation_basis_indices",
        "site_basis_values",
        "correlation_feature_metadata",
        "correlation_fingerprints",
        "local_site_order",
        "local_environment_signature",
        "local_environment_hash",
    })
    # Older payload keys: renamed, or no longer used by the model.
    RENAMED_PAYLOAD_KEYS = {"MigrationUnit_structure": "migration_unit_structure"}
    UNUSED_PAYLOAD_KEYS = frozenset({"clusters", "template_structure"})

    @classmethod
    def from_dict(cls, data: dict):
        """
        Load a LocalClusterExpansion object from a dictionary payload.

        Args:
            data: Dictionary containing serialized LocalClusterExpansion data.

        Returns:
            LocalClusterExpansion: The loaded LocalClusterExpansion object
        """
        from pymatgen.core.sites import PeriodicSite
        from pymatgen.core.structure import Molecule
        from kmcpy.structure.basis import BasisFunction, ChebyshevBasis

        data = cls._migrate_payload(data)
        obj = cls(name=data.get("name") or cls.__name__)

        if data.get("basis") is not None:
            obj.basis = BasisFunction.from_dict(data["basis"])
        else:
            state_counts = data.get("basis_site_state_counts") or [2]
            obj.basis = ChebyshevBasis(max_states=max(2, max(state_counts)))
        obj.basis_site_state_counts = data.get("basis_site_state_counts")
        obj.local_site_order = LocalSiteOrder.resolve(data.get("local_site_order"))

        if "orbits" in data:
            obj.orbits = [Orbit.from_dict(orbit_data) for orbit_data in data["orbits"]]
        if "center_site" in data:
            obj.center_site = PeriodicSite.from_dict(data["center_site"])
        structure_data = data.get("migration_unit_structure")
        if structure_data is not None:
            structure_class = (
                Molecule if structure_data.get("@class") == "Molecule" else Structure
            )
            obj.local_env_structure = structure_class.from_dict(structure_data)

        if "cluster_site_indices" in data:
            obj.cluster_site_indices = _to_numba_cluster_site_indices(
                data["cluster_site_indices"]
            )
        if data.get("correlation_basis_indices") is not None:
            obj.correlation_basis_indices = _to_numba_cluster_basis_indices(
                data["correlation_basis_indices"]
            )
        if data.get("site_basis_values") is not None:
            obj.site_basis_values = np.asarray(data["site_basis_values"], dtype=float)
        obj.correlation_feature_metadata = data.get("correlation_feature_metadata")
        if (
            getattr(obj.basis, "uses_state_indices", False)
            and obj.cluster_site_indices is not None
            and not obj._is_decorated()
        ):
            raise ValueError(
                "Chebyshev LocalClusterExpansion model files must include "
                "correlation_basis_indices and site_basis_values. Regenerate "
                "or resave the model with the current kMCpy schema."
            )

        obj.local_environment_signature = data.get("local_environment_signature")
        if obj.local_environment_signature is None and obj.local_env_structure is not None:
            obj.local_environment_signature = ordered_site_signature(obj.local_env_structure)
        obj.local_environment_hash = data.get("local_environment_hash")
        if obj.local_environment_hash is None and obj.local_environment_signature is not None:
            obj.local_environment_hash = ordered_site_hash(obj.local_environment_signature)

        obj.correlation_fingerprints = data.get("correlation_fingerprints")
        if obj.orbits is not None:
            obj._restore_correlation_fingerprints(data.get("orbit_fingerprints"))
        return obj

    @classmethod
    def _migrate_payload(cls, data: dict) -> dict:
        """Rename older payload keys and drop unused ones; warn about unknown keys."""
        migrated = {}
        unknown = []
        for key, value in data.items():
            key = cls.RENAMED_PAYLOAD_KEYS.get(key, key)
            if key.startswith("@") or key in cls.UNUSED_PAYLOAD_KEYS:
                continue
            if key not in cls.PAYLOAD_KEYS:
                unknown.append(key)
                continue
            migrated[key] = value
        if unknown:
            warnings.warn(
                f"Ignoring unknown LocalClusterExpansion payload keys: {sorted(unknown)}",
                UserWarning,
                stacklevel=3,
            )
        return migrated

    def _restore_correlation_fingerprints(self, stored_orbit_fingerprints) -> None:
        """Rebuild correlation fingerprints and check them against the stored ones."""
        if self.correlation_fingerprints is None:
            if self.correlation_feature_metadata is not None:
                self.correlation_fingerprints = [
                    self._decorated_feature_fingerprint(
                        self.orbits[int(metadata["orbit_index"])],
                        metadata,
                    )
                    for metadata in self.correlation_feature_metadata
                ]
            elif (
                stored_orbit_fingerprints is not None
                and len(stored_orbit_fingerprints) != len(self.orbits)
            ):
                self.correlation_fingerprints = [
                    str(value) for value in stored_orbit_fingerprints
                ]
        expected_orbit_fingerprints = self.get_orbit_fingerprints()
        if (
            stored_orbit_fingerprints is not None
            and list(stored_orbit_fingerprints) != expected_orbit_fingerprints
        ):
            raise ValueError(
                "Serialized LocalClusterExpansion orbit_fingerprints do not "
                "match reconstructed correlation features."
            )
        self.orbit_fingerprints = expected_orbit_fingerprints

    @staticmethod
    def _iter_cluster_site_indices(cluster_site_indices):
        for orbit in cluster_site_indices:
            for cluster in orbit:
                for site_idx in cluster:
                    yield int(site_idx)

    def get_orbit_fingerprints(self) -> list[str]:
        """Return orbit fingerprints in the same order as the correlation vector."""
        if self.correlation_fingerprints is not None:
            return [str(value) for value in self.correlation_fingerprints]
        if self.orbits is None:
            return []
        return [orbit.fingerprint for orbit in self.orbits]

    def _validate_parameter_orbits(
        self,
        keci,
        orbit_fingerprints=None,
        local_environment_hash=None,
    ) -> list[float]:
        """Validate that ECI values are aligned with this model's orbit order."""
        keci_values = list(keci)
        expected_orbit_fingerprints = self.get_orbit_fingerprints()
        if expected_orbit_fingerprints and len(keci_values) != len(expected_orbit_fingerprints):
            raise ValueError(
                "keci length does not match LocalClusterExpansion feature count: "
                f"{len(keci_values)} != {len(expected_orbit_fingerprints)}"
            )
        if expected_orbit_fingerprints and orbit_fingerprints is None:
            warnings.warn(
                "Parameter payload is missing orbit_fingerprints; keci values "
                "were only validated by length. Regenerate the parameter or "
                "model file to bind ECIs to orbit fingerprints.",
                UserWarning,
                stacklevel=3,
            )

        expected_local_environment_hash = self.local_environment_hash
        if expected_local_environment_hash and local_environment_hash is None:
            warnings.warn(
                "Parameter payload is missing local_environment_hash; keci "
                "values were not tied to a specific ordered local environment. "
                "Regenerate the parameter or model file with local-environment "
                "metadata.",
                UserWarning,
                stacklevel=3,
            )
        elif (
            expected_local_environment_hash
            and str(local_environment_hash) != str(expected_local_environment_hash)
        ):
            raise ValueError(
                "Parameter local_environment_hash does not match this "
                "LocalClusterExpansion local environment."
            )

        if orbit_fingerprints is not None:
            normalized_orbit_fingerprints = [str(value) for value in orbit_fingerprints]
            if len(normalized_orbit_fingerprints) != len(expected_orbit_fingerprints):
                raise ValueError(
                    "orbit_fingerprints length does not match "
                    "LocalClusterExpansion feature count: "
                    f"{len(normalized_orbit_fingerprints)} != "
                    f"{len(expected_orbit_fingerprints)}"
                )
            if normalized_orbit_fingerprints != expected_orbit_fingerprints:
                raise ValueError(
                    "Parameter orbit_fingerprints do not match this "
                    "LocalClusterExpansion orbit order."
                )
        return keci_values

    def validate_reference_lattice_structure(
        self,
        reference_local_lattice_structure: LocalLatticeStructure,
    ) -> None:
        """
        Validate that a reference local lattice has the same site order as the model.

        Args:
            reference_local_lattice_structure: Reference used to map structures
                into occupation vectors.

        Raises:
            ValueError: If the reference order is incompatible with this model.
        """
        if self.cluster_site_indices is None:
            raise ValueError("LocalClusterExpansion model must define cluster_site_indices.")

        model_hash = self.local_environment_hash
        if model_hash and hasattr(reference_local_lattice_structure, "get_ordered_site_hash"):
            reference_hash = reference_local_lattice_structure.get_ordered_site_hash()
            if reference_hash != model_hash:
                raise ValueError(
                    "Reference LocalLatticeStructure order does not match "
                    "the LocalClusterExpansion model."
                )

        cluster_site_indices = list(
            self._iter_cluster_site_indices(self.cluster_site_indices)
        )
        if cluster_site_indices and (
            min(cluster_site_indices) < 0
            or max(cluster_site_indices) >= len(reference_local_lattice_structure.site_indices)
        ):
            raise ValueError(
                "LocalClusterExpansion cluster_site_indices are incompatible "
                "with the reference LocalLatticeStructure site order."
            )

    def get_occ_corr_from_structure(
        self,
        structure: Structure,
        reference_local_lattice_structure: Optional[LocalLatticeStructure] = None,
        tol=1e-2,
        angle_tol=5,
    ):
        """
        Compute occupation and correlation vectors from a structure.

        Args:
            structure: Structure to featurize.
            reference_local_lattice_structure: Reference local lattice used to
                map structure sites into the model's local site order. If omitted,
                the model must carry ``local_lattice_structure`` from ``build``.
            tol: Structure matching tolerance.
            angle_tol: Structure matching angle tolerance.

        Returns:
            tuple: ``(occupation, correlation)``.
        """
        reference = reference_local_lattice_structure or self.local_lattice_structure
        if reference is None:
            raise ValueError(
                "Cannot compute correlation from structure without a reference "
                "LocalLatticeStructure. Pass reference_local_lattice_structure "
                "or build the model in memory first."
            )

        self.validate_reference_lattice_structure(reference)


        structure_for_occ = structure.copy()
        active_site_order = getattr(reference, "active_site_order", None)
        if active_site_order is not None:
            structure_for_occ = active_site_order.filter_active_structure(
                structure_for_occ, tol=tol
            )
        structure_for_occ.remove_oxidation_states()

        occ = reference.get_occ_from_structure(
            structure_for_occ,
            tol=tol,
            angle_tol=angle_tol,
        )
        local_occ = occ[reference.site_indices]
        corr = np.empty(shape=len(self.cluster_site_indices))
        self._calculate_correlation(corr, local_occ.data)
        return occ, corr

    def get_corr_from_structure(
        self,
        structure: Structure,
        reference_local_lattice_structure: Optional[LocalLatticeStructure] = None,
        tol=1e-2,
        angle_tol=5,
    ):
        '''get_corr_from_structure() returns a correlation numpy array.
        '''
        _, corr = self.get_occ_corr_from_structure(
            structure,
            reference_local_lattice_structure=reference_local_lattice_structure,
            tol=tol,
            angle_tol=angle_tol,
        )
        return corr
    
    def build_clusters(self, local_env_structure, indexes, cutoff):  # return a list of Cluster
        clusters = []
        logger.info("\nGenerating possible clusters within this migration unit...")
        logger.info(
            "Cutoffs: pair = %s Angst, triplet = %s Angst, quadruplet = %s Angst",
            cutoff[1],
            cutoff[2],
            cutoff[3],
        )
        for site_indices in indexes:
            sites = [local_env_structure[s] for s in site_indices]
            cluster = Cluster(site_indices, sites, analyze_symmetry=True)
            if cluster.max_length < cutoff[len(cluster.site_indices) - 1]:
                clusters.append(cluster)
        return clusters

    def build_orbits(self, clusters):
        """
        return a list of Orbit

        For each orbit, loop over clusters
            for each cluster, check if this cluster exists in this orbit
                if not, attach the cluster to orbit
                else,
        """
        orbit_clusters = []
        grouped_clusters = []
        for i in clusters:
            if i not in orbit_clusters:
                orbit_clusters.append(i)
                grouped_clusters.append([i])
            else:
                grouped_clusters[orbit_clusters.index(i)].append(i)
        orbits = []
        for i in grouped_clusters:
            orbit = Orbit()
            for cl in i:
                orbit.attach_cluster(cl)
            orbits.append(orbit)
        return orbits

    def _configure_correlation_terms(
        self,
        local_lattice_structure: LocalLatticeStructure,
    ) -> None:
        """Configure correlation-vector terms for the selected site basis."""
        local_site_indices = [int(index) for index in local_lattice_structure.site_indices]
        self.basis_site_state_counts = [
            len(local_lattice_structure.allowed_species[site_index])
            for site_index in local_site_indices
        ]

        if not getattr(self.basis, "uses_state_indices", False):
            self.cluster_site_indices = _to_numba_cluster_site_indices(
                [
                    [cluster.site_indices for cluster in orbit.clusters]
                    for orbit in self.orbits
                ]
            )
            self.correlation_basis_indices = None
            self.site_basis_values = None
            self.correlation_feature_metadata = None
            self.correlation_fingerprints = [orbit.fingerprint for orbit in self.orbits]
            return

        self.site_basis_values = self._build_site_basis_values(
            self.basis_site_state_counts
        )
        cluster_site_indices = []
        cluster_basis_indices = []
        feature_metadata = []
        feature_fingerprints = []

        for orbit_index, orbit in enumerate(self.orbits):
            representative = orbit.clusters[0]
            site_count = len(representative.site_indices)
            basis_counts = []
            for position in range(site_count):
                counts_for_position = [
                    self.basis.num_site_basis_functions(
                        self.basis_site_state_counts[int(cluster.site_indices[position])]
                    )
                    for cluster in orbit.clusters
                ]
                basis_counts.append(min(counts_for_position))

            for decoration in product(*(range(count) for count in basis_counts)):
                decoration = tuple(int(index) for index in decoration)
                cluster_site_indices.append(
                    [cluster.site_indices for cluster in orbit.clusters]
                )
                cluster_basis_indices.append(
                    [decoration for _cluster in orbit.clusters]
                )
                site_state_counts = [
                    self.basis_site_state_counts[int(site_index)]
                    for site_index in representative.site_indices
                ]
                metadata = {
                    "orbit_index": int(orbit_index),
                    "basis_indices": list(decoration),
                    "site_state_counts": site_state_counts,
                }
                feature_metadata.append(metadata)
                feature_fingerprints.append(
                    self._decorated_feature_fingerprint(orbit, metadata)
                )

        self.cluster_site_indices = _to_numba_cluster_site_indices(cluster_site_indices)
        self.correlation_basis_indices = _to_numba_cluster_basis_indices(
            cluster_basis_indices
        )
        self.correlation_feature_metadata = feature_metadata
        self.correlation_fingerprints = feature_fingerprints

    def _build_site_basis_values(self, site_state_counts: Sequence[int]) -> np.ndarray:
        """Return padded per-local-site basis lookup values."""
        max_states = max(int(count) for count in site_state_counts)
        max_basis_count = max(
            self.basis.num_site_basis_functions(int(count))
            for count in site_state_counts
        )
        values = np.zeros(
            (len(site_state_counts), max_states, max_basis_count),
            dtype=float,
        )
        for site_index, n_states in enumerate(site_state_counts):
            site_values = self.basis.site_basis_values(int(n_states))
            values[
                site_index,
                : site_values.shape[0],
                : site_values.shape[1],
            ] = site_values
        return values

    @staticmethod
    def _decorated_feature_fingerprint(orbit: Orbit, metadata: dict) -> str:
        """Return a stable fingerprint for a decorated orbit feature."""
        payload = {
            "orbit_fingerprint": orbit.fingerprint,
            "basis_indices": metadata["basis_indices"],
            "site_state_counts": metadata["site_state_counts"],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _calculate_correlation(self, corr: np.ndarray, occupation: np.ndarray) -> None:
        """Fill a correlation vector for an occupation array."""
        decorated = self._is_decorated()
        orbit_offsets, cluster_offsets, sites, basis = self._flat_cluster_indices(
            self.correlation_basis_indices if decorated else None
        )
        if decorated:
            decorated_correlation(
                corr,
                occupation.astype(np.int64),
                orbit_offsets,
                cluster_offsets,
                sites,
                basis,
                self.site_basis_values,
            )
        else:
            correlation(corr, occupation, orbit_offsets, cluster_offsets, sites)

    def _is_decorated(self) -> bool:
        """Return whether correlations use per-site basis decorations (Chebyshev)."""
        return (
            self.correlation_basis_indices is not None
            and self.site_basis_values is not None
        )

    def _flat_cluster_indices(self, correlation_basis_indices=None):
        """Return cached flat (CSR-style) arrays for the nested cluster indices.

        Passing nested ``numba.typed.List`` objects into an ``njit`` function
        costs more than the correlation calculation itself, so the nested
        indices are flattened once per assignment of ``cluster_site_indices``
        or ``correlation_basis_indices``.
        """
        cluster_site_indices = self.cluster_site_indices
        cache = self._flat_cluster_indices_cache
        if (
            cache is not None
            and cache[0] is cluster_site_indices
            and cache[1] is correlation_basis_indices
        ):
            return cache[2]

        flat = flatten_cluster_indices(cluster_site_indices, correlation_basis_indices)
        self._flat_cluster_indices_cache = (
            cluster_site_indices,
            correlation_basis_indices,
            flat,
        )
        return flat

    def compute(self, simulation_state:State, event:Event):
        """
        Compute the fitted scalar value for this event/local environment.
        
        ``LocalClusterExpansion`` does not distinguish barrier and
        site-energy-difference roles internally. A model passed as the
        ``kra_model`` in ``CompositeLCEModel`` returns ``E_KRA``. A model passed
        as the ``site_model`` returns the site-energy-difference contribution
        expected by that composite model.
        
        Args:
            simulation_state: State object containing occupation vector (preferred)
            event: Event object containing mobile ion indices (required for local environment)
            
        Returns:
            float: The fitted scalar value
        """
        if not self.has_parameters():
            raise ValueError("No stored parameters found. Call set_parameters() or load_parameters_from_file() first.")
        correlation_count = self._validate_keci_once()

        if simulation_state is None:
            raise ValueError("simulation_state is required")

        # Require event for local environment determination
        if event is None:
            raise ValueError("Event object is required for LocalClusterExpansion.compute() to determine local environment")

        # Initialize correlation array using stored cluster_site_indices
        corr = np.empty(shape=correlation_count)

        # Gather only the local-environment occupations; copying the full
        # occupation vector here would make every rate evaluation O(N_sites).
        occupations = simulation_state.occupations
        occ_sublat = np.array(
            [occupations[site_index] for site_index in event.local_env_indices]
        )

        self._calculate_correlation(corr, occ_sublat)

        # Compute energy using stored parameters
        result = np.inner(corr, self.keci) + self.empty_cluster
        return result

    def _validate_keci_once(self) -> int:
        """Validate ``keci`` and return the correlation-vector length.

        ``compute`` runs for every dependent event on every KMC step, so the
        orbit-order validation is only repeated when ``keci``, its parameter
        metadata, or ``cluster_site_indices`` are reassigned.
        """
        orbit_fingerprints = self.parameter_orbit_fingerprints
        local_environment_hash = self.parameter_local_environment_hash
        cluster_site_indices = self.cluster_site_indices
        cache = self._keci_cache
        if (
            cache is not None
            and cache[0] is self.keci
            and cache[1] is orbit_fingerprints
            and cache[2] == local_environment_hash
            and cache[3] is cluster_site_indices
        ):
            return cache[4]

        self.keci = self._validate_parameter_orbits(
            self.keci,
            orbit_fingerprints,
            local_environment_hash,
        )
        correlation_count = len(cluster_site_indices)
        self._keci_cache = (
            self.keci,
            orbit_fingerprints,
            local_environment_hash,
            cluster_site_indices,
            correlation_count,
        )
        return correlation_count

    def kernel_inputs(self) -> LCEKernelInputs:
        """Return this model's validated parameters as arrays for batch kernels."""
        correlation_count = self._validate_keci_once()
        decorated = self._is_decorated()
        orbit_offsets, cluster_offsets, sites, basis = self._flat_cluster_indices(
            self.correlation_basis_indices if decorated else None
        )
        site_basis_values = self.site_basis_values if decorated else EMPTY_SITE_BASIS_VALUES
        return LCEKernelInputs(
            decorated,
            correlation_count,
            orbit_offsets,
            cluster_offsets,
            sites,
            basis,
            np.asarray(site_basis_values, dtype=np.float64),
            np.asarray(self.keci, dtype=np.float64),
            float(self.empty_cluster),
        )

    def has_parameters(self) -> bool:
        """Return whether fitted ``keci`` and ``empty_cluster`` are set."""
        return self.keci is not None and self.empty_cluster is not None

    def set_parameters(self, parameters):
        """
        Set fitted parameters for this LocalClusterExpansion model.
        
        Args:
            parameters: LCEModelParameters object or dict containing keci and empty_cluster
        """
        from kmcpy.models.parameters import LCEModelParameters
        
        if isinstance(parameters, LCEModelParameters):
            keci = parameters.keci
            empty_cluster = parameters.empty_cluster
            orbit_fingerprints = getattr(parameters, "orbit_fingerprints", None)
            local_environment_hash = getattr(parameters, "local_environment_hash", None)
            local_site_order = getattr(parameters, "local_site_order", None)
        elif isinstance(parameters, dict):
            keci = parameters['keci']
            empty_cluster = parameters['empty_cluster']
            orbit_fingerprints = parameters.get("orbit_fingerprints")
            local_environment_hash = parameters.get("local_environment_hash")
            local_site_order = parameters.get("local_site_order")
        else:
            raise TypeError("Parameters must be LCEModelParameters object or dict")

        self.keci = self._validate_parameter_orbits(
            keci,
            orbit_fingerprints,
            local_environment_hash,
        )
        self.empty_cluster = empty_cluster
        self.parameter_orbit_fingerprints = (
            [str(value) for value in orbit_fingerprints]
            if orbit_fingerprints is not None
            else self.get_orbit_fingerprints()
        )
        self.parameter_local_environment_hash = (
            str(local_environment_hash)
            if local_environment_hash is not None
            else self.local_environment_hash
        )
        self.parameter_local_site_order = local_site_order
        self._parameters = parameters
        
        logger.info(f"Parameters set for LocalClusterExpansion: keci length={len(self.keci)}, empty_cluster={self.empty_cluster}")

    def load_parameters_from_file(self, filename: str):
        """
        Load fitted parameters from a JSON file.
        
        Args:
            filename: Path to the JSON file containing LCE parameters
        """
        from kmcpy.models.parameters import LCEModelParameters
        
        parameters = LCEModelParameters.from_file(filename)
        self.set_parameters(parameters)

    def write_representative_clusters(self, filename='representative_clusters.txt'):
        """
        Write representative clusters to a text file.
        
        Args:
            filename: Name of the output file
        """
        with open(filename, 'w') as f:
            f.write("Representative Clusters for LocalClusterExpansion\n")
            f.write("=" * 50 + "\n\n")
            f.write("Type\tIndex\tmax_length\tmin_length\tPoint Group\tMultiplicity\n")
            for i, orbit in enumerate(self.orbits):
                cluster = orbit.clusters[0]  # representative cluster
                f.write(f"{cluster.type}\t{i}\t{cluster.max_length:.3f}\t{cluster.min_length:.3f}\t{cluster.sym}\t{orbit.multiplicity}\n")
        logger.info(f"Representative clusters written to {filename}")
    
    def __str__(self):
        """String representation of the LocalClusterExpansion."""
        lines = [
            f"LocalClusterExpansion: {self.name}",
            f"Number of orbits: {self._orbit_count()}",
            f"Local environment sites: {self._local_site_count()}",
        ]
        if self.center_site is not None:
            lines.append(
                f"Center site: {self.center_site.species} at {self.center_site.frac_coords}"
            )
        return "\n".join(lines)

    def __repr__(self):
        """Detailed representation of the LocalClusterExpansion."""
        return (
            f"LocalClusterExpansion(orbits={self._orbit_count()}, "
            f"sites={self._local_site_count()})"
        )

    def _orbit_count(self) -> int:
        return len(self.orbits) if self.orbits is not None else 0

    def _local_site_count(self) -> int:
        return len(self.local_env_structure) if self.local_env_structure is not None else 0
    
    def as_dict(self):
        """
        Return a dictionary representation of the LocalClusterExpansion.
        """
        cluster_site_indices = []
        if self.cluster_site_indices is not None:
            # Normalize possible numba TypedList payloads to plain nested Python lists.
            cluster_site_indices = [
                [[int(site_idx) for site_idx in cluster] for cluster in orbit]
                for orbit in self.cluster_site_indices
            ]
        correlation_basis_indices = None
        if self.correlation_basis_indices is not None:
            correlation_basis_indices = [
                [[int(index) for index in cluster] for cluster in feature]
                for feature in self.correlation_basis_indices
            ]

        payload = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "name": self.name,
            "basis": self.basis.as_dict() if hasattr(self.basis, "as_dict") else None,
            "orbits": [orbit.as_dict() for orbit in self.orbits],
            "orbit_fingerprints": self.get_orbit_fingerprints(),
            "cluster_site_indices": cluster_site_indices,
            "center_site": self.center_site.as_dict(),
            "migration_unit_structure": self.local_env_structure.as_dict()
        }
        if self.basis_site_state_counts is not None:
            payload["basis_site_state_counts"] = [
                int(count) for count in self.basis_site_state_counts
            ]
        if correlation_basis_indices is not None:
            payload["correlation_basis_indices"] = correlation_basis_indices
        if self.site_basis_values is not None:
            payload["site_basis_values"] = self.site_basis_values.tolist()
        if self.correlation_feature_metadata is not None:
            payload["correlation_feature_metadata"] = self.correlation_feature_metadata
        payload["local_site_order"] = self.local_site_order.as_dict()
        if self.local_environment_signature is not None:
            payload["local_environment_signature"] = self.local_environment_signature
        if self.local_environment_hash is not None:
            payload["local_environment_hash"] = self.local_environment_hash
        return payload


register_fitter(LocalClusterExpansion, LCEFitter)

def _to_numba_cluster_site_indices(cluster_site_indices):
    """Convert nested cluster site indices into numba typed lists."""
    from numba.typed import List

    return List(
        [
            List([List([int(site_idx) for site_idx in cluster]) for cluster in orbit])
            for orbit in cluster_site_indices
        ]
    )


def _to_numba_cluster_basis_indices(cluster_basis_indices):
    """Convert nested cluster basis-function indices into numba typed lists."""
    from numba.typed import List

    return List(
        [
            List([List([int(basis_idx) for basis_idx in cluster]) for cluster in feature])
            for feature in cluster_basis_indices
        ]
    )
