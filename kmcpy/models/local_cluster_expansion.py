#!/usr/bin/env python
"""
This module provides classes and functions to build a Local Cluster Expansion (LCE) model for kinetic Monte Carlo (KMC) simulations, particularly for ionic conductors such as NaSICON materials. The main class, `LocalClusterExpansion`, reads a crystal structure file (e.g., CIF format), processes the structure to define a local migration unit, and generates clusters (points, pairs, triplets, quadruplets) within a specified cutoff. The clusters are grouped into orbits based on symmetry, and the resulting model can be serialized to JSON for use in KMC simulations.
"""
from itertools import combinations, product
from typing import TYPE_CHECKING, Optional, Sequence
from pymatgen.core import Structure
import numpy as np
import json
import hashlib
import logging
import warnings
from monty.serialization import loadfn
from kmcpy.models.base import BaseModel
from kmcpy.models.fitting.fitter import LCEFitter
from kmcpy.models.fitting.registry import register_fitter
from copy import deepcopy
from kmcpy.event import Event
from kmcpy.simulator.state import State
from kmcpy.structure.local_lattice_structure import LocalLatticeStructure
from kmcpy.structure.cluster import Cluster, Orbit
from kmcpy.structure.local_site_ordering import (
    LocalSiteOrderingConvention,
    ordered_site_hash,
    ordered_site_signature,
)
import numba as nb

if TYPE_CHECKING:
    from kmcpy.simulator.config import Configuration

logger = logging.getLogger(__name__) 
logging.getLogger('pymatgen').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

class LocalClusterExpansion(BaseModel):
    """
    LocalClusterExpansion will be initialized with a template structure where all the sites are occupied
    cutoff_cluster is the cutoff for pairs and triplet
    cutoff_region is the cutoff for generating local cluster region
    """
    def __init__(self):
        """
        Initialization of the LocalClusterExpansion object.
        """
        self.name = "LocalClusterExpansion"

    def fit(self, *args, **kwargs):
        """Fit parameters and include this model's compatibility metadata."""
        orbit_fingerprints = getattr(self, "orbit_fingerprints", None)
        if orbit_fingerprints is None and hasattr(self, "orbits"):
            orbit_fingerprints = self.get_orbit_fingerprints()
        if orbit_fingerprints is not None:
            kwargs.setdefault("orbit_fingerprints", orbit_fingerprints)

        local_environment_hash = getattr(self, "local_environment_hash", None)
        if local_environment_hash is not None:
            kwargs.setdefault("local_environment_hash", str(local_environment_hash))

        return super().fit(*args, **kwargs)

    def build(self, local_lattice_structure:LocalLatticeStructure, 
        cutoff_cluster: list = [6, 6, 6], **kwargs) -> None:
        """
        Build the LocalClusterExpansion model from a Structure object.

        There are 2 ways to define the local environment (migration unit):
        1) Use the center of the mobile ion as the center of the local environment (default, center_frac_coord = None), this mobile ion is excluded from the local environment.
        2) Use a dummy site as the center of the local environment (set center_frac_coord).

        Args:
            local_lattice_structure (LocalLatticeStructure): Local environment object containing the structure and center site.
            cutoff_cluster (list, optional): Cutoff distances for clusters [pair, triplet, quadruplet]. Defaults to [6, 6, 6].
            center_frac_coord (list, optional): Fractional coordinates of the center of the local environment. If empty, uses the mobile ion site. Defaults to [].
            exclude_site_with_identifier (list, optional): List of site identifiers to exclude from the local environment. Defaults to [].

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
        self.ordering_convention = local_lattice_structure.ordering_convention
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

    @classmethod
    def from_file(cls, filename: str):
        """
        Load a LocalClusterExpansion object from a serialized file.
        
        Args:
            filename: Path to the JSON file containing the LocalClusterExpansion data
            
        Returns:
            LocalClusterExpansion: The loaded LocalClusterExpansion object
        """
        return cls.from_dict(loadfn(filename, cls=None))

    @classmethod
    def from_dict(cls, data: dict):
        """
        Load a LocalClusterExpansion object from a dictionary payload.

        Args:
            data: Dictionary containing serialized LocalClusterExpansion data.

        Returns:
            LocalClusterExpansion: The loaded LocalClusterExpansion object
        """
        # Create a new instance without calling __init__
        obj = cls.__new__(cls)
        
        # Restore all attributes from the JSON data
        for key, value in data.items():
            if key.startswith('@'):
                # Skip metadata keys
                continue
            elif key == 'orbits':
                obj.orbits = [Orbit.from_dict(orbit_data) for orbit_data in value]
            elif key == 'center_site':
                # Reconstruct center_site from its dict representation
                from pymatgen.core.sites import PeriodicSite
                obj.center_site = PeriodicSite.from_dict(value)
            elif key == 'MigrationUnit_structure' or key == 'migration_unit_structure':
                # Reconstruct migration unit structure
                if value.get('@class') == 'Molecule':
                    from pymatgen.core.structure import Molecule
                    obj.local_env_structure = Molecule.from_dict(value)
                else:
                    from pymatgen.core.structure import Structure
                    obj.local_env_structure = Structure.from_dict(value)
            elif key == 'template_structure':
                # Reconstruct template structure if present
                from pymatgen.core.structure import Structure
                obj.template_structure = Structure.from_dict(value)
            elif key == 'basis':
                if value is None:
                    continue
                from kmcpy.structure.basis import BasisFunction

                obj.basis = BasisFunction.from_dict(value)
            elif key == 'ordering_convention':
                obj.ordering_convention = LocalSiteOrderingConvention.resolve(value)
            else:
                # For all other attributes, set them directly
                setattr(obj, key, value)
        
        # Convert cluster_site_indices to numba TypedList format if it exists
        if hasattr(obj, 'cluster_site_indices'):
            obj.cluster_site_indices = _to_numba_cluster_site_indices(
                obj.cluster_site_indices
            )
        if getattr(obj, "correlation_basis_indices", None) is not None:
            obj.correlation_basis_indices = _to_numba_cluster_basis_indices(
                obj.correlation_basis_indices
            )
        if getattr(obj, "site_basis_values", None) is not None:
            obj.site_basis_values = np.asarray(obj.site_basis_values, dtype=float)

        if not hasattr(obj, "basis"):
            from kmcpy.structure.basis import ChebyshevBasis

            basis_site_state_counts = getattr(obj, "basis_site_state_counts", None) or [2]
            max_states = max(2, max(basis_site_state_counts))
            obj.basis = ChebyshevBasis(max_states=max_states)

        if (
            getattr(obj.basis, "uses_state_indices", False)
            and hasattr(obj, "cluster_site_indices")
            and (
                getattr(obj, "correlation_basis_indices", None) is None
                or getattr(obj, "site_basis_values", None) is None
            )
        ):
            raise ValueError(
                "Chebyshev LocalClusterExpansion model files must include "
                "correlation_basis_indices and site_basis_values. Regenerate "
                "or resave the model with the current kMCpy schema."
            )

        # Minimal JSON payloads may not include `name`; keep serialization robust.
        if not getattr(obj, "name", None):
            obj.name = cls.__name__

        if not hasattr(obj, "ordering_convention"):
            obj.ordering_convention = LocalSiteOrderingConvention.resolve(None)
        if not hasattr(obj, "local_environment_signature") and hasattr(
            obj, "local_env_structure"
        ):
            obj.local_environment_signature = ordered_site_signature(
                obj.local_env_structure
            )
        if not hasattr(obj, "local_environment_hash") and hasattr(
            obj, "local_environment_signature"
        ):
            obj.local_environment_hash = ordered_site_hash(
                obj.local_environment_signature
            )

        if hasattr(obj, "orbits"):
            stored_orbit_fingerprints = getattr(obj, "orbit_fingerprints", None)
            if (
                not hasattr(obj, "correlation_fingerprints")
                and getattr(obj, "correlation_feature_metadata", None) is not None
            ):
                obj.correlation_fingerprints = [
                    obj._decorated_feature_fingerprint(
                        obj.orbits[int(metadata["orbit_index"])],
                        metadata,
                    )
                    for metadata in obj.correlation_feature_metadata
                ]
            elif (
                not hasattr(obj, "correlation_fingerprints")
                and stored_orbit_fingerprints is not None
                and len(stored_orbit_fingerprints) != len(obj.orbits)
            ):
                obj.correlation_fingerprints = [
                    str(value) for value in stored_orbit_fingerprints
                ]
            expected_orbit_fingerprints = obj.get_orbit_fingerprints()
            if (
                stored_orbit_fingerprints is not None
                and list(stored_orbit_fingerprints) != expected_orbit_fingerprints
            ):
                raise ValueError(
                    "Serialized LocalClusterExpansion orbit_fingerprints do not "
                    "match reconstructed correlation features."
                )
            obj.orbit_fingerprints = expected_orbit_fingerprints
        
        return obj

    @staticmethod
    def _iter_cluster_site_indices(cluster_site_indices):
        for orbit in cluster_site_indices:
            for cluster in orbit:
                for site_idx in cluster:
                    yield int(site_idx)

    def get_orbit_fingerprints(self) -> list[str]:
        """Return orbit fingerprints in the same order as the correlation vector."""
        if hasattr(self, "correlation_fingerprints"):
            return [str(value) for value in self.correlation_fingerprints]
        if not hasattr(self, "orbits"):
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

        expected_local_environment_hash = getattr(self, "local_environment_hash", None)
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
            ValueError: If the reference ordering is incompatible with this model.
        """
        if not hasattr(self, "cluster_site_indices"):
            raise ValueError("LocalClusterExpansion model must define cluster_site_indices.")

        model_hash = getattr(self, "local_environment_hash", None)
        if model_hash and hasattr(reference_local_lattice_structure, "get_ordered_site_hash"):
            reference_hash = reference_local_lattice_structure.get_ordered_site_hash()
            if reference_hash != model_hash:
                raise ValueError(
                    "Reference LocalLatticeStructure ordering does not match "
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
        exclude_species: Optional[Sequence[str]] = None,
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
            exclude_species: Removed legacy argument; use site_mapping fixed sites.
            tol: Structure matching tolerance.
            angle_tol: Structure matching angle tolerance.

        Returns:
            tuple: ``(occupation, correlation)``.
        """
        reference = reference_local_lattice_structure or getattr(
            self, "local_lattice_structure", None
        )
        if reference is None:
            raise ValueError(
                "Cannot compute correlation from structure without a reference "
                "LocalLatticeStructure. Pass reference_local_lattice_structure "
                "or build the model in memory first."
            )

        self.validate_reference_lattice_structure(reference)

        if exclude_species is not None:
            raise ValueError(
                "exclude_species is no longer supported; encode fixed sites in "
                "site_mapping with a single allowed species."
            )

        structure_for_occ = structure.copy()
        active_site_index_map = getattr(reference, "active_site_index_map", None)
        if active_site_index_map is not None:
            structure_for_occ = active_site_index_map.filter_active_structure(
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
        self._calculate_correlation(corr, local_occ.array)
        return occ, corr

    def get_corr_from_structure(
        self,
        structure: Structure,
        reference_local_lattice_structure: Optional[LocalLatticeStructure] = None,
        exclude_species: Optional[Sequence[str]] = None,
        tol=1e-2,
        angle_tol=5,
    ):
        '''get_corr_from_structure() returns a correlation numpy array.
        '''
        _, corr = self.get_occ_corr_from_structure(
            structure,
            reference_local_lattice_structure=reference_local_lattice_structure,
            exclude_species=exclude_species,
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
        if (
            getattr(self, "correlation_basis_indices", None) is not None
            and getattr(self, "site_basis_values", None) is not None
        ):
            _calc_corr_decorated(
                corr,
                occupation.astype(np.int64),
                self.cluster_site_indices,
                self.correlation_basis_indices,
                self.site_basis_values,
            )
        else:
            _calc_corr(corr, occupation, self.cluster_site_indices)

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
        # Check if parameters are stored
        if not hasattr(self, 'keci') or not hasattr(self, 'empty_cluster'):
            raise ValueError("No stored parameters found. Call set_parameters() or load_parameters_from_file() first.")
        self.keci = self._validate_parameter_orbits(
            self.keci,
            getattr(self, "parameter_orbit_fingerprints", None),
            getattr(self, "parameter_local_environment_hash", None),
        )
            
        if simulation_state is None:
            raise ValueError("simulation_state is required")

        occupations = np.array(simulation_state.occupations)
            
        # Require event for local environment determination
        if event is None:
            raise ValueError("Event object is required for LocalClusterExpansion.compute() to determine local environment")
            
        # Initialize correlation array using stored cluster_site_indices
        corr = np.empty(shape=len(self.cluster_site_indices))
        
        # Extract local occupation using event's local environment indices
        occ_sublat = deepcopy(occupations[list(event.local_env_indices)])
            
        self._calculate_correlation(corr, occ_sublat)
        
        # Compute energy using stored parameters
        result = np.inner(corr, self.keci) + self.empty_cluster
        return result

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
            ordering_convention = getattr(parameters, "ordering_convention", None)
        elif isinstance(parameters, dict):
            keci = parameters['keci']
            empty_cluster = parameters['empty_cluster']
            orbit_fingerprints = parameters.get("orbit_fingerprints")
            local_environment_hash = parameters.get("local_environment_hash")
            ordering_convention = parameters.get("ordering_convention")
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
            else getattr(self, "local_environment_hash", None)
        )
        self.parameter_ordering_convention = ordering_convention
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
            f"Number of orbits: {len(self.orbits)}",
            f"Local environment sites: {len(self.local_env_structure)}",
            f"Center site: {self.center_site.species} at {self.center_site.frac_coords}"
        ]
        return "\n".join(lines)
    
    def __repr__(self):
        """Detailed representation of the LocalClusterExpansion."""
        return f"LocalClusterExpansion(orbits={len(self.orbits)}, sites={len(self.local_env_structure)})"
    
    def as_dict(self):
        """
        Return a dictionary representation of the LocalClusterExpansion.
        """
        cluster_site_indices = []
        if hasattr(self, "cluster_site_indices"):
            # Normalize possible numba TypedList payloads to plain nested Python lists.
            cluster_site_indices = [
                [[int(site_idx) for site_idx in cluster] for cluster in orbit]
                for orbit in self.cluster_site_indices
            ]
        correlation_basis_indices = None
        if getattr(self, "correlation_basis_indices", None) is not None:
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
        if hasattr(self, "basis_site_state_counts"):
            payload["basis_site_state_counts"] = [
                int(count) for count in self.basis_site_state_counts
            ]
        if correlation_basis_indices is not None:
            payload["correlation_basis_indices"] = correlation_basis_indices
        if getattr(self, "site_basis_values", None) is not None:
            payload["site_basis_values"] = self.site_basis_values.tolist()
        if getattr(self, "correlation_feature_metadata", None) is not None:
            payload["correlation_feature_metadata"] = self.correlation_feature_metadata
        if hasattr(self, "ordering_convention"):
            payload["ordering_convention"] = self.ordering_convention.as_dict()
        if hasattr(self, "local_environment_signature"):
            payload["local_environment_signature"] = self.local_environment_signature
        if hasattr(self, "local_environment_hash"):
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


@nb.njit
def _calc_corr(corr, occ_latt, cluster_site_indices):
    """
    Calculate correlation function for cluster expansion.
    
    Args:
        corr: Output correlation array
        occ_latt: Occupation array for the lattice
        cluster_site_indices: Nested list structure [orbit][cluster][site]
    """
    i = 0
    for sublat_ind_orbit in cluster_site_indices: # loop through orbits
        corr[i] = 0
        for sublat_ind_cluster in sublat_ind_orbit: # loop through clusters in the orbit
            corr_cluster = 1
            for occ_site in sublat_ind_cluster:
                corr_cluster *= occ_latt[occ_site]
            corr[i] += corr_cluster
        i += 1


@nb.njit
def _calc_corr_decorated(
    corr,
    occ_latt,
    cluster_site_indices,
    cluster_basis_indices,
    site_basis_values,
):
    """
    Calculate decorated multicomponent correlation functions.

    ``occ_latt`` stores species-state indices. ``site_basis_values`` maps
    ``[local_site, state_index, basis_index]`` to the scalar basis value.
    """
    i = 0
    for sublat_ind_orbit in cluster_site_indices:
        corr[i] = 0.0
        basis_ind_orbit = cluster_basis_indices[i]
        for cluster_index, sublat_ind_cluster in enumerate(sublat_ind_orbit):
            corr_cluster = 1.0
            basis_ind_cluster = basis_ind_orbit[cluster_index]
            for site_position, occ_site in enumerate(sublat_ind_cluster):
                state_index = int(occ_latt[occ_site])
                basis_index = int(basis_ind_cluster[site_position])
                corr_cluster *= site_basis_values[occ_site, state_index, basis_index]
            corr[i] += corr_cluster
        i += 1
