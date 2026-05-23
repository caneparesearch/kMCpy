#!/usr/bin/env python
"""
This module provides classes and functions to build a Local Cluster Expansion (LCE) model for kinetic Monte Carlo (KMC) simulations, particularly for ionic conductors such as NaSICON materials. The main class, `LocalClusterExpansion`, reads a crystal structure file (e.g., CIF format), processes the structure to define a local migration unit, and generates clusters (points, pairs, triplets, quadruplets) within a specified cutoff. The clusters are grouped into orbits based on symmetry, and the resulting model can be serialized to JSON for use in KMC simulations.
"""
from itertools import combinations
from typing import TYPE_CHECKING, Optional, Sequence
from kmcpy.external.structure import StructureKMCpy
import numpy as np
import json
import logging
from kmcpy.models.base import BaseModel
from kmcpy.models.fitting.fitter import LCEFitter
from kmcpy.models.fitting.registry import register_fitter
from copy import deepcopy
from kmcpy.event import Event
from kmcpy.simulator.state import State
from kmcpy.structure.local_lattice_structure import LocalLatticeStructure
from kmcpy.structure.local_site_ordering import LocalSiteOrderingConvention
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

    def build(self, local_lattice_structure:LocalLatticeStructure, 
        cutoff_cluster: list = [6, 6, 6], **kwargs) -> None:
        """
        Build the LocalClusterExpansion model from a StructureKMCpy object.

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

        self.cluster_site_indices = _to_numba_cluster_site_indices([
            [cluster.site_indices for cluster in orbit.clusters]
            for orbit in self.orbits
        ])  # cluster_site_indices[orbit,cluster,site]
        
                
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
        with open(filename, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_json(cls, filename: str):
        """
        Compatibility alias for JSON model loading.
        """
        return cls.from_file(filename)

    @classmethod
    def from_dict(cls, data: dict):
        """
        Load a LocalClusterExpansion object from a dictionary payload.

        Args:
            data: Dictionary containing serialized LocalClusterExpansion data.

        Returns:
            LocalClusterExpansion: The loaded LocalClusterExpansion object
        """
        from kmcpy.models.cluster import Orbit, Cluster

        # Create a new instance without calling __init__
        obj = cls.__new__(cls)
        
        # Restore all attributes from the JSON data
        for key, value in data.items():
            if key.startswith('@'):
                # Skip metadata keys
                continue
            elif key == 'orbits':
                # Reconstruct orbits from the stored data
                obj.orbits = []
                for orbit_data in value:
                    orbit = Orbit()
                    orbit.multiplicity = orbit_data.get('multiplicity', 0)
                    
                    # Reconstruct clusters for each orbit
                    orbit.clusters = []
                    for cluster_data in orbit_data.get('clusters', []):
                        # Create cluster from stored data
                        from pymatgen.core.structure import Molecule
                        cluster = Cluster.__new__(Cluster)
                        cluster.site_indices = cluster_data.get('site_indices', [])
                        cluster.type = cluster_data.get('type', 'point')
                        cluster.structure = Molecule.from_dict(cluster_data.get('structure', {}))
                        cluster.sym = cluster_data.get('sym', '')
                        cluster.max_length = cluster_data.get('max_length', 0)
                        cluster.min_length = cluster_data.get('min_length', 0)
                        cluster.bond_distances = cluster_data.get('bond_distances', [])
                        orbit.clusters.append(cluster)
                    
                    obj.orbits.append(orbit)
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
                # Reconstruct basis if present, default to ChebyshevBasis if unknown
                basis_class = value.get('@class', 'ChebyshevBasis')
                if basis_class == 'ChebyshevBasis':
                    from kmcpy.structure.basis import ChebyshevBasis
                    obj.basis = ChebyshevBasis()
                elif basis_class == 'OccupationBasis':
                    from kmcpy.structure.basis import OccupationBasis
                    obj.basis = OccupationBasis()
                else:
                    # Default to ChebyshevBasis if class is not recognized
                    from kmcpy.structure.basis import ChebyshevBasis
                    obj.basis = ChebyshevBasis()
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

        # Legacy JSON fixtures may not include `name`; keep serialization robust.
        if not getattr(obj, "name", None):
            obj.name = cls.__name__
        
        return obj

    @classmethod
    def from_config(cls, config: 'Configuration'):
        """
        Create a LocalClusterExpansion from a Configuration object.
        
        Args:
            config: Configuration containing `model_file` path
            
        Returns:
            LocalClusterExpansion: Loaded LocalClusterExpansion instance
        """
        return cls.from_file(config.model_file)

    @staticmethod
    def _iter_cluster_site_indices(cluster_site_indices):
        for orbit in cluster_site_indices:
            for cluster in orbit:
                for site_idx in cluster:
                    yield int(site_idx)

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
        structure: StructureKMCpy,
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
            exclude_species: Species removed before occupation mapping. If
                omitted, use the reference local lattice's exclusion list.
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

        structure_for_occ = structure.copy()
        species_to_exclude = (
            exclude_species
            if exclude_species is not None
            else getattr(reference, "exclude_species", None)
        )
        if species_to_exclude:
            structure_for_occ.remove_species(species_to_exclude)
        structure_for_occ.remove_oxidation_states()

        occ = reference.get_occ_from_structure(
            structure_for_occ,
            tol=tol,
            angle_tol=angle_tol,
        )
        local_occ = occ[reference.site_indices]
        corr = np.empty(shape=len(self.cluster_site_indices))
        _calc_corr(corr, local_occ.array, self.cluster_site_indices)
        return occ, corr

    def get_corr_from_structure(
        self,
        structure: StructureKMCpy,
        reference_local_lattice_structure: Optional[LocalLatticeStructure] = None,
        exclude_species: Optional[Sequence[str]] = None,
        tol=1e-2,
        angle_tol=5,
    ):
        '''get_corr_from_structure() returns a correlation numpy array of correlation 0/-1 is the same as template and +1 is different
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
        from kmcpy.models.cluster import Cluster
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
            cluster = Cluster(site_indices, sites)
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
        from kmcpy.models.cluster import Orbit

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

    def compute(self, simulation_state:State, event:Event):
        """
        Compute energy value using stored parameters and correlation coefficients.
        
        This method uses the fitted parameters (keci, empty_cluster) stored in the object
        and the predefined cluster_site_indices to compute energy values.
        
        Args:
            simulation_state: State object containing occupation vector (preferred)
            event: Event object containing mobile ion indices (required for local environment)
            
        Returns:
            float: The computed energy value
        """
        # Check if parameters are stored
        if not hasattr(self, 'keci') or not hasattr(self, 'empty_cluster'):
            raise ValueError("No stored parameters found. Call set_parameters() or load_parameters_from_file() first.")
            
        # Get occupation array - prefer simulation_state over occ_global
        if simulation_state is not None:
            occ_global = simulation_state.occupations
        elif occ_global is None:
            raise ValueError("Either simulation_state or occ_global must be provided")
            
        # Ensure occ_global is a numpy array for proper indexing
        occ_global = np.array(occ_global)
            
        # Require event for local environment determination
        if event is None:
            raise ValueError("Event object is required for LocalClusterExpansion.compute() to determine local environment")
            
        # Initialize correlation array using stored cluster_site_indices
        corr = np.empty(shape=len(self.cluster_site_indices))
        
        # Extract local occupation using event's local environment indices
        occ_sublat = deepcopy(occ_global[event.local_env_indices])
            
        _calc_corr(corr, occ_sublat, self.cluster_site_indices)
        
        # Compute energy using stored parameters
        result = np.inner(corr, self.keci) + self.empty_cluster
        return result

    def compute_probability(self, *args, **kwargs):
        raise NotImplementedError("You cannot compute probability from a single LCE model, you should use CompositeLCEModel.")
    
    def set_parameters(self, parameters):
        """
        Set fitted parameters for this LocalClusterExpansion model.
        
        Args:
            parameters: LCEModelParameters object or dict containing keci and empty_cluster
        """
        from kmcpy.models.parameters import LCEModelParameters
        
        if isinstance(parameters, LCEModelParameters):
            self.keci = parameters.keci
            self.empty_cluster = parameters.empty_cluster
            self._parameters = parameters
        elif isinstance(parameters, dict):
            self.keci = parameters['keci']
            self.empty_cluster = parameters['empty_cluster']
            self._parameters = parameters
        else:
            raise TypeError("Parameters must be LCEModelParameters object or dict")
        
        logger.info(f"Parameters set for LocalClusterExpansion: keci length={len(self.keci)}, empty_cluster={self.empty_cluster}")

    def load_parameters_from_file(self, filename: str):
        """
        Load fitted parameters from a JSON file.
        
        Args:
            filename: Path to the JSON file containing LCE parameters
        """
        from kmcpy.models.parameters import LCEModelParameters
        
        parameters = LCEModelParameters.from_json(filename)
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

        payload = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "name": self.name,
            "orbits": [orbit.as_dict() for orbit in self.orbits],
            "cluster_site_indices": cluster_site_indices,
            "center_site": self.center_site.as_dict(),
            "migration_unit_structure": self.local_env_structure.as_dict()
        }
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
