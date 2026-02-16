from pymatgen.core import Structure, PeriodicSite, DummySpecies, Molecule
import numpy as np
import logging
from typing import TYPE_CHECKING, List, Dict, Any

from kmcpy.structure.lattice_structure import LatticeStructure

if TYPE_CHECKING:
    from kmcpy.structure.local_environment_comparator import LocalEnvironmentComparator

logger = logging.getLogger(__name__) 
logging.getLogger('pymatgen').setLevel(logging.WARNING)

class LocalLatticeStructure(LatticeStructure):
    """
    Class to handle local environment around a site in a structure.
    """
    def __init__(self, template_structure:Structure, 
                 center, cutoff, 
                 specie_site_mapping=None,
                 basis_type = 'chebyshev',
                 is_write_basis=False, 
                 exclude_species=None):
        # Work on a copy so local environment construction never mutates the caller's structure.
        working_structure = template_structure.copy()

        super().__init__(template_structure=working_structure, specie_site_mapping=specie_site_mapping,
                         basis_type=basis_type)
        self.cutoff = cutoff
        self.is_write_basis = is_write_basis

        if isinstance(center, int):
            self.center_site = self.template_structure[center]
        elif isinstance(center, list) or isinstance(center, tuple) or isinstance(center, np.ndarray):
            self.center_site = PeriodicSite(species=DummySpecies('X'),
                              coords=center,
                              coords_are_cartesian=False,
                              lattice = self.template_structure.lattice.copy())
            logger.debug(f"Dummy site: {self.center_site}")
        else:
            raise ValueError("Center must be an index or a list of fractional coordinates.")
        self.template_structure.remove_oxidation_states()  # Remove oxidation states if present
        if exclude_species:
            self.template_structure.remove_species(exclude_species)

        local_env_sites = self.template_structure.get_sites_in_sphere(
            self.center_site.coords, cutoff, include_index=True
        )
        
        # Sort by species name
        local_env_sites.sort(key=lambda x: x[0].species_string)

        self.site_indices = [site[2] for site in local_env_sites]
        
        local_env_structure_sites = [site[0] for site in local_env_sites]

        local_env_structure = Molecule.from_sites(local_env_structure_sites)
        local_env_structure.translate_sites(np.arange(0, len(local_env_structure), 1).tolist(), -1 * self.center_site.coords)
        if is_write_basis:
            from pymatgen.symmetry.analyzer import PointGroupAnalyzer
            logger.info("Local environment: ")
            logger.info(local_env_structure)
            local_env_structure.to(fmt="xyz", filename="local_env.xyz")
            logger.info(
            "The point group of local environment is: %s",
            PointGroupAnalyzer(local_env_structure).sch_symbol,
            )
        self.structure = local_env_structure
        
        # Initialize comparator for neighbor sequence matching
        self._comparator = None
        self._neighbor_info = None

    @staticmethod
    def sort_neighbor_info(neighbor_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deterministically sort neighbor dictionaries while preserving all metadata.

        The ordering matches the historical event-generator behavior:
        species first, then x coordinate.
        """
        return sorted(
            sorted(neighbor_info, key=lambda x: x["site"].coords[0]),
            key=lambda x: x["site"].specie,
        )

    @classmethod
    def ordered_neighbor_info_from_finder(
        cls,
        structure: Structure,
        center_index: int,
        local_env_finder,
    ) -> List[Dict[str, Any]]:
        """
        Fetch near-neighbor dictionaries from a finder and return deterministic order.

        This keeps keys such as image/local_index/label intact for downstream
        primitive-to-supercell mapping.
        """
        return cls.sort_neighbor_info(local_env_finder.get_nn_info(structure, center_index))
    
    def get_comparator(self, rtol: float = 1e-3, atol: float = 1e-3) -> 'LocalEnvironmentComparator':
        """
        Get a comparator for this local environment.
        
        Args:
            rtol: Relative tolerance for distance matrix comparison
            atol: Absolute tolerance for distance matrix comparison
            
        Returns:
            LocalEnvironmentComparator for this environment
        """
        if self._comparator is None:
            from kmcpy.structure.local_environment_comparator import LocalEnvironmentComparator
            self._comparator = LocalEnvironmentComparator.from_local_lattice_structure(
                self, rtol=rtol, atol=atol
            )
        return self._comparator
    
    def match_with_reference(
        self,
        reference_local_env: 'LocalLatticeStructure',
        rtol: float = 1e-3,
        atol: float = 1e-3,
        find_nearest_if_fail: bool = False
    ) -> 'LocalLatticeStructure':
        """
        Create a new LocalLatticeStructure with neighbors reordered to match reference.
        
        Args:
            reference_local_env: Reference local environment to match
            rtol: Relative tolerance for matching
            atol: Absolute tolerance for matching
            find_nearest_if_fail: Find approximate match if exact match fails
            
        Returns:
            New LocalLatticeStructure with reordered neighbors
        """
        reference_comparator = reference_local_env.get_comparator(rtol, atol)
        this_comparator = self.get_comparator(rtol, atol)
        
        # Match the neighbor sequences
        matched_neighbors = reference_comparator.match_neighbor_sequence(
            this_comparator.neighbor_info, find_nearest_if_fail
        )
        
        # Create a new LocalLatticeStructure with matched ordering
        # This is a simplified version - in practice you might want to 
        # reconstruct the full structure with proper ordering
        matched_local_env = LocalLatticeStructure(
            template_structure=self.template_structure,
            center=self.center_site if hasattr(self, 'center_site') else 0,
            cutoff=self.cutoff,
            specie_site_mapping=self.specie_site_mapping,
            basis_type=self.basis_type if hasattr(self, 'basis_type') else 'occupation'
        )
        
        # Update the structure with reordered sites
        matched_sites = [neighbor["site"] for neighbor in matched_neighbors]
        matched_local_env.structure = Molecule.from_sites(matched_sites)
        matched_local_env.structure.translate_sites(
            list(range(len(matched_local_env.structure))), 
            -1 * self.center_site.coords
        )
        
        return matched_local_env
    
    def get_environment_signature(self) -> tuple:
        """Get a signature that uniquely identifies the environment type."""
        comparator = self.get_comparator()
        return comparator.signature
    
    def is_equivalent_to(
        self,
        other_local_env: 'LocalLatticeStructure',
        rtol: float = 1e-3,
        atol: float = 1e-3
    ) -> bool:
        """
        Check if this local environment is equivalent to another.
        
        Args:
            other_local_env: Other local environment to compare
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
            
        Returns:
            True if environments are equivalent
        """
        try:
            this_comparator = self.get_comparator(rtol, atol)
            other_comparator = other_local_env.get_comparator(rtol, atol)
            
            # Check signatures first (quick check)
            if this_comparator.signature != other_comparator.signature:
                return False
            
            # Check distance matrices
            return np.allclose(
                this_comparator.distance_matrix,
                other_comparator.distance_matrix,
                rtol=rtol, atol=atol
            )
        except Exception:
            return False

    @classmethod
    def from_lattice_structure(cls, lattice_structure: LatticeStructure, center, cutoff,
                               specie_site_mapping=None, basis_type='chebyshev',
                               is_write_basis=False, exclude_species=None):
        """
        Create a LocalLatticeStructure from an existing LatticeStructure.
        
        Args:
            lattice_structure (LatticeStructure): The base lattice structure.
            center: Center site or coordinates for the local environment.
            cutoff (float): Cutoff distance for the local environment.
            specie_site_mapping (dict): Mapping of species to sites.
            basis_type (str): Type of basis to use.
            is_write_basis (bool): Whether to write the basis to a file.
            exclude_species (list): Species to exclude from the local environment.
        
        Returns:
            LocalLatticeStructure: The created local lattice structure.
        """
        return cls(
            template_structure=lattice_structure.template_structure,
            center=center,
            cutoff=cutoff,
            specie_site_mapping=(
                specie_site_mapping
                if specie_site_mapping is not None
                else lattice_structure.specie_site_mapping
            ),
            basis_type=basis_type,
            is_write_basis=is_write_basis,
            exclude_species=exclude_species,
        )
