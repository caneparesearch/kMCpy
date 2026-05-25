from pymatgen.core import Structure, PeriodicSite, DummySpecies, Molecule, Species
import numpy as np
import logging
from typing import List, Dict, Any

from kmcpy.structure.lattice_structure import LatticeStructure
from kmcpy.structure.active_site_index_map import ActiveSiteIndexMap
from kmcpy.structure.cluster import Cluster, ClusterMatcher
from kmcpy.structure.local_site_ordering import (
    LocalSiteOrderingConvention,
    ordered_site_hash,
    ordered_site_signature,
)

logger = logging.getLogger(__name__) 
logging.getLogger('pymatgen').setLevel(logging.WARNING)

class LocalLatticeStructure(LatticeStructure):
    """
    Class to handle local environment around a site in a structure.
    """
    def __init__(self, template_structure:Structure, 
                 center, cutoff, 
                 site_mapping=None,
                 basis_type = 'chebyshev',
                 is_write_basis=False, 
                 exclude_species=None,
                 ordering_convention=None,
                 exclude_center_site=None):
        if exclude_species:
            raise ValueError(
                "exclude_species is no longer supported; encode fixed sites in "
                "site_mapping with a single allowed species."
            )

        # Work on a copy so local environment construction never mutates the caller's structure.
        working_structure = template_structure.copy()
        active_site_index_map = ActiveSiteIndexMap.from_structure_and_mapping(
            working_structure, site_mapping
        )
        if isinstance(center, int):
            primitive_to_active = active_site_index_map.primitive_to_active
            if center not in primitive_to_active:
                raise ValueError(
                    f"center site {center} is fixed by site_mapping and is "
                    "not part of the active-site index space"
                )
            center = primitive_to_active[center]
        working_structure = active_site_index_map.active_structure()
        working_structure.remove_oxidation_states()
        ordering = LocalSiteOrderingConvention.resolve(ordering_convention)
        if exclude_center_site is not None:
            ordering = ordering.with_exclude_center_site(exclude_center_site)

        super().__init__(template_structure=working_structure, site_mapping=site_mapping,
                         basis_type=basis_type)
        self.active_site_index_map = active_site_index_map
        self.cutoff = cutoff
        self.is_write_basis = is_write_basis
        self.ordering_convention = ordering

        if isinstance(center, int):
            self.center_site = self.template_structure[center]
            self.center_index = center
        elif isinstance(center, list) or isinstance(center, tuple) or isinstance(center, np.ndarray):
            self.center_site = PeriodicSite(species=DummySpecies('X'),
                              coords=center,
                              coords_are_cartesian=False,
                              lattice = self.template_structure.lattice.copy())
            self.center_index = None
            logger.debug(f"Dummy site: {self.center_site}")
        else:
            raise ValueError("Center must be an index or a list of fractional coordinates.")
        self.exclude_species = []

        local_env_sites = self.template_structure.get_sites_in_sphere(
            self.center_site.coords, cutoff, include_index=True
        )
        if self.ordering_convention.exclude_center_site:
            local_env_sites = [
                site_info
                for site_info in local_env_sites
                if not self._is_center_site(site_info)
            ]
        
        local_env_sites = self.ordering_convention.sort_local_env_sites(local_env_sites)

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
        self.local_environment_signature = ordered_site_signature(self.structure)
        self.local_environment_hash = ordered_site_hash(self.local_environment_signature)


    @staticmethod
    def _normalize_exclude_species(exclude_species) -> list[str]:
        """Return exclude tokens that match oxidized and neutral structures."""
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

    def _is_center_site(self, site_info) -> bool:
        """Return whether a sphere result corresponds to the center site."""
        site = site_info[0]
        site_index = site_info[2]
        if self.center_index is not None and int(site_index) == int(self.center_index):
            return True
        return (
            np.linalg.norm(site.coords - self.center_site.coords)
            <= self.ordering_convention.center_match_tolerance
        )

    @staticmethod
    def sort_neighbor_info(neighbor_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deterministically sort neighbor dictionaries while preserving all metadata.

        The ordering matches the historical event-generator behavior:
        species first, then x coordinate.
        """
        convention = LocalSiteOrderingConvention.from_name("nasicon_nat_commun_2022")
        return sorted(neighbor_info, key=lambda x: convention._sort_key(x["site"]))

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
    
    def to_cluster(self):
        """Return this local environment as a finite structural cluster."""
        return Cluster.from_sites(
            self.structure,
            site_indices=self.site_indices,
        )
    
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
        match = ClusterMatcher(
            reference_local_env.to_cluster(),
            rtol=rtol,
            atol=atol,
        ).match(
            self.to_cluster(),
            find_nearest_if_fail=find_nearest_if_fail,
        )

        matched_local_env = self.__class__.__new__(self.__class__)
        matched_local_env.__dict__.update(self.__dict__.copy())
        matched_local_env.structure = Molecule.from_sites(
            [self.structure[index] for index in match.reference_to_candidate]
        )
        matched_local_env.site_indices = [
            self.site_indices[index] for index in match.reference_to_candidate
        ]
        matched_local_env.local_environment_signature = ordered_site_signature(
            matched_local_env.structure
        )
        matched_local_env.local_environment_hash = ordered_site_hash(
            matched_local_env.local_environment_signature
        )
        return matched_local_env
    
    def get_environment_signature(self) -> tuple:
        """Get a signature that uniquely identifies the environment type."""
        return self.to_cluster().signature

    def get_ordered_site_signature(self) -> list[dict[str, Any]]:
        """Get an order-sensitive signature for this local environment."""
        return ordered_site_signature(self.structure)

    def get_ordered_site_hash(self) -> str:
        """Get an order-sensitive hash for this local environment."""
        return ordered_site_hash(self.get_ordered_site_signature())
    
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
            ClusterMatcher(
                self.to_cluster(),
                rtol=rtol,
                atol=atol,
            ).match(other_local_env.to_cluster())
        except ValueError:
            return False
        return True

    @classmethod
    def from_lattice_structure(cls, lattice_structure: LatticeStructure, center, cutoff,
                               site_mapping=None, basis_type='chebyshev',
                               is_write_basis=False, exclude_species=None,
                               ordering_convention=None, exclude_center_site=None):
        """
        Create a LocalLatticeStructure from an existing LatticeStructure.
        
        Args:
            lattice_structure (LatticeStructure): The base lattice structure.
            center: Center site or coordinates for the local environment.
            cutoff (float): Cutoff distance for the local environment.
            site_mapping (dict): Mapping of species to sites.
            basis_type (str): Type of basis to use.
            is_write_basis (bool): Whether to write the basis to a file.
            exclude_species: Removed legacy argument; use site_mapping fixed sites.
        
        Returns:
            LocalLatticeStructure: The created local lattice structure.
        """
        return cls(
            template_structure=lattice_structure.template_structure,
            center=center,
            cutoff=cutoff,
            site_mapping=(
                site_mapping
                if site_mapping is not None
                else lattice_structure.site_mapping
            ),
            basis_type=basis_type,
            is_write_basis=is_write_basis,
            exclude_species=exclude_species,
            ordering_convention=ordering_convention,
            exclude_center_site=exclude_center_site,
        )
