from pymatgen.core.structure import Structure, Species
import numpy as np
from kmcpy.structure.basis import Occupation, get_basis, BasisFunction
from pymatgen.analysis.structure_matcher import StructureMatcher
from abc import ABC
import logging
from kmcpy.structure.vacancy import Vacancy
from kmcpy.structure.comparator import SupercellComparator
from typing import List, Union

logger = logging.getLogger(__name__) 


class LatticeStructure(ABC):
    '''LatticeStructure deal with the structure template which converts the structure to an occupation array and vice versa
    '''
    def __init__(self, template_structure: Structure,
                 specie_site_mapping: dict,
                 basis_type: str = 'chebyshev'):
        '''Initialization of LatticeStructure
            Args:
            template_structure: pymatgen Structure object, this should include all possible sites (no doping, vacancy etc.)
            specie_site_mapping: a dictionary mapping from species to site type (possible species), including those immutable sites, 
            e.g. {"Na":["Na","X"],"X":["Na","X"],"Sb":["Sb","W"],"W":["Sb","W"]} X is the vacancy site
            basis_type: str, the type of basis function: 'occupation':[0,1] or 'chebyshev'[-1,+1]
        '''
        self.template_structure = template_structure

        # Avoid modifying the dictionary while iterating
        items = list(specie_site_mapping.items())
        for key, value in items:
            if isinstance(key, str):
                # If the key is a string, convert it to Pymatgen Species
                specie_site_mapping[Species(key)] = specie_site_mapping.pop(key)
        items = list(specie_site_mapping.items())
        for key, value in items:
            if isinstance(value, str):
                if value == 'X':
                    # If the value is 'X', treat it as a vacancy
                    specie_site_mapping[key] = [Vacancy()]
                else:
                    # Otherwise, treat it as a regular species
                    specie_site_mapping[key] = [Species(value)]
            elif isinstance(value, list):
                # If the value is a list, convert each string to Pymatgen Species
                specie_site_mapping[key] = [
                    Vacancy() if (isinstance(v, str) and v == 'X') else (Species(v) if isinstance(v, str) else v)
                    for v in value
                ]
        self.specie_site_mapping = specie_site_mapping
                
        # Initialize basis using new registry system
        try:
            self.basis = get_basis(basis_type)
            self.basis_type = basis_type
        except ValueError as e:
            raise ValueError(f'Basis type {basis_type} not supported. {e}')
        
        # Initialization of species for LatticeStructure
        # allowed_species is like [["Na","Va"],["Na","Va"],["Na","Va"],["Na","Va"], ... ,["Sb","W"],["Sb","W"],["Sb","W"]]
        self.allowed_species = []
        for site in self.template_structure:
            species = Species(site.species_string)
            # Create a list of allowed species for the site
            allowed_specie = self.specie_site_mapping.get(species)
            self.allowed_species.append(allowed_specie)

        if len(self.allowed_species) != len(self.template_structure):
            raise ValueError(f"Species length {len(self.allowed_species)} does not match template structure length {len(self.template_structure)}!")
 
    def get_occ_from_structure(self, structure: Structure, tol=0.1, angle_tol=5, sc_matrix=None) -> Occupation:
        """
        get_occ_from_structure() returns an Occupation object based on a
        comparison with the instance's template_structure.

        This implementation uses pymatgen's StructureMatcher to robustly find
        the supercell relationship and the site mapping between the input
        structure and the template.

        Args:
            structure (Structure): The input structure, which may be a supercell
                of the template and may contain vacancies.
            tol (float): Tolerance for structure matching.
            angle_tol (float): Angle tolerance for structure matching.
            sc_matrix (np.ndarray, optional): Supercell matrix if known.

        Returns:
            Occupation: The occupation object for the structure with proper basis.
        """
        # 1. Determine supercell matrix if not provided
        if sc_matrix is None:
            # Attempt to automatically detect supercell matrix
            # Compare lattice vectors to determine the supercell transformation
            template_lattice = self.template_structure.lattice.matrix
            structure_lattice = structure.lattice.matrix

            # Try to solve: structure_lattice = sc_matrix @ template_lattice
            try:
                sc_matrix_candidate = structure_lattice @ np.linalg.inv(template_lattice)

                # Round to nearest integers (supercell matrix should be integer)
                sc_matrix = np.round(sc_matrix_candidate).astype(int)

                # Validate that this is a good supercell matrix
                # Use stricter tolerance for automatic detection
                reconstructed = sc_matrix @ template_lattice
                strict_tol = min(tol, 0.01)  # Use at most 1% tolerance for supercell detection
                if not np.allclose(reconstructed, structure_lattice, rtol=strict_tol, atol=strict_tol):
                    logger.debug("Could not find integer supercell matrix, structures may be incompatible")
                    raise ValueError("No mapping found: cannot find valid supercell transformation")
                else:
                    logger.debug(f"Detected supercell matrix:\n{sc_matrix}")
            except np.linalg.LinAlgError:
                logger.debug("Singular matrix encountered")
                raise ValueError("No mapping found: lattice matrices are incompatible")

        logger.debug(f"Using supercell matrix:\n {sc_matrix}")
        
        # 2. Create the supercell template
        supercell_template = self.template_structure.copy()
        supercell_template.make_supercell(sc_matrix)
        logger.debug(f"Supercell template has {len(supercell_template)} sites")
        logger.debug(f"Input structure has {len(structure)} sites")
        
        # Initialize all sites as mismatch (vacant - no atom present)
        occ_data = np.full(len(supercell_template), self.basis.mismatch_value,
                          dtype=type(self.basis.mismatch_value))

        # Handle empty structure case (all sites are vacant)
        if len(structure) == 0:
            logger.debug("Empty structure - all sites are vacant")
            return Occupation(occ_data, basis=self.basis, validate=False)
        
        # 3. Set up StructureMatcher with OrderDisorderElementComparator for vacancy handling
        from pymatgen.analysis.structure_matcher import OrderDisorderElementComparator
        matcher = StructureMatcher(ltol=tol, stol=tol, angle_tol=angle_tol,
                                 primitive_cell=False, allow_subset=True,
                                 scale=True,
                                 comparator=OrderDisorderElementComparator())
        
        # 4. Validate structure compatibility
        # Check lattice compatibility
        if not np.allclose(supercell_template.lattice.matrix, structure.lattice.matrix,
                          rtol=tol, atol=tol):
            logger.debug("Lattice mismatch detected")
            raise ValueError("No mapping found: lattice parameters don't match within tolerance")

        # 4.5. Use distance-based mapping to find which template sites correspond to structure sites
        # For each structure site, find the nearest template site
        from scipy.spatial.distance import cdist
        template_coords = supercell_template.frac_coords
        structure_coords = structure.frac_coords

        # distances[i, j] = distance from template site i to structure site j
        distances = cdist(template_coords, structure_coords, metric='euclidean')

        # For each structure site j, find the closest template site
        # This gives us which template sites are occupied
        template_site_indices = np.argmin(distances, axis=0)

        # Validate that all mappings are within tolerance
        min_distances = np.min(distances, axis=0)
        if np.any(min_distances > tol):
            logger.debug(f"Some sites exceed tolerance: max distance = {np.max(min_distances)}")
            raise ValueError(f"No mapping found: some atoms are too far from template sites (max distance: {np.max(min_distances):.4f}, tolerance: {tol})")

        logger.debug(f"Structure sites map to template sites: {template_site_indices}")

        # 5. Create occupation vector
        # Sites that have a structure atom (are in template_site_indices) should be match (occupied)
        occ_data[template_site_indices] = self.basis.match_value
        
        logger.debug(f"Occupation vector: {occ_data}")
        
        # Return Occupation object
        return Occupation(occ_data, basis=self.basis, validate=False)
        
    def get_structure_from_occ(self, occ: Occupation, sc_matrix=np.eye(3)) -> Structure:
        '''get_structure_from_occ() takes an Occupation object and returns a pymatgen Structure
        
        Args:
            occ: Occupation object containing site occupation data
            sc_matrix: Supercell matrix for creating the supercell
            
        Returns:
            Structure: pymatgen Structure with species assigned based on match/mismatch
        '''
        
        supercell_lattice_structure = self.copy()
        supercell_lattice_structure.make_supercell(sc_matrix)
        
        if len(occ) != len(supercell_lattice_structure.template_structure):
            raise ValueError(f"Occupation array length {len(occ)} does not match template structure length {len(supercell_lattice_structure.template_structure)}!")
        
        # Create a new structure based on the template
        new_lattice_structure = supercell_lattice_structure.copy()
        
        # Iterate through the sites and set species based on occupation
        for i, site in enumerate(new_lattice_structure.template_structure):
            occ_value = occ[i]  # Get occupation value at site i
            if occ_value == self.basis.mismatch_value:
                # If mismatch, set to the second allowed species for this site
                site.species = self.allowed_species[i][1]
            elif occ_value == self.basis.match_value:
                # If match, set to the first allowed species (template)
                site.species = self.allowed_species[i][0]
                # if isinstance(self.allowed_species[i][0], Vacancy):
                #     # If it's a vacancy, we should remove the site
                #     # Note: This needs to be done carefully to avoid index issues
                #     pass  # Handle vacancy removal in a separate step
        
        # Remove vacancy sites in reverse order to avoid index shifting issues
        vacancy_indices = []
        for i, site in enumerate(new_lattice_structure.template_structure):
            if isinstance(site.species, Vacancy):
                vacancy_indices.append(i)
        
        # Remove vacancy sites from the end to avoid index issues
        for i in reversed(vacancy_indices):
            new_lattice_structure.template_structure.remove_sites([i])

        return new_lattice_structure.template_structure

    def copy(self):
        '''Create a copy of the LatticeStructure'''
        return LatticeStructure(self.template_structure.copy(),
                                self.specie_site_mapping.copy(),
                                self.basis_type)
    
    def make_supercell(self, sc_matrix: np.ndarray):
        '''Create a supercell of the template structure'''
        self.template_structure.make_supercell(sc_matrix)
        # Update allowed_species accordingly
        original_allowed_species = self.allowed_species.copy()
        self.allowed_species = []
        for i in range(sc_matrix[0,0]):
            for j in range(sc_matrix[1,1]):
                for k in range(sc_matrix[2,2]):
                    self.allowed_species.extend(original_allowed_species)
        if len(self.allowed_species) != len(self.template_structure):
            raise ValueError(f"After supercell, species length {len(self.allowed_species)} does not match template structure length {len(self.template_structure)}!")
        
    def __str__(self):
        return f"""LatticeStructure with {len(self.template_structure)} sites
        Template structure:\n {self.template_structure}
        Allowed species: {self.allowed_species}
        Species mapping: {self.specie_site_mapping}
        Basis type: {self.basis_type}"""
    
    def __repr__(self):
        return self.__str__()
    
    def as_dict(self):
        """
        Convert the model object to a dictionary representation.
        """
        return {
            "template_structure": self.template_structure.as_dict(),
            "specie_site_mapping": self.specie_site_mapping,
            "basis_type": self.basis_type
        }

