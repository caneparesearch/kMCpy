from pymatgen.core.structure import Structure, Species
import numpy as np
from kmcpy.models.basis import ChebychevBasis, OccupationBasis
from pymatgen.analysis.structure_matcher import StructureMatcher
from abc import  ABC
import logging
from kmcpy.structure.vacancy import Vacancy
from kmcpy.structure.comparator import SupercellComparator

logger = logging.getLogger(__name__) 


class LatticeStructure(ABC):
    '''LatticeStructure deal with the structure template which converts the structure to an occupation array and vice versa
    '''
    def __init__(self,template_structure:Structure,
                 specie_site_mapping:dict,
                 basis_type:str='chebyshev'):
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
                
        # initializing basis
        if basis_type == 'chebyshev' or basis_type == 'trigonometric':
            self.basis = ChebychevBasis()
        elif basis_type == 'occupation':
            self.basis = OccupationBasis()
        else:
            raise NotImplementedError(f'Basis type {basis_type} not implemented!')
        
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
 
    def get_occ_from_structure(self, structure: Structure, tol=0.1, angle_tol=5, sc_matrix=None):
        """
        get_occ_from_structure() returns an occupation numpy array based on a
        comparison with the instance's template_structure.

        This implementation uses pymatgen's StructureMatcher to robustly find
        the supercell relationship and the site mapping between the input
        structure and the template.

        Args:
            structure (Structure): The input structure, which may be a supercell
                of the template and may contain vacancies.
            tol (float): Tolerance for structure matching.
            angle_tol (float): Angle tolerance for structure matching.

        Returns:
            np.array: The occupation vector for the structure.
        """
        # # 1. Find the supercell matrix
        if sc_matrix is None:
            # Try to detect supercell relationship by comparing lattice vectors
            sc_matrix = self._detect_supercell_matrix(structure, tol)
            if sc_matrix is None:
                # Fall back to StructureMatcher
                sc_matcher = StructureMatcher(ltol=tol, stol=tol, angle_tol=angle_tol, 
                                               primitive_cell=False, allow_subset=True,
                                               comparator=SupercellComparator(), 
                                               attempt_supercell=True)
                match = sc_matcher.fit(self.template_structure, structure)

                if match:
                    logger.debug("Structures match geometrically (possibly supercell).")
                    try:
                        mapping = sc_matcher.get_mapping(self.template_structure, structure)
                        sc_matrix = sc_matcher.get_supercell_matrix(self.template_structure, structure)
                        # mapping[i] = index of site in input corresponding to template i
                    except ValueError:
                        # Fallback: try the reverse mapping
                        logger.debug("Trying reverse mapping...")
                        try:
                            mapping = sc_matcher.get_mapping(structure, self.template_structure)
                            sc_matrix = sc_matcher.get_supercell_matrix(structure, self.template_structure)
                            if sc_matrix is not None:
                                sc_matrix = np.linalg.inv(sc_matrix)  # Invert for correct direction
                        except ValueError:
                            # Last fallback: use identity matrix
                            logger.warning("Could not determine supercell matrix, using identity")
                            sc_matrix = np.eye(3)
                            mapping = None
                else:
                    # Check for vacancy case: same lattice parameters but different number of sites
                    if (np.allclose(structure.lattice.matrix, self.template_structure.lattice.matrix, atol=tol) and
                        len(structure) < len(self.template_structure)):
                        logger.debug("Same lattice with fewer sites, likely vacancy case, using identity matrix")
                        sc_matrix = np.eye(3)
                    elif (np.allclose(structure.lattice.abc, self.template_structure.lattice.abc, atol=tol) and
                          np.allclose(structure.lattice.angles, self.template_structure.lattice.angles, atol=angle_tol) and
                          len(structure) < len(self.template_structure)):
                        logger.debug("Similar lattice with fewer sites, likely vacancy case, using identity matrix")
                        sc_matrix = np.eye(3)
                    else:
                        raise ValueError("Input structure does not match the template structure or its supercell.")
        logger.debug(f"Supercell matrix:\n {sc_matrix}")
        # 2. Create the ideal supercell from the template
        supercell_template = self.template_structure.copy()
        supercell_template.make_supercell(sc_matrix)
        logger.debug(f"Supercell template structure:\n {supercell_template}")

        # 3. Find the mapping from the supercell template to the input structure
        site_matcher = StructureMatcher(ltol=tol, stol=tol, angle_tol=angle_tol, 
                                        primitive_cell=False, allow_subset=True,
                                        scale=False)
        
        """
        mapping is a numpy array such that self.template_structure.sites[mapping] is within matching
        tolerance of structure.sites or None if no such mapping is possible
        """
        # Try to get mapping - input structure might be subset of supercell (has vacancies)
        # or supercell might be subset of input structure
        mapping = None
        try:
            # First try: supercell template is superset, input structure is subset (has vacancies)
            mapping = site_matcher.get_mapping(superset=supercell_template, subset=structure) 
            logger.debug(f"StructureMatcher mapping: template_structure[{mapping}] matches structure[{np.arange(len(structure))}]")
            
            # Validate the mapping - sometimes StructureMatcher gives suboptimal results
            manual_mapping = self._find_site_mapping_manual(supercell_template, structure, tol)
            logger.debug(f"Manual mapping: {manual_mapping}")
            
            # Check if manual mapping is better (has more exact matches)
            if manual_mapping is not None:
                manual_distances = []
                struct_distances = []
                
                for i, (manual_idx, struct_idx) in enumerate(zip(manual_mapping, mapping)):
                    if manual_idx is not None:
                        manual_dist = np.linalg.norm(structure[i].frac_coords - supercell_template[manual_idx].frac_coords)
                        manual_distances.append(manual_dist)
                    if struct_idx is not None:
                        struct_dist = np.linalg.norm(structure[i].frac_coords - supercell_template[struct_idx].frac_coords)
                        struct_distances.append(struct_dist)
                
                # Use manual mapping if it has better (smaller) average distance
                if manual_distances and struct_distances:
                    manual_avg = np.mean(manual_distances)
                    struct_avg = np.mean(struct_distances) 
                    logger.debug(f"Manual mapping avg distance: {manual_avg}, StructureMatcher avg distance: {struct_avg}")
                    
                    if manual_avg < struct_avg:
                        logger.debug("Using manual mapping (better distances)")
                        mapping = manual_mapping
            
            # 4. Create the occupation vector based on the mapping
            # Initialize occupation vector with vacancy (or default) values
            occ = np.array([self.basis.basis_function[1]] * len(supercell_template))
            # For each site in the input structure, set the occupation at the mapped index
            occ[mapping] = self.basis.basis_function[0] 
            
        except ValueError:
            # Second try: input structure is superset, supercell template is subset
            try:
                reverse_mapping = site_matcher.get_mapping(superset=structure, subset=supercell_template)
                logger.debug(f"structure[{reverse_mapping}] matches template_structure[{np.arange(len(supercell_template))}]")
                
                # For this case, we create occ based on supercell template size 
                # and mark occupied sites based on reverse mapping
                occ = np.array([self.basis.basis_function[0]] * len(supercell_template))
                
            except ValueError as e:
                logger.error(f"Could not establish mapping in either direction: {e}")
                raise ValueError("Could not establish site mapping between the structure and the template supercell.")

        return occ
        
    def get_structure_from_occ(self,occ, sc_matrix=np.eye(3)):
        '''get_structure_from_occ() takes an occupation array and returns a pymatgen Structure
        '''
        supercell_template = self.template_structure.copy()
        supercell_template.make_supercell(sc_matrix)
        if len(occ) != len(supercell_template):
            raise ValueError(f"Occupation array length {len(occ)} does not match template structure length {len(supercell_template)}!")
        # Create a new structure based on the template
        new_structure = supercell_template.copy()
        # Iterate through the sites and set species based on occupation
        for i, site in enumerate(new_structure):
            if occ[i] == self.basis.basis_function[0]:
                # If occupied, set to the first allowed species for this site
                site.species = self.allowed_species[i][0]
            elif occ[i] == self.basis.basis_function[1]:
                # If vacant, we should remove the species
                site.species = self.allowed_species[i][1]
                if isinstance(self.allowed_species[i][0], Vacancy()):
                    new_structure.remove_sites([i])
        return new_structure

    def _find_site_mapping_manual(self, supercell_template, structure, tol=0.1):
        """
        Manually find the mapping from supercell template sites to input structure sites.
        This is a fallback when StructureMatcher gives incorrect results.
        
        Returns:
            np.array: mapping such that supercell_template.sites[mapping[i]] matches structure.sites[i]
                     or None if no match within tolerance
        """
        mapping = []
        
        for input_site in structure:
            best_match = None
            best_distance = float('inf')
            
            for j, template_site in enumerate(supercell_template):
                # Calculate distance in fractional coordinates
                distance = np.linalg.norm(input_site.frac_coords - template_site.frac_coords)
                
                if distance < best_distance and distance <= tol:
                    best_distance = distance
                    best_match = j
                    
            mapping.append(best_match)
            
        return np.array(mapping)

    def _detect_supercell_matrix(self, structure, tol=0.1):
        """
        Detect supercell matrix by comparing lattice vectors.
        This is a simple approach that works for basic supercell relationships.
        """
        template_matrix = self.template_structure.lattice.matrix
        structure_matrix = structure.lattice.matrix
        
        # Try to find integer matrix M such that structure_matrix = M * template_matrix
        try:
            # Calculate the transformation matrix
            inv_template = np.linalg.inv(template_matrix)
            transformation = structure_matrix @ inv_template
            
            # Check if transformation is close to integer matrix
            rounded_transform = np.round(transformation)
            if np.allclose(transformation, rounded_transform, atol=tol):
                # Check if this gives a reasonable supercell
                det = np.linalg.det(rounded_transform)
                if det > 0 and det == int(det):
                    logger.debug(f"Detected supercell matrix:\n{rounded_transform}")
                    return rounded_transform
        except np.linalg.LinAlgError:
            logger.debug("Could not invert template lattice matrix")
            
        return None

    def __str__(self):
        return f"""LatticeStructure with {len(self.template_structure)} sites
        Template structure:\n {self.template_structure}
        Allowed species: {self.allowed_species}
        Species mapping: {self.specie_site_mapping}
        Basis type: {type(self.basis).__name__}"""
    
    def __repr__(self):
        return self.__str__()
    
    def as_dict(self):
        """
        Convert the model object to a dictionary representation.
        """
        return {
            "template_structure": self.template_structure.as_dict(),
            "specie_site_mapping": self.specie_site_mapping,
            "basis_type": "occupation" if isinstance(self.basis, OccupationBasis) else "chebychev"
        }

