from re import match
from .model import BaseModel
from pymatgen.core.structure import Structure, DummySpecies, Species
import numpy as np
from kmcpy.models.basis import ChebychevBasis, OccupationBasis
from pymatgen.analysis.structure_matcher import StructureMatcher, AbstractComparator, OrderDisorderElementComparator
from abc import  ABC
import logging

logger = logging.getLogger(__name__) 


class LatticeModel(ABC):
    '''LatticeModel deal with the structure template which converts the structure to an occupation array and vice versa
    '''
    def __init__(self,template_structure:Structure,
                 specie_site_mapping:dict,
                 basis_type:str='chebyshev'):
        '''Initialization of LatticeModel
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
        
        # Initialization of species for LatticeModel
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
            sc_matcher = StructureMatcher(ltol=tol, stol=tol, angle_tol=angle_tol, 
                                           primitive_cell=False, allow_subset=True,
                                           comparator=SupercellComparator(), 
                                           attempt_supercell=True)
            match = sc_matcher.fit(self.template_structure, structure)

            if match:
                logger.debug("Structures match geometrically (possibly supercell).")
                mapping = sc_matcher.get_mapping(self.template_structure, structure)
                sc_matrix = sc_matcher.get_supercell_matrix(self.template_structure, structure)
                # mapping[i] = index of site in input corresponding to template i
            else:
                raise ValueError("Input structure does not match the template structure or its supercell.")

            sc_matrix = sc_matcher.get_supercell_matrix(structure, self.template_structure)

        # if supercell_matrix is None:
        #     raise ValueError("Could not establish a supercell relationship between the input structure and the template.")
            # If no supercell matrix is provided, use the identity matrix
            sc_matrix = np.eye(3)
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
        mapping = site_matcher.get_mapping(superset=supercell_template, subset=structure) 
        logger.debug(f"template_structure{mapping} matches structure{np.arange(len(structure))}")

        if mapping is None:
            raise ValueError("Could not establish site mapping between the structure and the template supercell.")

        # 4. Create the occupation vector based on the mapping
        # Initialize occupation vector with vacancy (or default) values
        occ = np.array([self.basis.basis_function[1]] * len(supercell_template))

        # For each site in the input structure, set the occupation at the mapped index
        occ[mapping] = self.basis.basis_function[0] 
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


    def __str__(self):
        return f"""LatticeModel with {len(self.template_structure)} sites
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

class Vacancy(DummySpecies):
    """
    A dummy species to represent a vacancy in the lattice model.
    """
    def __init__(self):
        super().__init__('X', 0)  # 'X' is a common symbol for vacancies
        self.is_vacancy = True

    def __repr__(self):
        return "Vacancy()"
    


class SupercellComparator(AbstractComparator):
    """
    A Comparator that matches sites, given some overlap in the element
    composition.
    """

    def are_equal(self, sp1, sp2) -> bool:
        """
        True if sp1 and sp2 are considered equivalent according to site_specie_mapping.

        Args:
            sp1: First species. A dict of {specie/element: amt} as per the
                definition in Site and PeriodicSite.
            sp2: Second species. A dict of {specie/element: amt} as per the
                definition in Site and PeriodicSite.

        Returns:
            True if sp1 and sp2 are allowed to be on the same site according to mapping.
        """
        return True

    def get_hash(self, composition):
        """Get the fractional composition."""
        return composition.fractional_composition