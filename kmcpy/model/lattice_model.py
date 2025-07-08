from kmcpy.model.model import BaseModel
from pymatgen.core.structure import Structure
import numpy as np
from kmcpy.model.basis import ChebychevBasis, OccupationBasis
from pymatgen.analysis.structure_matcher import StructureMatcher

class LatticeModel(BaseModel):
    '''LatticeModel deal with the structure template which converts the structure to an occupation array and vice versa
    '''
    def __init__(self,template_structure:Structure,
                 species_to_site:dict,
                 basis_type:str='occupation'):
        '''Initialization of LatticeModel
            Args:
            template_structure: pymatgen Structure object, this should include all possible sites (no doping, vacancy etc.)
            species_to_site: a dictionary mapping from species to site type (possible species), including those immutable sites, 
            e.g. {"Na":["Na","X"],"X":["Na","X"],"Sb":["Sb","W"],"W":["Sb","W"]} X is the vacancy site
            basis_type: str, the type of basis function: 'occupation':[0,1] or 'chebyshev'[-1,+1]
        '''
        self.template_structure = template_structure
        self.species_to_site = species_to_site
 
        # initializing basis
        if basis_type == 'chebychev':
            self.basis = ChebychevBasis()
        elif basis_type == 'occupation':
            self.basis = OccupationBasis()
        else:
            raise NotImplementedError(f'Basis type {basis_type} not implemented!')
        
        # Initialization of species for LatticeModel
        # species is like [["Na","Va"],["Na","Va"],["Na","Va"],["Na","Va"], ... ,["Sb","W"],["Sb","W"],["Sb","W"]]
        self.species = []
        for site in self.template_structure:
            if len(site.species.elements)>1:
                from pymatgen.core.composition import CompositionError
                raise CompositionError('This code cannot deal with partial occupation!')
            species = self.get_site_species(site)
            for key in species_to_site:
                if key == species:
                    self.species.append(species_to_site[key])
        print(self.species)
        if len(self.species) != len(self.template_structure):
            raise ValueError('Length of template_structure is not the same as the species')
 
    def get_site_species(self,site): 
        try:
            species = site.species.remove_charges().elements[0].symbol
        except:
            species = site.species.elements[0].symbol
            print('oxdation state not removed!')
        return species
    
    def get_occ_from_structure(self, structure: Structure, tol=1e-2, angle_tol=5):
        '''get_occ_from_structure() returns an occupation numpy array of occupation 0/-1 is the same as template and +1 is different
        '''
        occ = []

        # Automatically determine the supercell matrix
        supercell_matrix = np.linalg.solve(self.template_structure.lattice.matrix.T, structure.lattice.matrix.T).T
        if not np.allclose(np.dot(supercell_matrix, self.template_structure.lattice.matrix), structure.lattice.matrix, atol=tol):
            raise ValueError('Lattice of given structure and template_structure (in supercell) are not the same!')

        supercell_template = self.template_structure.copy()
        supercell_template.make_supercell(supercell_matrix)

        if not np.allclose(supercell_template.lattice.matrix, structure.lattice.matrix, atol=tol):
            raise ValueError('Lattice of structure and template_structure are not the same!')

        # Initialize StructureMatcher
        matcher = StructureMatcher(ltol=tol, stol=tol, angle_tol=angle_tol)

        for each_site in supercell_template:
            # Create a temporary structure with the single site to match
            temp_structure = Structure(lattice=supercell_template.lattice, species=[each_site.species_string], coords=[each_site.frac_coords])

            if matcher.fit(temp_structure, structure):
                occ.append(self.basis.basis_function[0])  # same as the template structure
            else:
                occ.append(self.basis.basis_function[1])  # not the same as the template structure

        return np.array(occ)
        
    # def get_occ_from_structure(self, structure: Structure, tol=1e-3):
    #     '''get_occ_from_structure() returns an occupation numpy array of occupation 0/-1 is the same as template and +1 is different
    #     '''
    #     occ = []
        
    #     # Automatically determine the supercell matrix
    #     supercell_matrix = np.linalg.solve(self.template_structure.lattice.matrix.T, structure.lattice.matrix.T).T
    #     if not np.allclose(np.dot(supercell_matrix, self.template_structure.lattice.matrix), structure.lattice.matrix, atol=tol):
    #         raise ValueError('Lattice of given structure and template_structure (in supercell) are not the same!')
        
    #     supercell_template = self.template_structure.copy()
    #     supercell_template.make_supercell(supercell_matrix)
        
    #     if np.allclose(supercell_template.lattice.matrix, structure.lattice.matrix, atol=tol):
    #         for each_site in supercell_template:
    #             if self.is_in(each_site, structure, tol):
    #                 occ.append(self.basis.basis_function[0]) # same as the template structure
    #             else:
    #                 occ.append(self.basis.basis_function[1]) # not the same as the template structure
    #     else:
    #         raise ValueError('Lattice of structure and template_structure are not the same!')
    #     return np.array(occ)
    
    # def is_in(self,site,structure,tol=1e-3):# check if site is in structure, it matches the first site that encountered in the structure
    #     exist = False
    #     if site.lattice != structure.lattice:
    #         raise ValueError('Checking the existence of a site in a structure with different lattices!')
    #     for i,s in enumerate(structure.sites):
    #         if  self.get_site_species(site) == self.get_site_species(s):
    #             # the one __eq__ implemented in pymatgen uses np.allclose for coordinates, which is not working with PBC condition 
    #             if abs(site.distance(s)<tol) and exist == False:
    #                 exist = True
    #             elif abs(site.distance(s)<tol) and exist == True:
    #                 raise ValueError('There are more than two sites exist in the structure!')
    #     return exist
    
    def get_structure_from_occ(self,occ):
        '''get_structure_from_occ() takes an occupation array and returns a pymatgen Structure
        '''
        try:
            occupation = [self.basis.basis_function[o] for o in occ]
            species = [s[o] for o,s in zip(occupation,self.species)]
        except:
            raise ValueError(' get_structure_from_occ has a problem')
        lattice = self.template_structure.lattice
        frac_coords = self.template_structure.frac_coords
        non_vacancy_idx = [i for (i,s) in enumerate(species) if 'X' not in s]
        non_vacancy_species = [s for s in species if 'X' not in s]
        return Structure(species=non_vacancy_species,lattice=lattice,coords=frac_coords[non_vacancy_idx],coords_are_cartesian=False)

