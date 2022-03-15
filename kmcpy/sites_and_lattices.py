
import numpy as np
from sympy import chebyshevu

# use spglib!
# 看bond length 一不一样长
# 测试一下生成的structure跟原来一不一样


def dot_product(array1,array2):
    result_product=[]
    if len(array1)!=len(array2):
        raise ValueError("sites_and_lattices.dot_product: array length unequal")
    for i in range(0,len(array1)):
        result_product.append(array1[i]*array2[i])
    return result_product

class Site(object):

    def __init__(self,possible_species=["Na","Vac"],tag="Na1",relative_coordinate_in_cell=[0.0,0.0,0.0],belongs_to_supercell=[0,0,0],wyckoff_sequence=0):
        """this define a general Site object. I think this is for analyzing neighbors and event_generators, etc.

        Args:
            possible_species (list, optional): list, with len=1 or 2. Define possible species that could be on this site. Defaults to ["Na","Vac"].
            tag (str, optional): a unique identifier for the site. Na1, Na2, S+P, Zr, O, etc. This is for identifying different site when analyzing the neighbors. Defaults to "Na1".
            relative_coordinate_in_cell (list, optional): This is the fractional coordinate inside the supercell. with len=3. and each element value 0<=x<=1. Defaults to [0.0,0.0,0.0].
            belongs_to_supercell (list, optional): belongs to which supercell in the whole lattice. sequence of interger. Defaults to [0,0,0].
            wyckoff_sequence (int, optional): I don't know the correct terminology for this. This identify the sequence of the symmetry operation. For example, R-3c has 12 equivalent position, which means that at most the wyckoff_sequence=11 if considering special position. Defaults to 0.
        """

        self.possible_species=possible_species
        self.relative_coordinate_in_cell=relative_coordinate_in_cell
        if type(self.relative_coordinate_in_cell) is not list:
            raise TypeError("relative_coordinate_in_cell must be list, not np.array!")

        self.belongs_to_supercell=belongs_to_supercell
        if type(self.belongs_to_supercell) is not list:
            raise TypeError("supercell must be list, not np.array!")
        self.wyckoff_sequence=wyckoff_sequence
        self.tag=tag
        
        # d4coordinate is 4*1 array for symmetry operations.
        self.d4coordinate=np.array(self.relative_coordinate_in_cell+[1.0]).reshape((4,1))
        
        # this shows how d4coordinate returns to relatice_coordinate_in_cell:
        # self.relative_coordinate_in_cell=self.d4coordinate.reshape(1,4).tolist()[0][0:3]
        
        
        self.default_specie=self.possible_species[0]# choose the 1st one as default_specie if not defined.
        pass

    def as_dict(self):
        d = {"@module":self.__class__.__module__,
        "@class": self.__class__.__name__,
        "possible_species":self.possible_species,
        "relative_coordinate_in_cell":self.relative_coordinate_in_cell,
        "belongs_to_supercell":self.belongs_to_supercell,
        "wyckoff_sequence":self.wyckoff_sequence,
        "tag":self.tag}
        return d

    def __hash__(self):
        """hash function. Take the: [tag(na1), supercell, wyckoff sequence] as identifier

        Returns:
            int: hash
        """
        return (self.tag,tuple(self.belongs_to_supercell),self.wyckoff_sequence).__hash__()

    

class kmcSite(Site):
    def __init__(self, possible_species=["Na", "Vac"], tag="Na1", relative_coordinate_in_cell=[0, 0, 0], absolute_coordinate_in_space=[0, 0, 0], belongs_to_supercell=[0, 0, 0], wyckoff_sequence=0,chebyshev_state=1):
        super().__init__(possible_species, tag, relative_coordinate_in_cell, absolute_coordinate_in_space, belongs_to_supercell, wyckoff_sequence)
        self.chebyshev_state=chebyshev_state
        if len(self.possible_species)==1:
            self.current_specie=self.possible_species[0]
        elif len(self.possible_species)==2:
            if self.chebyshev_state==1:
                self.current_specie=self.possible_species[0]
            elif self.chebyshev_state==-1:
                self.current_specie=self.possible_species[1]
            else:
                raise ValueError("got wrong chebyshev state:",self.chebyshev_state,type(self.chebyshev_state))
        else:
            raise ValueError("something wrong with possible species",self.possible_species,type(self.possible_species))
    
class PrimitiveCell(object):
    def __init__(self,containing_sites=[Site(),]):
        """PrimitiveCell object, that define one Primitive cell.

        Args:
            containing_sites (list, optional): a list that containing all Site() object in this cell. Generally you don't want to initialize the primitive cell by __init__. Defaults to [Site(),].
        """
        self.sites=containing_sites
        pass

    @classmethod
    def from_sites_and_symmetry_matrices(self,symmetry_operations=[np.array([[1,0, 0, 0],[0, 1, 0,0],[0, 0, 1,0],[0,0,0,1]])],initial_sites=[Site(),Site(relative_coordinate_in_cell=[0.1, 0.2, 0.3],tag="dummy")]):
        """generate a primitive cell object from the symmetry operations of the primitive cell and initial sites
        
        for every initial site, do the symmetry operations and add the new site to the collection as long as no degeneracy

        Args:
            symmetry_operations (list, optional): list of 4*4 np array, the np.array is the symmetry operations. Defaults to [np.array([[1,0, 0, 0],[0, 1, 0,0],[0, 0, 1,0],[0,0,0,1]])].
            initial_sites (list, optional): list of Site object that in the primitive cell, for example, for LZSP should contain sites corresponding to  Zr, O, Na1, Na2, S/P. Defaults to [Site(),Site()].

        Returns:
            PrimitiveCell: PrimitiveCell object with its sites attribute being a list, that containing all Sites() in the primitive cell
        """
        sites=[]
        new_d4coordinates=[]
        for initial_site in initial_sites:
            wyckoff_sequence=0
            for symmetry_operation in symmetry_operations:

                new_d4coordinate=np.matmul(symmetry_operation,initial_site.d4coordinate)
                
                if is_new_site(new_d4coordinate,new_d4coordinates,tolerence=0.001,axis=1):
                    sites.append(Site(possible_species=initial_site.possible_species,tag=initial_site.tag,relative_coordinate_in_cell=d4coordinate_to_relative_coordinate_in_cell(new_d4coordinate),belongs_to_supercell=[0,0,0],wyckoff_sequence=wyckoff_sequence))
                    
                    new_d4coordinates.append(new_d4coordinate)
                
                wyckoff_sequence+=1
                
                
        return PrimitiveCell(containing_sites=sites)


    
    
def is_new_site(new_coordinate=np.array([[ 1.66666667],
    [-0.66666667],
    [ 3.83333333],
    [ 1.        ]]),coordinates=[np.array([[ 1.66666667],
    [-0.66666667],
    [ 3.83333333],
    [ 1.        ]]),np.array([[ 2.66666667],
    [-0.66666667],
    [ 3.83333333],
    [ 1.        ]])],tolerence=0.001,axis=1):
    """
    determine if this is new site
    
    Mainly use in the sites_and_lattices.PrimitiveCell class
    
    This is trying to see if the "new coordinat"e is same as any coordinate (element of the list "coordinates") in the "coordinates"
    
    
    example:
    
    coordinates=array_list=[np.array([[ 1.66666667],
    [-0.66666667],
    [ 3.83333333],
    [ 1.        ]]),np.array([[ 2.66666667],
    [-0.66666697],
    [ 3.83333323],
    [ 1.        ]]),np.array([[ 1.66666667],
    [-0.67666667],
    [ 3.83333333],
    [ 1.        ]])]

    new_coordinate=to_be_compared=np.array([[ 1.66666667],
        [-0.66666667],
        [ 3.83333333],
        [ 1.        ]])

    print(array_list-to_be_compared)
    print("sum axis0:",np.sum(array_list-to_be_compared,axis=0))
    print("sum axis1:",np.sum(array_list-to_be_compared,axis=1))
    print("sum axis2:",np.sum(array_list-to_be_compared,axis=2))
    print(" sum of axis 1 is correct.")
    print("sum of abs of axis1:",np.sum(np.abs(array_list-to_be_compared),axis=1))
    tolerence=0.01
    print("determine abs:",np.sum(np.abs(array_list-to_be_compared),axis=1)<tolerence)
    print("test any",np.any(np.sum(np.abs(array_list-to_be_compared),axis=1)<tolerence))

    Args:
        new_coordinate (1*4 np.array, optional): the 4 dimention coordinate that want to be compared. Defaults to np.array([[ 1.66666667], [-0.66666667], [ 3.83333333], [ 1.        ]]).
        
        coordinates (list, optional): list containing 1*4 np.array. To see if new_coordinate is in the coordinates or not. Defaults to [np.array([[ 1.66666667], [-0.66666667], [ 3.83333333], [ 1.        ]]),np.array([[ 2.66666667], [-0.66666667], [ 3.83333333], [ 1.        ]])].
        
        tolerence (float, optional): tolerance of the sum of absolute difference between new_coordinate and elements in coordinates. Defaults to 0.001.
        
        axis (int,tuple,..?) axis parameter for np.sum. axis=1 is working

    Returns:
        bool: whether the new_coordinate is the new coordinate or it is already in the coordinates
    """
    
    if len(coordinates)==0:
        return True
    return (not np.any(np.sum(np.abs(coordinates-new_coordinate),axis=axis)<tolerence))
        
    
def d4coordinate_to_relative_coordinate_in_cell(d4coordinate=np.array([[ 1.66666667],
    [-0.66666667],
    [ 3.83333333],
    [ 1.        ]])):
    return d4coordinate.reshape(1,4).tolist()[0][0:3]


def get_rotation_matrices_from_pymatgen(filename="EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif"):
    """
    use pymatgen to get all the rotation matrices

    Args:
        filename (str, optional): the crystal file name that pymatgen can read. Defaults to "EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif".
    
    Return:
        a list, with the element of the list being 4*4 symmetry operation matrix, and the length of list equal to the number of possible symmetry operations.
    """
    from pymatgen.symmetry.analyzer  import SpacegroupAnalyzer
    from pymatgen.core.structure import Structure
    
    a=SpacegroupAnalyzer(Structure.from_file(filename))
    #print(a.get_conventional_to_primitive_transformation_matrix())
    #print(a.get_space_group_operations())
    #print(a.get_symmetry_dataset())

    
    rotation_matrices=[]

    for i in a.get_symmetry_operations():
        rotation_matrices.append(i.affine_matrix)
    return rotation_matrices

class Supercell():
    def __init__(self,supercell=[2,1,1],primitive_cell=PrimitiveCell()) -> None:
        pass