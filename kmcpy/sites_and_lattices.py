
import numpy as np
from sympy import chebyshevu
from pymatgen.core import Structure
import json
from kmcpy.io import convert
from kmcpy.event_generator import generate_event_kernal

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

    def __str__(self):
        
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
    
    def get_copy(self):
        return Site(possible_species=self.possible_species,tag=self.tag,relative_coordinate_in_cell=self.relative_coordinate_in_cell,belongs_to_supercell=self.belongs_to_supercell,wyckoff_sequence=self.wyckoff_sequence)


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

                new_d4coordinate=self._new_d4coordinate_from_symmetry_operation(symmetry_operation=symmetry_operation,d4coordinate=initial_site.d4coordinate)
                
                if self.is_new_site(new_d4coordinate,new_d4coordinates,tolerence=0.001,axis=1):
                    temp_site=initial_site.get_copy()
                    temp_site.relative_coordinate_in_cell=self.d4coordinate_to_relative_coordinate_in_cell(new_d4coordinate)
                    temp_site.wyckoff_sequence=wyckoff_sequence
                    sites.append(temp_site)
                    
                    new_d4coordinates.append(new_d4coordinate)
                
                    wyckoff_sequence+=1
                
                
        return PrimitiveCell(containing_sites=sites)

    def what_is_it_at(self,coordinate=np.array([0.,0.,0.]),rtol=1e-05, atol=1e-08,raise_error=False):
        """Try to know what it is at the given coordinate.

        Args:
            coordinate (np.array, optional): or arrya like. Will look through all the sites and see if any site match this coordinate. Defaults to np.array([0.,0.,0.]).
            rtol (rtolerance, optional): for np. Defaults to 1e-05.
            atol (float, optional): for np. Defaults to 1e-08.
            raise_error (bool, optional): whether raise error or not, if fail to find anything at the given coordinate. Defaults to False.

        Raises:
            ValueError: if raise_error=True and there is nothing at given coordinate,raise ValueError

        Returns:
            tuple: tag and wyckoff sequence
        """
        for site in self.sites:
            if np.allclose(coordinate,np.array(site.relative_coordinate_in_cell)):
                return (site.tag,site.wyckoff_sequence)
        if raise_error:
            raise ValueError("not found at that coordinate")
        else:
            return False


    def _new_d4coordinate_from_symmetry_operation(symmetry_operation=np.array([[1,0, 0, 0],[0, 1, 0,0],[0, 0, 1,0],[0,0,0,1]]),d4coordinate=[1,0,0,1]):
        """input 4d coordinate, multiply with symmetry operation, and cut to [0,1)

        Args:
            symmetry_operation (np.array, optional): symmetry matrix. Defaults to np.array([[1,0, 0, 0],[0, 1, 0,0],[0, 0, 1,0],[0,0,0,1]]).
            d4coordinate (list, optional): len=4 list or array?. Defaults to [1,0,0,1].

        Returns:
            len=4 array: position after
        """
        
        
        # multiply the symmetry matrix
        new_d4coordinate=np.matmul(symmetry_operation,d4coordinate)
        
        # make sure in range [0,1]
        
        new_d4coordinate=new_d4coordinate-np.floor(new_d4coordinate)
        new_d4coordinate[3]=1.0
        
        return new_d4coordinate
        
    
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




class nearest_neighbor_analyzer:
    
    def __init__(self,original_structure=Structure.from_file("EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif"), local_env_cutoff_dict = {('Na+','Na+'):4,('Na+','Si4+'):4},reference_structure=PrimitiveCell.from_sites_and_symmetry_matrices(symmetry_operations=get_rotation_matrices_from_pymatgen("EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif"),initial_sites=[Site(tag="Na1",relative_coordinate_in_cell=[0.0,0.0,0.0]),Site(tag="Na2",relative_coordinate_in_cell=[0.63967, 0., 0.25]),Site(tag="Si",relative_coordinate_in_cell=[0.29544,0.,0.25])])):
        """nearest neighbor analyzer object. Use pymatgen structure as well as primitiveCell to find nearest neighbors

        Args:
            original_structure (pymatgen.core.Structure, optional): the original structure. Defaults to Structure.from_file("EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif").
            local_env_cutoff_dict (dict, optional): environment cutoff dictionary that will be passed to the finding function. Defaults to {('Na+','Na+'):4,('Na+','Si4+'):4}.
            reference_structure (kmcpy.sites_and_lattices.PrimitiveCell, optional): a primitive cell object for finding the wyckoff sequence. This shall have same structure with the original structure. Although non-interested sites (Zr and O for nasicon) can be omitted during initialization. Defaults to PrimitiveCell.from_sites_and_symmetry_matrices(symmetry_operations=get_rotation_matrices_from_pymatgen("EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif"),initial_sites=[Site(tag="Na1",relative_coordinate_in_cell=[0.0,0.0,0.0]),Site(tag="Na2",relative_coordinate_in_cell=[0.63967, 0., 0.25]),Site(tag="Si",relative_coordinate_in_cell=[0.29544,0.,0.25])]).
        """

        self.structure=original_structure
        #self.structure.make_supercell([3,3,3])
        self.local_env_cutoff_dict=local_env_cutoff_dict
        
        self.reference_structure=reference_structure
    
    
    def find(self,index_of_center_atom_of_pymatgen_structure=0,verbose=False):
        """find the nearest neighbor of site at index_of_center_atom in pymatgen structure, and get its wyckoff sequence and tag using the reference PrimitiveCell object
        
        index in pymatgen structure may be different from index in primitivecell. But, the range is the same. If there is 6 Na1 in pymatgen primitive cell that occupying index0~5, then index0~5 in primitivecell is also Na1
        

        Args:
            index_of_center_atom (int, optional): index of center atom, for NaSiCON, interested indices of center atom are 0 and 1. Defaults to 0.
            verbose (bool, optional):  whether to enable verbose output(not implemented). Defaults to False.

        Returns:
            dict: a dictionary with key=tags of nearest neighbor, and value=list of tuple, with the 1st element=wyckoff sequence and 2nd element=tuple of image, and sorted by wyckoff sequence. 
        """
        from pymatgen.analysis.local_env import CutOffDictNN
        local_env_finder = CutOffDictNN(self.local_env_cutoff_dict)
        local_env_info_list =local_env_finder.get_nn_info(self.structure,index_of_center_atom_of_pymatgen_structure)
        log=""
        neighbor_list={}
        
        index_of_center_atom_of_primitive_cell=self.reference_structure.what_is_it_at(self.structure[0].frac_coords,raise_error=True)
        
        for i in local_env_info_list:
            temp=self.reference_structure.what_is_it_at(i["site"].frac_coords-np.floor(i["site"].frac_coords),raise_error=True)
            if temp[0] not in neighbor_list:
                neighbor_list[temp[0]]=[(temp[1],i["image"])]
            else:
                neighbor_list[temp[0]].append((temp[1],i["image"]))    
            
        for tag in neighbor_list:
            neighbor_list[tag]=sorted(neighbor_list[tag],key=lambda x:x[0])
            
        return index_of_center_atom_of_primitive_cell,neighbor_list
    
    def create_supercell(self,indices_of_center_atom=[0,1,2,3,4,5],center_atom_tag="Na1",diffuse_to="Na2",environment=["Na2","Si"],supercell_shape=[5,6,7],event_fname="events.json",event_kernal_fname='event_kernal.csv'):
        
        from kmcpy.event import Event
        

        supercell_sites={}
        """
        supercell_sites={"Na1":{(1,4,3,0):123,(1,4,3,1):124}}
        
        
        {tag:{(supercell,wyckoff_sequence):global sequence}}
        
        """        
        global_sequence_dict={}
        
        """
        global_sequence_dict={123:((1,4,3,0),Na1),124:((1,4,3,1),Na1)}
        
        """

        
        global_sequence=0
        
        
        for site in self.reference_structure.sites:
            if site.tag not in supercell_sites:
                supercell_sites[site.tag]={}
        
        for i in range(1,supercell_shape[0]+1):
            for j in range(1,supercell_shape[1]+1):
                for k in range(1,supercell_shape[2]+1):
                    for site in self.reference_structure.sites:
                        # build an abstract supercell. With 
                        
                        supercell_and_wyckoff_sequence=(i,j,k,site.wyckoff_sequence)
                        supercell_sites[site.tag][supercell_and_wyckoff_sequence]=global_sequence
                        global_sequence_dict[global_sequence]=(supercell_and_wyckoff_sequence,site.tag)
                        global_sequence+=1
        
        
        events=[]
        events_dict = []        
        
        for index_of_center_atom in indices_of_center_atom:
            center_atom_information,neighbor_dict=self.find(index_of_center_atom=index_of_center_atom)
            """neighbor_dict sample {'Si': [(7, (0, 0, 0)), (9, (0, -1, 0)), (11, (-1, -1, 0)), (12, (-1, 0, 0)), (14, (-1, -1, 0)), (16, (0, 0, 0))], 'Na2': [(7, (0, 0, 0)), (9, (-1, -1, 0)), (11, (-1, 0, 0)), (12, (0, 0, 0)), (14, (-1, -1, 0)), (16, (0, -1, 0))]}
            
            center_atom_information=('Na1', 3)
            """
            
            if center_atom_information[0] != center_atom_tag:
                raise ValueError("indices_of_center_atom is not appropriate, index",index_of_center_atom," is not center atom")
            
            

        
            for i in range(1,supercell_shape[0]+1):
                for j in range(1,supercell_shape[1]+1):
                    for k in range(1,supercell_shape[2]+1):
                        center_atom_key=(i,j,k,center_atom_information[1])
                        all_neighbors=[]
                        #diffuse_to_neighbors=[]
                        for environment_atom in environment:
                            for neighbor in neighbor_dict:
                                
                                diffuse_to_atom_key=(self._equivalent_position_in_periodic_supercell(site_belongs_to_supercell=[i,j,k],image_of_site=neighbor[1],supercell_shape=supercell_shape),neighbor[0])
                                all_neighbors.append(supercell_sites[environment_atom][diffuse_to_atom_key])



                        environment_atom=diffuse_to                        
                        for neighbor in neighbor_dict:

                                
                            diffuse_to_atom_key=(self._equivalent_position_in_periodic_supercell(site_belongs_to_supercell=[i,j,k],image_of_site=neighbor[1],supercell_shape=supercell_shape),neighbor[0])
                            #diffuse_to_neighbors.append(supercell_sites[environment_atom][diffuse_to_atom_key])
                            this_event = Event()
                            this_event.initialization2(center_atom=supercell_sites[center_atom_tag][center_atom_key],diffuse_to=supercell_sites[environment_atom][diffuse_to_atom_key],sorted_sublattice_indices=all_neighbors)
                            events.append(this_event)
                            events_dict.append(this_event.as_dict())    
        print('Saving:',event_fname)
        with open(event_fname,'w') as fhandle:
            jsonStr = json.dumps(events_dict,indent=4,default=convert)
            fhandle.write(jsonStr)
        
        events_site_list = []
        for event in events:
            # sublattice indices: local site index for each site
            events_site_list.append(event.sorted_sublattice_indices)
        
        np.savetxt('./events_site_list.txt',np.array(events_site_list,dtype=int))
        generate_event_kernal(len(self.structure),np.array(events_site_list),event_kernal_fname=event_kernal_fname)                            
                                
                                
                        
                                
                      
                        
                        
        
        
        
        
        pass

    def _get_global_sequence(self,supercell_sites={"Na1":{(1,4,3,0):123,(1,4,3,1):124},"Na2":{(1,4,3,0):1234,(1,4,3,1):1244}},tag="Na1",supercell=[1,4,3],image_of_site=(-1,0,1),local_wyckoff_sequence=0):
        new_supercell_position=self._equivalent_position_in_periodic_supercell(sites_belongs_to_supercell)
        return supercell_sites[tag]


    def _equivalent_position_in_periodic_supercell(site_belongs_to_supercell=[5,1,7],image_of_site=(0,-1,1),supercell_shape=[5,6,7]):
        
        # 5 1 7 with image 0 -1 1 -> 5 0 8 -> in periodic 567 supercell should change to 561, suppose supercell start with index1
        
        temp=np.array(site_belongs_to_supercell)+np.array(image_of_site)
        # 517+(0-11)=508
        
        
        # 508-1=4-17 mod: 4 5 0 
        #+1 : 561
        temp=np.mod(temp-1,supercell_shape)+1
        
        return temp
    

