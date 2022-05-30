import numpy as np
from kmcpy.io import convert
import itertools
import logging

neighbor_info_logger=logging.getLogger("neighbor information:")
neighbor_info_logger.setLevel("INFO")
neighbor_info_logger.addHandler(logging.StreamHandler())
neighbor_info_logger.addHandler(logging.FileHandler("debug.log"))

class neighbor_info_matcher():
    
    def __init__(self,neighbor_species=(('Cl-', 4),('Li+', 8)),distance_matrix=np.array([[0,1],[1,0]]),neighbor_sequence=[{}],neighbor_species_respective_distance_matrix_dict={"Cl-":np.array([[0,1],[1,0]]),"Li+":np.array([[0,1],[1,0]])},neighbor_species_respective_neighbor_sequence_dict={"Cl-":[{}],"Li+":[{}]}):
        """neighbor_info matcher, the __init__ method shouln't be used. Use the from_neighbor_info() instead. This is neighbor_info matcher to match the nearest neighbor info output from local_env.cutoffdictNN.get_nn_info. This neighbor_info_matcher Class is initialized by a reference neighbor_info, a distance matrix is built as reference. Then user can call the neighbor_info_matcher.brutal_match function to sort another nn_info so that the sequence of neighbor of "another nn_info" is arranged so that the distance matrix are the same 

        Args:
            neighbor_species (tuple, optional): tuple ( tuple ( str(species), int(number_of_this_specie in neighbors)  )  ). Defaults to (('Cl-', 4),('Li+', 8)).
            distance_matrix (np.array, optional): np.2d array as distance matrix. Defaults to np.array([[0,1],[1,0]]).
            neighbor_sequence (list, optional): list of dictionary in the format of nn_info returning value. Defaults to [{}].
            neighbor_species_respective_distance_matrix_dict (dict, optional): this is a dictionary with key=species and value=distance_matrix(2D numpy array) which record the distance matrix of respective element. . Defaults to {"Cl-":np.array([[0,1],[1,0]]),"Li+":np.array([[0,1],[1,0]])}.
            neighbor_species_respective_neighbor_sequence_dict (dict, optional): dictionary with key=species and value=list of dictionary which is just group the reference neighbor sequence by different elements. Defaults to {"Cl-":[{}],"Li+":[{}]}.
        """
        
        self.neighbor_species=neighbor_species
        self.distance_matrix=distance_matrix
        self.neighbor_species_respective_distance_matrix_dict=neighbor_species_respective_distance_matrix_dict
        self.neighbor_species_respective_neighbor_sequence_dict=neighbor_species_respective_neighbor_sequence_dict
        self.neighbor_sequence=neighbor_sequence
        
    
        
        pass
    
    @classmethod
    def from_neighbor_sequences(self,neighbor_sequences=[{}]):
        cn_dict={}
        neighbor_species_respective_neighbor_sequence_dict={}
        
        for neighbor in neighbor_sequences:
            
            site_element = neighbor["site"].species_string
            
            if site_element not in cn_dict:
                cn_dict[site_element] = 1
            else:
                cn_dict[site_element] += 1
            if site_element not in neighbor_species_respective_neighbor_sequence_dict:
                neighbor_species_respective_neighbor_sequence_dict[site_element]=[neighbor]
            else:
                neighbor_species_respective_neighbor_sequence_dict[site_element].append(neighbor)
        
            
        neighbor_species_respective_distance_matrix_dict={}
        
        for species in neighbor_species_respective_neighbor_sequence_dict:
            neighbor_species_respective_distance_matrix_dict[species]=self.build_distance_matrix_from_getnninfo_output(neighbor_species_respective_neighbor_sequence_dict[species])
        
        neighbor_species=tuple(sorted(cn_dict.items(),key=lambda x:x[0]))
        
        distance_matrix=self.build_distance_matrix_from_getnninfo_output(neighbor_sequences)
        
                
        return neighbor_info_matcher(neighbor_species=neighbor_species,distance_matrix=distance_matrix,neighbor_sequence=neighbor_sequences,neighbor_species_respective_distance_matrix_dict=neighbor_species_respective_distance_matrix_dict,neighbor_species_respective_neighbor_sequence_dict=neighbor_species_respective_neighbor_sequence_dict)
        
    
    @classmethod
    def build_distance_matrix_from_getnninfo_output(self,cutoffdnn_output=[{}]):
        """build a distance matrix from the output of CutOffDictNN.get_nn_info

        nn_info looks like: 
        [{'site': PeriodicSite: Si4+ (-3.2361, -0.3015, 9.2421) [-0.3712, -0.0379, 0.4167], 'image': (-1, -1, 0), 'weight': 3.7390091507903174, 'site_index': 39, 'wyckoff_sequence': 15, 'local_index': 123, 'label': 'Si1'}, {'site': PeriodicSite: Na+ (-1.2831, -2.6519, 9.2421) [-0.3063, -0.3333, 0.4167], 'image': (-1, -1, 0), 'weight': 3.4778161424304046, 'site_index': 23, 'wyckoff_sequence': 17, 'local_index': 35, 'label': 'Na2'}, {'site': ...]
        
        or say:
        
        nn_info is a list, the elements of list is dictionary, the keys of dictionary are: "site":pymatgen.site, "wyckoff_sequence": ....
        
        Use the site.distance function to build matrix
        

        Args:
            cutoffdnn_output (nn_info, optional): nninfo. Defaults to neighbor_sequences.

        Returns:
            np.2darray: 2d distance matrix, in format of numpy.array. The Column and the Rows are following the input sequence.
        """
    
        distance_matrix=np.zeros(shape=(len(cutoffdnn_output),len(cutoffdnn_output)))
          

        for sitedictindex1 in range(0,len(cutoffdnn_output)):
            for sitedictindex2 in range(0,sitedictindex1):
                """Reason for jimage=[0,0,0]
                
                site.distance is calculated by frac_coord1-frac_coord0 and get the cartesian distance. Note that for the two sites in neighbors,  the frac_coord itself already contains the information of jimage. For exaple:Si4+ (-3.2361, -0.3015, 9.2421) [-0.3712, -0.0379, 0.4167], 'image': (-1, -1, 0),  see that the frac_coord of this Si4+ is not normalized to (0,1)!

                .
                """
                distance_matrix[sitedictindex1][sitedictindex2]=cutoffdnn_output[sitedictindex1]["site"].distance(cutoffdnn_output[sitedictindex2]["site"],jimage=[0,0,0])
            
            
        
        return distance_matrix

    def brutal_match(self,unsorted_nninfo=[{}],rtol=0.01,atol=0.01,find_nearest_if_fail=True):
        """brutally sort the inputted unsorted_nninfo. Although brutal but fast enough for now

        Args:
            unsorted_nninfo (list, optional): the unsorted nn_info of an element. The nn_info are compared with the nn_info of class instance. Defaults to [{}].
            rtol (float, optional): rtolerance of np.allclose in order to determine if the distance matrix are the same. Better not too small. Defaults to 0.01.

        Raises:
            ValueError: this will perform a check if the inputted unsorted nn_info has the same neighbor species type and amount
            ValueError: if the unsorted_nninfo cannot be sort with reference of the distance matrix of this neighbor_info_matcher instance. Probably due to too small rtol or it's just not the same neighbor_infos

        Returns:
            list of dictionary: in the format of cutoffdictNN.get_nn_info
        """
        
        unsorted_neighbor_info=neighbor_info_matcher.from_neighbor_sequences(unsorted_nninfo)
        
        if self.neighbor_species!=unsorted_neighbor_info.neighbor_species:
            raise ValueError("input neighbor_info has different environment")
        
        if np.allclose(unsorted_neighbor_info.distance_matrix,self.distance_matrix,rtol=rtol,atol=atol):
            neighbor_info_logger.info("no need to rearrange this neighbor_info. The distance matrix is already the same. The differece matrix is : \n")
            neighbor_info_logger.info(str(unsorted_neighbor_info.distance_matrix-self.distance_matrix))
            
            return unsorted_neighbor_info.neighbor_sequence
        
        
        
        sorted_neighbor_sequence_dict={}

        for specie in unsorted_neighbor_info.neighbor_species_respective_neighbor_sequence_dict:
            
            sorted_neighbor_sequence_dict[specie]=[]

            for possible_local_sequence in itertools.permutations(unsorted_neighbor_info.neighbor_species_respective_neighbor_sequence_dict[specie]):
                
                if np.allclose(self.build_distance_matrix_from_getnninfo_output(possible_local_sequence),self.neighbor_species_respective_distance_matrix_dict[specie],rtol=rtol,atol=atol):
                    
                    sorted_neighbor_sequence_dict[specie].append(list(possible_local_sequence))
            
            if len(sorted_neighbor_sequence_dict[specie])==0:
                raise ValueError("no sorted sequence found for "+str(specie)+" please check if the rtol or atol is too small")
            
        #neighbor_info_logger.warning(str(sorted_neighbor_sequence_dict))
        
        
        sorted_neighbor_sequence_list=[]
        
        for specie in sorted_neighbor_sequence_dict:
            sorted_neighbor_sequence_list.append(sorted_neighbor_sequence_dict[specie])
        
        

        if find_nearest_if_fail:
            closest_smilarity_score=999999.0
            closest_sequence=[]
            for possible_complete_sequence in itertools.product(*sorted_neighbor_sequence_list):
                
  
                re_sorted_neighbors_list=[]
                
                for neighbor in possible_complete_sequence:

                    re_sorted_neighbors_list.extend(list(neighbor))               
            
                this_smilarity_score=np.sum(np.abs(self.build_distance_matrix_from_getnninfo_output(re_sorted_neighbors_list)-self.distance_matrix))
                
                if this_smilarity_score<closest_smilarity_score:
                    closest_smilarity_score=this_smilarity_score
                    closest_sequence=re_sorted_neighbors_list
                    
            neighbor_info_logger.warning("the closest neighbor_info identified. Total difference"+str(closest_smilarity_score))
            neighbor_info_logger.info("new sorting is found,new distance matrix is ")
            neighbor_info_logger.info(str(self.build_distance_matrix_from_getnninfo_output(closest_sequence)))
            neighbor_info_logger.info("The differece matrix is : \n")
            neighbor_info_logger.info(str(self.build_distance_matrix_from_getnninfo_output(closest_sequence)-self.distance_matrix))                            
            return closest_sequence
                
            
      
        else:
        
            for possible_complete_sequence in itertools.product(*sorted_neighbor_sequence_list):
                
                #neighbor_info_logger.warning(str(possible_complete_sequence))
                
                re_sorted_neighbors_list=[]
                
                for neighbor in possible_complete_sequence:

                    re_sorted_neighbors_list.extend(list(neighbor))               
            
                if np.allclose(self.build_distance_matrix_from_getnninfo_output(re_sorted_neighbors_list),self.distance_matrix,rtol=rtol,atol=atol):
                    neighbor_info_logger.info("new sorting is found,new distance matrix is ")
                    neighbor_info_logger.info(str(self.build_distance_matrix_from_getnninfo_output(re_sorted_neighbors_list)))
                    neighbor_info_logger.warning("The differece matrix is : \n")
                    neighbor_info_logger.info(str(self.build_distance_matrix_from_getnninfo_output(re_sorted_neighbors_list)-self.distance_matrix))                
                    
                    return re_sorted_neighbors_list
  
            raise ValueError("sequence not founded!")                
      
      
def find_atom_indices(structure,atom_identifier_type="specie",atom_identifier="Li+"):
    """a function for generating a list of site indices that satisfy the identifier

    Args:
        structure (kmcpy.external.pymatgen_structure): structure object to work on
        atom_identifier_type (str, optional): elect from: ["specie","label","list"]. Defaults to "specie".
        atom_identifier (str, optional): identifier of atom. Defaults to "Li+".
        
        typical input:
        atom_identifier_type=specie, atom_identifier="Li+"
        atom_identifier_type=label, atom_identifier="Li1"
        atom_identifier_type=list, atom_identifier=[0,1,2,3,4,5]

    Raises:
        ValueError: atom_identifier_type argument is strange

    Returns:
        list: list of atom indices that satisfy the identifier
    """
    center_atom_indices=[]    
    if atom_identifier_type=="specie":
        for i in range(0,len(structure)):
            if atom_identifier in structure[i].species:
                center_atom_indices.append(i)
                
    elif atom_identifier_type=="label":

        for i in range(0,len(structure)):
            if structure[i].properties["label"]==atom_identifier:
                center_atom_indices.append(i)
                
    elif atom_identifier_type=="list":
        center_atom_indices=atom_identifier
    
    else:
        raise ValueError('unrecognized atom_identifier_type. Please select from: ["specie","label","list"] ')
    
    neighbor_info_logger.warning("please check if these are center atom:")
    for i in center_atom_indices:
        
        neighbor_info_logger.warning(str(structure[i]))        
    
    return center_atom_indices
        