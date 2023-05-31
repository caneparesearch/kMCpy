#!/usr/bin/env python

from turtle import distance
from sympy import primitive
from kmcpy.dummy_center.event import Event
import numpy as np
from numba.typed import List
import numba as nb
from kmcpy.io import convert
import itertools
import logging
from pymatgen.util.coord import get_angle

def print_divider():
    print("\n\n-------------------------------------------\n\n")

# Jerry: divider is used to seperate a log file into sections
def get_divider(title,string):
    return "\n"+string*20+title+string*20+"\n"

def generate_events(api=3,**kwargs):

    return generate_events3(**kwargs)




neighbor_info_logger=logging.getLogger("neighbor information:")
neighbor_info_logger.setLevel("INFO")
neighbor_info_logger.addHandler(logging.StreamHandler())
neighbor_info_logger.addHandler(logging.FileHandler("neighbor_info_debug.log"))

class neighbor_info_matcher():
    def __init__(self,neighbor_species=(('Cl-', 4),('Li+', 8)),distance_matrix=np.array([[0,1],[1,0]]),neighbor_sequence=[{}],neighbor_species_distance_matrix_dict={"Cl-":np.array([[0,1],[1,0]]),"Li+":np.array([[0,1],[1,0]])},neighbor_species_sequence_dict={"Cl-":[{}],"Li+":[{}]}):
        """neighbor_info matcher, the __init__ method shouln't be used. Use the from_neighbor_info() instead. This is neighbor_info matcher to match the nearest neighbor info output from local_env.cutoffdictNN.get_nn_info. This neighbor_info_matcher Class is initialized by a reference neighbor_info, a distance matrix is built as reference. Then user can call the neighbor_info_matcher.brutal_match function to sort another nn_info so that the sequence of neighbor of "another nn_info" is arranged so that the distance matrix are the same 

        Args:
            neighbor_species (tuple, optional): tuple ( tuple ( str(species), int(number_of_this_specie in neighbors)  )  ). Defaults to (('Cl-', 4),('Li+', 8)).
            distance_matrix (np.array, optional): np.2d array as distance matrix. Defaults to np.array([[0,1],[1,0]]).
            neighbor_sequence (list, optional): list of dictionary in the format of nn_info returning value. Defaults to [{}].
            neighbor_species_distance_matrix_dict (dict, optional): this is a dictionary with key=species and value=distance_matrix(2D numpy array) which record the distance matrix of respective element. . Defaults to {"Cl-":np.array([[0,1],[1,0]]),"Li+":np.array([[0,1],[1,0]])}.
            neighbor_species_sequence_dict (dict, optional): dictionary with key=species and value=list of dictionary which is just group the reference neighbor sequence by different elements. Defaults to {"Cl-":[{}],"Li+":[{}]}.
        """
        
        self.neighbor_species=neighbor_species
        self.distance_matrix=distance_matrix
        self.neighbor_species_distance_matrix_dict=neighbor_species_distance_matrix_dict
        self.neighbor_species_sequence_dict=neighbor_species_sequence_dict
        self.neighbor_sequence=neighbor_sequence
    
    @classmethod
    def from_neighbor_sequences(self,neighbor_sequences=[{}]):
        """generally generate the neighbor info matcher from this

        Args:
            neighbor_sequences (list, optional): list of dictionary from cutoffdictNN.get_nn_info(). Defaults to [{}].

        Returns:
            neighbor_info_matcher: a neighbor_info_matcher object, initialized from get_nn_info output
        """
        
        # -------------------------------------------------------
        # this part of function is adapted from pymatgen.analysis.local_env
        #Shyue Ping Ong, William Davidson Richards, Anubhav Jain, Geoffroy Hautier, Michael Kocher, Shreyas Cholia, Dan Gunter, Vincent Chevrier, Kristin A. Persson, Gerbrand Ceder. Python Materials Genomics (pymatgen) : A Robust, Open-Source Python Library for Materials Analysis. Computational Materials Science, 2013, 68, 314-319. doi:10.1016/j.commatsci.2012.10.028
        cn_dict={}
        
        neighbor_species_sequence_dict={}
        
        for neighbor in neighbor_sequences:
            """for example NaSICON has 12 neighbors, 6 Na and 6 Si, here for neighbor we build the coordination number dict.
            """
            
            site_element = neighbor["site"].species_string
            
            if site_element not in cn_dict:
                cn_dict[site_element] = 1
            else:
                cn_dict[site_element] += 1
            if site_element not in neighbor_species_sequence_dict:
                neighbor_species_sequence_dict[site_element]=[neighbor]
            else:
                neighbor_species_sequence_dict[site_element].append(neighbor)
        # end of adapting.
        # -------------------------------------------------------        
            
        neighbor_species_distance_matrix_dict={}
        
        for species in neighbor_species_sequence_dict:
            neighbor_species_distance_matrix_dict[species]=self.build_distance_matrix_from_getnninfo_output(neighbor_species_sequence_dict[species])
        
        neighbor_species=tuple(sorted(cn_dict.items(),key=lambda x:x[0]))
        
        distance_matrix=self.build_distance_matrix_from_getnninfo_output(neighbor_sequences)
        
        
         
        return neighbor_info_matcher(neighbor_species=neighbor_species,distance_matrix=distance_matrix,neighbor_sequence=neighbor_sequences,neighbor_species_distance_matrix_dict=neighbor_species_distance_matrix_dict,neighbor_species_sequence_dict=neighbor_species_sequence_dict)
        
    
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
            for sitedictindex2 in range(0,len(cutoffdnn_output)):
                """Reason for jimage=[0,0,0]
                
                site.distance is calculated by frac_coord1-frac_coord0 and get the cartesian distance. Note that for the two sites in neighbors,  the frac_coord itself already contains the information of jimage. For exaple:Si4+ (-3.2361, -0.3015, 9.2421) [-0.3712, -0.0379, 0.4167], 'image': (-1, -1, 0),  see that the frac_coord of this Si4+ is not normalized to (0,1)!

                .
                """
                distance_matrix[sitedictindex1][sitedictindex2]=cutoffdnn_output[sitedictindex1]["site"].distance(cutoffdnn_output[sitedictindex2]["site"],jimage=[0,0,0])
            
            
        
        return distance_matrix

    @classmethod
    def build_angle_matrix_from_getnninfo_output(self,cutoffdnn_output=[{}]):
        """build a distance matrix from the output of CutOffDictNN.get_nn_info

        nn_info looks like: 
        [{'site': PeriodicSite: Si4+ (-3.2361, -0.3015, 9.2421) [-0.3712, -0.0379, 0.4167], 'image': (-1, -1, 0), 'weight': 3.7390091507903174, 'site_index': 39, 'wyckoff_sequence': 15, 'local_index': 123, 'label': 'Si1'}, {'site': PeriodicSite: Na+ (-1.2831, -2.6519, 9.2421) [-0.3063, -0.3333, 0.4167], 'image': (-1, -1, 0), 'weight': 3.4778161424304046, 'site_index': 23, 'wyckoff_sequence': 17, 'local_index': 35, 'label': 'Na2'}, {'site': ...]
        
        or say:
        
        nn_info is a list, the elements of list is dictionary, the keys of dictionary are: "site":pymatgen.site, "wyckoff_sequence": ....
        
        Use the site.distance function to build matrix
        

        Args:
            cutoffdnn_output (nn_info, optional): nninfo. Defaults to neighbor_sequences.

        Returns:
            np.3darray: 3d distance matrix, in format of numpy.array. The Column and the Rows are following the input sequence.
        """
    
        angle_matrix=np.zeros(shape=(len(cutoffdnn_output),len(cutoffdnn_output),len(cutoffdnn_output)))
          
        
        for sitedictindex1 in range(0,len(cutoffdnn_output)):
            for sitedictindex2 in range(0,len(cutoffdnn_output)):
                for sitedictindex3 in range(0,len(cutoffdnn_output)):
                    v1=cutoffdnn_output[sitedictindex2]["site"].coords-cutoffdnn_output[sitedictindex1]["site"].coords
                    v2=cutoffdnn_output[sitedictindex3]["site"].coords-cutoffdnn_output[sitedictindex1]["site"].coords
                    angle_matrix[sitedictindex1][sitedictindex2][sitedictindex3]=get_angle(v1,v2,"degrees")
            
            
        
        return angle_matrix

    def rearrange(self,wrong_distance_matrix_of_specie=[],species='Na',atol=0.01,rtol=0.01):
        """
        A very fast version of rearranging the neighbors with same species

        Args:
            wrong_distance_matrix_of_specie (np.2Drray, optional): distance matrix of wrong distance matrix that is supposed to be rearranged and match with self.distance matrix. Defaults to [].
            species (str, optional): the species to match, is the key to self.neighbor_species_distance_matrix_dict. Defaults to 'Na'.

        Raises:
            ValueError: no correct sequence found

        Returns:
            list: list of list, of which is the sequence of index of rearranged sequence
        """
        # distance_matrix=distance matrix of wrong neighbor sequence
        correct_distance_matrix=self.neighbor_species_distance_matrix_dict[species]# np.2darray
        previous_possible_sequences=[]
        
        for i in range(0,len(correct_distance_matrix)):
            previous_possible_sequences.append([i])# init
        
        for i in range(0,len(correct_distance_matrix)-1):
        
            new_possible_sequences=[]
            
            correct_distance_matrix_in_this_round=correct_distance_matrix[0:2+i,0:2+i]
            
            
            for previous_possible_sequence in previous_possible_sequences:
                for i in range(0,len(correct_distance_matrix)):
                    if i not in previous_possible_sequence:
                        tmp_sequence=previous_possible_sequence.copy()
                        #print(tmp_sequence)
                        tmp_sequence.append(i)
                        tmp_rebuilt_submatrix=self.rebuild_submatrix(distance_matrix=wrong_distance_matrix_of_specie,sequences=tmp_sequence)
                        if np.allclose(tmp_rebuilt_submatrix,correct_distance_matrix_in_this_round,atol=atol,rtol=rtol):
                            new_possible_sequences.append(tmp_sequence)
                            
            previous_possible_sequences=new_possible_sequences.copy()
            
        if len(new_possible_sequences)==0:
            raise ValueError("new possible sequence=0.")

        return new_possible_sequences
            
    def rebuild_submatrix(self,distance_matrix,sequences=[2,1]):
        """rebuild the submatrix, with given seuqneces and distance matrix, rebuild the distance matrix from given sequence

        Args:
            distance_matrix (np.2Darray): distance matrix, length = sequences.len()
            sequences (list, optional): new sequences. Defaults to [2,1].

        Returns:
            np.2darray: rebuilt matrix
        """
        # wrong_distance_matrix is the matrix that is different from the reference.
        rebuilt_matrix=np.zeros(shape=(len(sequences),len(sequences)))
        
        for idx1 in range(len(sequences)):
            for idx2 in range(len(sequences)):
                rebuilt_matrix[idx1][idx2]=distance_matrix[sequences[idx1]][sequences[idx2]]
        return rebuilt_matrix


    def brutal_match(self,unsorted_nninfo=[{}],rtol=0.001,atol=0.001,find_nearest_if_fail=False):
        """brutally sort the input unsorted_nninfo. Although brutal but fast enough for now
        
        update 220621: not fast enough for LiCoO2 with 12 neighbors,
        
        rewrite the finding sequence algo. Now is freaking fast again!

        Args:
            unsorted_nninfo (list, optional): the unsorted nn_info of an element. The nn_info are compared with the nn_info of class instance. Defaults to [{}].
            
            Tolerance: Refer to np.allclose
            rtol (float, optional): relative tolerance of np.allclose in order to determine if the distance matrix are the same. Better not too small. Defaults to 0.01.
            atol (float, optional): absolute tolerance
            find_nearest_if_fail(bool, optional): This should be true only for grain boundary model! 

        Raises:
            ValueError: this will perform a check if the inputted unsorted nn_info has the same neighbor species type and amount
            ValueError: if the unsorted_nninfo cannot be sort with reference of the distance matrix of this neighbor_info_matcher instance. Probably due to too small rtol or it's just not the same neighbor_infos

        Returns:
            sorted nninfo, as list of dictionary: in the format of cutoffdictNN.get_nn_info
        """
        
        unsorted_neighbor_info=neighbor_info_matcher.from_neighbor_sequences(unsorted_nninfo)
        
        if self.neighbor_species!=unsorted_neighbor_info.neighbor_species:
            raise ValueError("input neighbor_info has different environment")
        
        if np.allclose(unsorted_neighbor_info.distance_matrix,self.distance_matrix,rtol=rtol,atol=atol):
            neighbor_info_logger.info("no need to rearrange this neighbor_info. The distance matrix is already the same. The differece matrix is : \n")
            neighbor_info_logger.info(str(unsorted_neighbor_info.distance_matrix-self.distance_matrix))
            
            return unsorted_neighbor_info.neighbor_sequence
        
        sorted_neighbor_sequence_dict={}

        for specie in unsorted_neighbor_info.neighbor_species_sequence_dict:
            
            rearranged_sequences_of_neighbor=self.rearrange(wrong_distance_matrix_of_specie=unsorted_neighbor_info.neighbor_species_distance_matrix_dict[specie],species=specie,atol=atol,rtol=rtol)
            
            sorted_neighbor_sequence_dict[specie]=[]
            for rearranged_sequence_of_neighbor in rearranged_sequences_of_neighbor:
                possible_local_sequence=[]
                for new_index in rearranged_sequence_of_neighbor:
                    possible_local_sequence.append(unsorted_neighbor_info.neighbor_species_sequence_dict[specie][new_index])
                sorted_neighbor_sequence_dict[specie].append(possible_local_sequence)
            
            """
            print(sorted_neighbor_sequence_dict[specie])
            raise ValueError()
            
            for possible_local_sequence in itertools.permutations(unsorted_neighbor_info.neighbor_species_sequence_dict[specie]):
                
                if np.allclose(self.build_distance_matrix_from_getnninfo_output(possible_local_sequence),self.neighbor_species_distance_matrix_dict[specie],rtol=rtol,atol=atol):
                    
                    sorted_neighbor_sequence_dict[specie].append(list(possible_local_sequence))
            """
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
  

def find_atom_indices(structure,mobile_ion_identifier_type="specie",atom_identifier="Li+"):
    """a function for generating a list of site indices that satisfies the identifier

    Args:
        structure (kmcpy.external.pymatgen_structure): structure object to work on
        mobile_ion_identifier_type (str, optional): elect from: ["specie","label"]. Defaults to "specie".
        atom_identifier (str, optional): identifier of atom. Defaults to "Li+".
        
        typical input:
        mobile_ion_identifier_type=specie, atom_identifier="Li+"
        mobile_ion_identifier_type=label, atom_identifier="Li1"
        mobile_ion_identifier_type=list, atom_identifier=[0,1,2,3,4,5]

    Raises:
        ValueError: mobile_ion_identifier_type argument is strange

    Returns:
        list: list of atom indices that satisfy the identifier
    """
    mobile_ion_specie_1_indices=[]    
    if mobile_ion_identifier_type=="specie":
        for i in range(0,len(structure)):
            if atom_identifier in structure[i].species:
                mobile_ion_specie_1_indices.append(i)
                
    elif mobile_ion_identifier_type=="label":

        for i in range(0,len(structure)):
            if structure[i].properties["label"]==atom_identifier:
                mobile_ion_specie_1_indices.append(i)
                
    #elif mobile_ion_identifier_type=="list":
        #mobile_ion_specie_1_indices=atom_identifier
    
    else:
        raise ValueError('unrecognized mobile_ion_identifier_type. Please select from: ["specie","label"] ')
    
    neighbor_info_logger.warning("please check if these are mobile_ion_specie_1:")
    for i in mobile_ion_specie_1_indices:
        
        neighbor_info_logger.warning(str(structure[i]))        
    
    return mobile_ion_specie_1_indices
        
def generate_events3(prim_cif_name="210.cif",
                         convert_to_primitive_cell=False,
                         local_env_cutoff_dict={("Li+","Cl-"):4.0,("Li+","Li+"):3.0},
                         mobile_ion_identifier_type="label",
                         mobile_ion_specie_1_identifier="Na1",
                         mobile_ion_specie_2_identifier="Na2",
                         migration_unit_center_identifier_type="label",
                         migration_unit_center_identifier = "X",
                         migration_distance_range = [3,4],
                         species_to_be_removed=["O2-","O"],
                         distance_matrix_rtol=0.01,
                         distance_matrix_atol=0.01,
                         find_nearest_if_fail=True,
                         export_local_env_structure=True,
                         supercell_shape=[2,1,1],
                         event_fname="events.json",
                         event_kernal_fname='event_kernal.csv',
                         verbosity="WARNING",**kwargs):
    """
Jerry: modified based on the generate_events3 for Na3-xSb1-xWxS4
    Using the center of the MigrationUnit as an identifier for the event
    Migration happens between mobile_ion_specie_1 <-> mobile_ion_specie_2 
    migration_distance_range (in Angstrom) is used to select the desired type of migration events, e.g. between nearest neighbor, between 2nd nearest neighbor, and etc.
    
    Args:
        prim_cif_name (str, optional): the file name of primitive cell of KMC model. Strictly limited to cif file because only cif parser is capable of taking label information of site. This cif file should include all possible site i.e., no vacancy. For example when dealing with NaSICON The input cif file must be a fully occupied composition, which includes all possible Na sites N4ZSP; the varied Na-Vacancy should only be tuned by occupation list.
        convert_to_primitive_cell (bool, optional): whether convert to primitive cell. For rhombohedral, if convert_to_primitive_cell, will use the rhombohedral primitive cell, otherwise use the hexagonal primitive cell. Defaults to False.
        local_env_cutoff_dict (dict, optional): cutoff dictionary for finding the local environment. This will be passed to local_env.cutoffdictNN`. Defaults to {("Li+","Cl-"):4.0,("Li+","Li+"):3.0}.
        mobile_ion_identifier_type (str, optional): atom identifier type, choose from ["specie", "label"].. Defaults to "specie".
        mobile_ion_specie_1_identifier (str, optional): identifier for mobile_ion_specie_1. Defaults to "Li+".
        mobile_ion_specie_2_identifier (str, optional): identifier for the atom that mobile_ion_specie_1 will diffuse to . Defaults to "Li+".
        species_to_be_removed (list, optional): list of species to be removed, those species are not involved in the KMC calculation. Defaults to ["O2-","O"].
        distance_matrix_rtol (float, optional): r tolerance of distance matrix for determining whether the sequence of neighbors are correctly sorted in local envrionment. For grain boundary model, please allow the rtol up to 0.2~0.4, for bulk model, be very strict to 0.01 or smaller. Smaller rtol will also increase the speed for searching neighbors. Defaults to 0.01.
        distance_matrix_atol (float, optional): absolute tolerance , . Defaults to 0.01.
        find_nearest_if_fail (bool, optional): if fail to sort the neighbor with given rtolerance and atolerance, find the best sorting that have most similar distance matrix? This should be False for bull model because if fail to find the sorting ,there must be something wrong. For grain boundary , better set this to True because they have various coordination type. Defaults to True.
        export_local_env_structure (bool, optional): whether to export the local environment structure to cif file. If set to true, for each representatibe local environment structure, a cif file will be generated for further investigation. This is for debug purpose. Once confirming that the code is doing correct thing, it's better to turn off this feature. Defaults to True.
        supercell_shape (list, optional): shape of supercell passed to the kmc_build_supercell function, array type that can be 1D or 2D. Defaults to [2,1,1].
        event_fname (str, optional): file name for the events json file. Defaults to "events.json".
        event_kernal_fname (str, optional): file name for event kernal. Defaults to 'event_kernal.csv'.
        verbosity (str, optional): verbosity that passed to logging.logger. Select from ["INFO","warning"], higher level not yet implemented. Defaults to "INFO".

    Raises:
        NotImplementedError: the atom identifier type=list is not yet implemented
        ValueError: unrecognized atom identifier type 
        ValueError: if no events are generated, there might be something wrong with cif file? or atom identifier?

    Returns:
        nothing: nothing is returned
    """

    # --------------
    import json
    import logging
    from kmcpy.external.pymatgen_structure import Structure
    from kmcpy.external.pymatgen_local_env import CutOffDictNN

    from kmcpy.io import convert
    from kmcpy.event import Event

    # Jerry: add this
    from kmcpy.external.pymatgen_local_env import CutOffDictNNrange

    
    # build the logger
    event_generator_logger=logging.getLogger("event generator")
    event_generator_logger.setLevel(verbosity)
    event_generator_logger.addHandler(logging.StreamHandler())
    event_generator_logger.addHandler(logging.FileHandler("event_generator_debug.log"))
    event_generator_logger.warning("Extracting clusters from primitive cell structure. This primitive cell should be bulk structure, grain boundary model not implemented yet.")
    
    
    # generate primitive cell
    primitive_cell=Structure.from_cif(prim_cif_name,primitive=convert_to_primitive_cell)
    #primitive_cell.add_oxidation_state_by_element({"Na":1,"O":-2,"P":5,"Si":4,"V":2.5})
    primitive_cell.add_oxidation_state_by_guess()
    
    primitive_cell.remove_species(species_to_be_removed)
    event_generator_logger.info(get_divider('Generate Primitve Cell','='))

    event_generator_logger.warning("primitive cell composition after adding oxidation state and removing uninvolved species: ")
    event_generator_logger.info(str(primitive_cell.composition))
    event_generator_logger.info(get_divider('Generate Primitve Cell','='))

    event_generator_logger.warning("building migration_unit_center index list")
    
    # find indices of all Na+
    # mobile_ion_specie_1_indices=find_atom_indices(primitive_cell,
    #                                               mobile_ion_identifier_type=mobile_ion_identifier_type,
    #                                               atom_identifier=mobile_ion_specie_1_identifier)  
    migration_unit_center_indicies=find_atom_indices(primitive_cell,
                                                  mobile_ion_identifier_type=migration_unit_center_identifier_type,
                                                  atom_identifier=migration_unit_center_identifier)  
    #--------
    
    local_env_finder = CutOffDictNN(local_env_cutoff_dict)

    reference_local_env_dict={}
    """this is aimed for grain boundary model. For bulk model, there should be only one type of reference local environment. i.e., len(reference_local_env_dict)=1
    
    The key is a tuple type: For NaSICON, there is only one type of local environment: 6 Na2 and 6 Si. The tuple is in the format of (('Na+', 6), ('Si4+', 6)) . The key is a neighbor_info_matcher as the reference local environment.
    """
    
    
    local_env_info_dict = {}
    """add summary to be done

    """
   
    reference_local_env_type=0

    event_generator_logger.info(get_divider('Begin: Find Local Env in Primitve Cell','='))
    event_generator_logger.info("start finding the neighboring sequence of migration_unit_center_indicies")
    event_generator_logger.info("total number of migration_unit_center_indicies:"+str(len(migration_unit_center_indicies)))
    
    neighbor_has_been_found=0
    
    # Jerry: through migration unit center
    for migration_unit_center_index in migration_unit_center_indicies:
        event_generator_logger.info(get_divider('Begin: Loop migration_unit_center','-'))

        unsorted_neighbor_sequences=sorted(sorted(local_env_finder.get_nn_info(primitive_cell,migration_unit_center_index),
                                                  key=lambda x:x["site"].coords[0]),
                                                  key = lambda x:x["site"].specie)      

        this_nninfo=neighbor_info_matcher.from_neighbor_sequences(unsorted_neighbor_sequences)
        
        #print(this_nninfo.build_angle_matrix_from_getnninfo_output(primitive_cell,unsorted_neighbor_sequences))
        # Jerry: reference_local_env_dict: unique local environment
        if this_nninfo.neighbor_species not in reference_local_env_dict:
            
            # then take this as the reference neighbor info sequence
            
            # Jerry: this is not relavent to the project, ignored
            if export_local_env_structure:
                
                reference_local_env_sites=[primitive_cell[migration_unit_center_index]]
                
                for i in unsorted_neighbor_sequences:
                    
                    reference_local_env_sites.append(i["site"])
                    reference_local_env_structure=Structure.from_sites(sites=reference_local_env_sites)
                    
                reference_local_env_structure.to("cif",str(reference_local_env_type)+"th_reference_local_env.cif") 
                reference_local_env_type+=1
                
                event_generator_logger.info(str(reference_local_env_type)+"th type of reference local_env structure cif file is created. please check")
            
            # Jerry: add new unique local_env to the database
            reference_local_env_dict[this_nninfo.neighbor_species]=this_nninfo

            local_env_info_dict[primitive_cell[migration_unit_center_index].properties['local_index']]=this_nninfo.neighbor_sequence
            
            
            event_generator_logger.warning("a new type of local environment is recognized with the species "+str(this_nninfo.neighbor_species)+" \nthe distance matrix are \n"+str(this_nninfo.distance_matrix))
            event_generator_logger.info('The shape of the generated distance matrix is:'+str(this_nninfo.distance_matrix.shape))

            
        else:
            event_generator_logger.info("a local environment is created with the species "+str(this_nninfo.neighbor_species)+" \nthe distance matrix are \n"+str(this_nninfo.distance_matrix))
            
            

            sorted_neighbor_sequence=reference_local_env_dict[this_nninfo.neighbor_species].brutal_match(this_nninfo.neighbor_sequence,rtol=distance_matrix_rtol,atol=distance_matrix_atol,find_nearest_if_fail=find_nearest_if_fail)
            
            local_env_info_dict[primitive_cell[migration_unit_center_index].properties['local_index']]=sorted_neighbor_sequence
            
        neighbor_has_been_found+=1
        
        event_generator_logger.warning(str(neighbor_has_been_found)+" out of "+str(len(migration_unit_center_indicies))+" neighboring sequence has been found")
        event_generator_logger.info(get_divider('End: Loop migration_unit_center','-'))
    event_generator_logger.info(get_divider('End: Find Local Env in Primitve Cell','='))
    # Jerry: create supercell after finding all unique local env in primtive cell
    event_generator_logger.info(get_divider('Start: Generate Supercell','='))
    supercell=primitive_cell.make_kmc_supercell(supercell_shape)
    event_generator_logger.warning("supercell is created")
    event_generator_logger.info(str(supercell))
    event_generator_logger.info(get_divider('End: Generate Supercell','='))

    # Jerry: find all migration_unit_center_indices in supercell
    supercell_migration_unit_center_indices=find_atom_indices(supercell,mobile_ion_identifier_type=migration_unit_center_identifier_type,atom_identifier=migration_unit_center_identifier)

    events = []
    events_dict = []
    
    # Jerry advice: it's better to re-implement this function... this dictionary is used to get the site index in supercell by giving supercell tuple, label, and sublattice index
    indices_dict_from_identifier=supercell.kmc_build_dict3(skip_check=False)#a dictionary. Key is the tuple with format of ([supercell[0],supercell[1],supercell[2],label,local_index]) that contains the information of supercell, local index (index in primitive cell), Value is the corresponding global site index.  This hash dict for acceleration purpose
    event_generator_logger.info(get_divider('Start: Generate events in supercell','='))
    # Jerry: loop through all migration_unit_center in supercell
    event_generator_logger.info('supercell_migration_unit_center_indices'+str(supercell_migration_unit_center_indices))

    event_generator_logger.info(get_divider('Begin: Loop migration unit in supercell','-'))
    for supercell_migration_unit_center_index in supercell_migration_unit_center_indices:
        # for mobile_ion_specie_1s of newly generated supercell, find the neighbors
        event_generator_logger.info('migration_unit_center at site['+str(supercell_migration_unit_center_index)+'] in supercell: '+str(supercell[supercell_migration_unit_center_index]))
        
        supercell_tuple=supercell[supercell_migration_unit_center_index].properties["supercell"]
        # Jerry: local_index_of_migration_unit_center is the sublattice index in the primitive cell
        local_index_of_migration_unit_center=supercell[supercell_migration_unit_center_index].properties["local_index"]
        
        local_env_info=[]# list of integer / indices of local environment
        # Jerry: loop through sites in the local environment <- here we should loop mobile_specie_pairs within each migration unit?
        event_generator_logger.info(get_divider('Begin: Loop sites in local env','.'))
        for neighbor_site_in_primitive_cell in local_env_info_dict[local_index_of_migration_unit_center]:
            event_generator_logger.info('\nneighbor_site_in_primitive_cell: '+str(neighbor_site_in_primitive_cell))
            """
            
             local_env_info_dict[local_index_of_migration_unit_center]
            
            In primitive cell, the mobile_ion_specie_1 has 1 unique identifier: The "local index inside the primitive cell"
            
            In supercell, the mobile_ion_specie_1 has an additional unique identifier: "supercell_tuple"
            
            However, as long as the "local index inside the primitive cell" is the same, 
            no matter which supercell this mobile_ion_specie_1 belongs to, 
            the sequence of "local index" of its neighbor sites are the same. 
            For example, In primitive cell, mobile_ion_specie_1 with index 1 has neighbor arranged in 1,3,2,4,6,5, 
            then for every mobile_ion_specie_1 with index 1 in supercell, the neighbor is arranged in 1,3,2,4,6,5
            
            In the loop. I'm mapping the sequence in primitive cell to supercell!
            
            In order to accelerate the speed

            use the dictionary to store the index of atoms
            
            indices_dict_from_identifier is a dictionary by pymatgen_structure.kmc_build_dict(). Key is the tuple with format of ([supercell[0],supercell[1],supercell[2],label,local_index]) that contains the information of supercell, local index (index in primitive cell), Value is the corresponding global site index.  This hash dict for acceleration purpose
            
            This loop build the local_env_info list, [supercell_neighbor_index1, supercell_neighbor_index2, .....]
            
            """
            # Jerry: normalize_supercell_tuple returns the correct supercell image tuple of a site
            normalized_supercell_tuple=normalize_supercell_tuple(site_belongs_to_supercell=supercell_tuple,image_of_site=neighbor_site_in_primitive_cell["image"],supercell_shape=supercell_shape)
            event_generator_logger.info('normalized_supercell_tuple: '+str(normalized_supercell_tuple))
            # Jerry: tuple_key_of_such_neighbor_site is the site index in a supercell by giving ([supercell[0],supercell[1],supercell[2],label,local_index])
            tuple_key_of_such_neighbor_site=supercell.site_index_vector(local_index=neighbor_site_in_primitive_cell["local_index"],label=neighbor_site_in_primitive_cell["label"],supercell=normalized_supercell_tuple)
            event_generator_logger.info('tuple_key_of_such_neighbor_site: '+str(tuple_key_of_such_neighbor_site))
            local_env_info.append(indices_dict_from_identifier[tuple_key_of_such_neighbor_site])
            # event_generator_logger.info('indices_dict_from_identifier[tuple_key_of_such_neighbor_site]',str(indices_dict_from_identifier[tuple_key_of_such_neighbor_site])+'\n')
        event_generator_logger.info(get_divider('End: Loop sites in local env','.'))
        
        event_generator_logger.info(get_divider('Start: Generate Events','.'))
        # Jerry: start to generate Event
        # Jerry: loop through sites in local env and generate event when this site is the mobile species with addition condition
        mobile_ion_indices = []
        #already_appended

        for local_env in local_env_info:
            # generate event
            """
            generally use the migration_unit_center_identifier_type="label", to see the Na1 and Na2 of nasicon.

            specie is suitable for grain boundary model. Generally don't use it


            """
            this_migration_unit_center = supercell[supercell_migration_unit_center_index]
            dist_condition = (this_migration_unit_center.distance(supercell[local_env])<=migration_distance_range[1]) and (this_migration_unit_center.distance(supercell[local_env])>=migration_distance_range[0])
            event_generator_logger.info('distance between migration_unit_center:'+str(supercell_migration_unit_center_index)+ ' and mobile_specie:'+str(local_env)+' : '+str(this_migration_unit_center.distance(supercell[local_env])))
            event_generator_logger.info('dist_condition: '+str(dist_condition))
            
            
            if migration_unit_center_identifier_type=="specie":
                if migration_unit_center_identifier in supercell[local_env].species and dist_condition:
                    mobile_ion_indices.append(local_env)
 
                        
            elif migration_unit_center_identifier_type=="label":                
                if (supercell[local_env].properties["label"] in [mobile_ion_specie_1_identifier, mobile_ion_specie_2_identifier]) and dist_condition:
                    # or for understanding, if any site in local environment, its label== "Na2"
                    # initialize the event
                    mobile_ion_indices.append(local_env)
 
                        
            elif migration_unit_center_identifier_type=="list":
                raise NotImplementedError() 
                #"potentially implement this for grain boundary model"
                        
            else:
                raise ValueError('unrecognized migration_unit_center_identifier_type. Please select from: ["specie","label"] ')
            
        event_generator_logger.info('\nmobile_ion_indices: '+str(mobile_ion_indices))
        if len(mobile_ion_indices) > 2:
            raise NotImplementedError("Find hopping event involving more than 2 sites, currently it's not implemented!")
        
        if len(mobile_ion_indices) < 2:
            raise ValueError('No hopping between two sites has been found: len(mobile_ion_indices) < 2')
        this_event = Event()
        this_event.initialization3(mobile_ion_indices[0],mobile_ion_indices[1],local_env_info)
        events.append(this_event)
        events_dict.append(this_event.as_dict())
        this_event.show_info()

        event_generator_logger.info('\nsupercell_migration_unit_center_index: '+str(supercell_migration_unit_center_index))
        event_generator_logger.info('local_env: '+str(local_env))
        event_generator_logger.info('local_env_info: '+str(local_env_info)+'\n')
        event_generator_logger.info(get_divider('End: Generate Events','.'))
        
        event_generator_logger.info(get_divider('End: Loop migration unit in supercell','-'))
    event_generator_logger.info(get_divider('End: Generate events in supercell','='))
    if len(events)==0:
        raise ValueError("There is no event generated. This is probably caused by wrong input parameters.")
    print('Saving:',event_fname)
    with open(event_fname,'w') as fhandle:
        jsonStr = json.dumps(events_dict,indent=4,default=convert)
        fhandle.write(jsonStr)
    
    events_site_list = []

    for event in events:
        # sublattice indices: local site index for each site
        events_site_list.append(event.local_env_indices_list)
    
    #np.savetxt('./events_site_list.txt',np.array(events_site_list,dtype=int),fmt="%i") # dimension not equal error
    generate_event_kernal(len(supercell),np.array(events_site_list),event_kernal_fname=event_kernal_fname)       
    
    return reference_local_env_dict
    

def normalize_supercell_tuple(site_belongs_to_supercell=[5,1,7],image_of_site=(0,-1,1),supercell_shape=[5,6,7],additional_input=False,verbose=False):
    """finding the equivalent position in periodic supercell considering the periodic boundary condition. i.e., normalize the supercell tuple to make sure that each component of supercell is greater than zero
    
    
    for example,
    
        # 5 1 7 with image 0 -1 1 -> 5 0 8 -> in periodic 567 supercell should change to 561, suppose supercell start with index1
    
    input:
    site_belongs_to_supercell: site belongs to which supercell

    Returns:
        tuple: supercell tuple
    """
    if verbose:
        print ("equivalent position",site_belongs_to_supercell,image_of_site)

    
    supercell=np.array(site_belongs_to_supercell)+np.array(image_of_site)

    supercell=np.mod(supercell,supercell_shape)
    
    supercell=supercell.tolist()
    if additional_input is not False:
        supercell.append(additional_input)
        
    return tuple(supercell)    
    
    
@nb.njit
def _generate_event_kernal(len_structure,events_site_list):
    """to be called by generate_event_kernal for generating the event_kernal.csv
    
    for  a event and find all other event that include the site of this event
    

    Args:
        len_structure (int): _description_
        events_site_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    n_sites  = len_structure
    all_site_list = np.arange(n_sites)
    results = List()
    for site in all_site_list:
        # print('Looking for site:',site)
        row = List()
        #is_Na1=False
        event_index = 0
        for event in events_site_list:
            if site in event:
                row.append(event_index)
            event_index+=1
            #if len(row)==0:
            #    is_Na1=True
        results.append(row)
    return results


def generate_event_kernal(len_structure,events_site_list,event_kernal_fname='event_kernal.csv'):
    """
    event_kernal.csv: 
        event_kernal[i] tabulates the index of sites that have to be updated after event[i] has been executed
        
        
    
    """
    print('Generating event kernal ...')
    event_kernal = _generate_event_kernal(len_structure,events_site_list)
    with open(event_kernal_fname, 'w') as f:
        print('Saving into:',event_kernal_fname)
        for row in event_kernal:
            for item in row:
                f.write('%5d ' % item)
            f.write('\n')
    return event_kernal


