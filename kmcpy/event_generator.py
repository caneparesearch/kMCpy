#!/usr/bin/env python

from sympy import primitive
from kmcpy.event import Event
import numpy as np
from numba.typed import List
import numba as nb
from kmcpy.io import convert
import itertools
import logging

def print_divider():
    print("\n\n-------------------------------------------\n\n")


def generate_events(api=3,**kwargs):
    
    if api==1:
        return generate_events1(**kwargs)
    elif api==2:
        return generate_events2(**kwargs)
    elif api==3:
        return generate_events3(**kwargs)
    else:
        raise NotImplementedError("debug information from event_generator.generate_events. Unsupport API value: api=",api)




def generate_events2(prim_cif_name="EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",convert_to_primitive_cell=True,supercell_shape=[2,1,1],local_env_cutoff_dict = {('Na+','Na+'):4,('Na+','Si4+'):4},event_fname="events.json",event_kernal_fname='event_kernal.csv',center_atom_label_or_indices="Na1",diffuse_to_atom_label="Na2",species_to_be_removed=['Zr4+','O2-','O','Zr'],verbose=False,hacking_arg={1:[27,29,28,30,32,31,117,119,118,120,122,121],2:[21,22,23,32,30,31,111,112,113,122,120,121],3:[18,20,19,34,33,35,108,110,109,124,123,125],5:[21,23,22,24,26,25,111,113,112,114,116,115]},add_oxidation_state=True,rtol_for_neighbor=0.001,export_reference_cluster="reference_cluster.cif"):
    """
    XIE WEIHANG
    
    220603: This function should no longer be used! Just use the generate_event_v3, which is fast and behave more likely as the original generate_events_v1
    
    
    This is 2nd version of generate_events2
    
    methodology: set the 1st center atom as the reference center atom. Set the neighbors sequence (i.e. environment/cluster) of 1st center atom as the reference sequence. Calculate the distance matrix of reference cluster as reference distance matrix following the reference sequence. For all other center atom in the primitive cell, calculate its distance matrix. If the same as reference distance matrix, then pass. If different from reference, then brutally rearrange the sequence of neighbor until the distance matrix are the same

    Args:
        prim_cif_name (str, optional): Path to the cif file. ONLY CIF FILE IS ACCEPTED! REMEMBER TO CHANGE THE label of different sites. Defaults to "EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif".
        supercell_shape (list, optional): list of supercell shape with len=3. Defaults to [2,1,1].
        
        convert_to_primitive_cell(bool, optional): whether convert the input cif file to primitive cell (if R symmetry, from hexagonal to rhombohedral)
        
        local_env_cutoff_dict (dict, optional): cutoff dict for finding neighbors. Must include the charge otherwise cannot find. Defaults to {('Na+','Na+'):4,('Na+','Si4+'):4}.
        
        event_fname (str, optional): output filename of event.json. Defaults to "events.json".
        
        event_kernal_fname (str, optional): output csv file of event kernel, this is : which events need to be updated if event with index=column is updated. Defaults to 'event_kernal.csv'.
        
        center_atom_label_or_indices (str / list , optional): string or list of integers . If passing a string, the program will find the indices of all sites in the cif file which has the label=string. if passing a list, then will treat it as the list of indices of center atom. (center of local cluster). Defaults to "Na1".
        diffuse_to_atom_label (str, optional): diffuse to what atom? . Defaults to "Na2".
        species_to_be_removed (list, optional): the species that do not participate in the local cluster expansion calculation. Defaults to ['Zr4+','O2-','O','Zr'].
        verbose (bool or int, optional): if False, then only limited output. If true, then standard verbose output. If verbose=2, a lot of annoying output. Defaults to False.
        
        hacking_arg (dict, optional): dictionary of hacking arg. The key is the index of center atom, the value is a list, the element of the list is the "local_index" of neighbors. If hacking_ar is present, then when looking for the neighbors and trying to sort them, if the center_atom_index is in the hacking_arg_dict, then the neighbor will be arranged by the sequence of hacking_arg[center_atom_index] The example here is for the NASICON. Defaults to {1:[27,29,28,30,32,31,117,119,118,120,122,121],2:[21,22,23,32,30,31,111,112,113,122,120,121],3:[18,20,19,34,33,35,108,110,109,124,123,125],5:[21,23,22,24,26,25,111,113,112,114,116,115]}.
        
        rtol_for_neighbor(float): tolerance for determining whether the distance matrix is the same. This is passed to np.allclose() function. Default to 0.001. No need to change this.
        
        If you do not pass into a hacking_arg, if the site can be correctly sorted by wyckoff sequence and label, then you are good, if cannot, then a json will be generated, also the dict will be printed to standard output to tell you the hacking_arg. This hacking_arg shall be passed to model.py

    Raises:
        TypeError: if center_atom_label_or_indices is neither list nor list
        ValueError:.?

    Returns:
        none: none
    """

    import json
    from kmcpy.external.pymatgen_structure import Structure
    from kmcpy.external.pymatgen_local_env import CutOffDictNN
    from pymatgen.core.lattice import Lattice

    # Read the initial cif file--------------------
    print('Generating events ...')
    shape = supercell_shape
    
    print('Initializing model with input cif file at',prim_cif_name,'...')

    # this from_cif function is modified so that, it keeps all the label from the cif file
    primitive_cell=Structure.from_cif(prim_cif_name,primitive=convert_to_primitive_cell)
    
    if add_oxidation_state:
        primitive_cell.add_oxidation_state_by_guess()
    
    primitive_cell.remove_species(species_to_be_removed)

    if verbose:
        print_divider()
        print("primitive cell:",primitive_cell)


    # find the indices of center atom, if not assigned explicitly
    if type(center_atom_label_or_indices) is str:
        center_atom_label=center_atom_label_or_indices
        center_atom_indices=[]
        print_divider()
        print("receive string type in 'center atom label' parameter. Trying to find the indices of center atom")        
        for i in range(0,len(primitive_cell)):
            if primitive_cell[i].properties["label"]==center_atom_label_or_indices:
                center_atom_indices.append(i)
                print("please check if this is a center atom:",primitive_cell[i],"fractional_coordinate:",primitive_cell[i].frac_coords,primitive_cell[i].properties)
        print_divider()
        print("all center atom indices:",center_atom_indices)
        
    elif type(center_atom_label_or_indices) is list:
        center_atom_indices=center_atom_label_or_indices
        center_atom_label=primitive_cell[center_atom_label_or_indices[0]].properties["label"]
        
    else:
        raise TypeError("center_atom_label_or_indices is either string or list")
                
    #-------------------------------------------
    # find the reference cluster----------------
    
    local_env_finder = CutOffDictNN(local_env_cutoff_dict)

    # one primitive cell may have more than 1 center atom (for NaSICON, more than 1 Na1). Set the 1st Na1 (center atom) as the reference

    reference_neighbor_sequences=sorted(sorted(local_env_finder.get_nn_info(primitive_cell,center_atom_indices[0]),key=lambda x:x["wyckoff_sequence"]),key = lambda x:x["label"])       

    reference_cluster_sites=[primitive_cell[center_atom_indices[0]]]
    for i in reference_neighbor_sequences:
        reference_cluster_sites.append(i["site"])
    
    # for this reference cluster export the cif structure for observation and validation purpose
    reference_cluster_structure=Structure.from_sites(sites=reference_cluster_sites)
    reference_cluster_structure.to("cif",export_reference_cluster)    
    
    
    def build_distance_matrix_from_getnninfo_output(cutoffdnn_output=reference_neighbor_sequences,verbose=verbose):
        """build a distance matrix from the output of CutOffDictNN.get_nn_info

        nn_info looks like: 
        [{'site': PeriodicSite: Si4+ (-3.2361, -0.3015, 9.2421) [-0.3712, -0.0379, 0.4167], 'image': (-1, -1, 0), 'weight': 3.7390091507903174, 'site_index': 39, 'wyckoff_sequence': 15, 'local_index': 123, 'label': 'Si1'}, {'site': PeriodicSite: Na+ (-1.2831, -2.6519, 9.2421) [-0.3063, -0.3333, 0.4167], 'image': (-1, -1, 0), 'weight': 3.4778161424304046, 'site_index': 23, 'wyckoff_sequence': 17, 'local_index': 35, 'label': 'Na2'}, {'site': ...]
        
        or say:
        
        nn_info is a list, the elements of list is dictionary, the keys of dictionary are: "site":pymatgen.site, "wyckoff_sequence": ....
        
        Use the site.distance function to build matrix
        

        Args:
            cutoffdnn_output (nn_info, optional): nninfo. Defaults to reference_neighbor_sequences.

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
    
    
    reference_distance_matrix=build_distance_matrix_from_getnninfo_output(cutoffdnn_output=reference_neighbor_sequences)

    if verbose:
        np.set_printoptions(precision=2,suppress=True)  
        np.savetxt("reference_distance_matrix.csv",reference_distance_matrix,delimiter=",",newline="\n")
        print_divider()
        print("finding neighbors in primitive cell")
        print("looking for the neighbors of 1st center atom for reference. The 1st center atom is ",primitive_cell[center_atom_indices[0]])
        print("neighbor is arranged in this way")

        print([(neighbor["wyckoff_sequence"],neighbor["label"],neighbor["image"]) for neighbor in reference_neighbor_sequences])
        print_divider()
        print("the reference distance matrix is ")
        print(reference_distance_matrix)
        print("all other center atoms should arrange in this way. If not, then will try to rearrange the sequence of their neighbor")
        print("starting the validation process")
    #-------------------------------------------  
    
    
            

    # try to align all other clusters in the primitive cell to the reference_neighbor_sequences ------------------
    # make sure the distance matrix are the same
    
    local_env_info_dict = {}
    """0: [{'site': PeriodicSite: Si4+ (-3.2361, -0.3015, 9.2421) [-0.3712, -0.0379, 0.4167], 'image': (-1, -1, 0), 'weight': 3.7390091507903174, 'site_index': 39, 'wyckoff_sequence': 15, 'local_index': 123, 'label': 'Si1'}, {'site': PeriodicSite: Na+ (-1.2831, -2.6519, 9.2421) [-0.3063, -0.3333, 0.4167], 'image': (-1, -1, 0), 'weight': 3.4778161424304046, 'site_index': 23, 'wyckoff_sequence': 17, 'local_index': 35, 'label': 'Na2'}, {'site': ...
    """
    
    
    def brutely_rearrange_neighbors(wrong_neighbor_sequence=reference_neighbor_sequences,corect_neighbor_sequence=reference_neighbor_sequences,reference_distance_matrix=reference_distance_matrix):
        """
        reference distance matrix: numpy.2Darray
        wrong neighbor sequence: list of dictionary
        
        [{'site': PeriodicSite: Si4+ (-3.2361, -0.3015, 9.2421) [-0.3712, -0.0379, 0.4167], 'image': (-1, -1, 0), 'weight': 3.7390091507903174, 'site_index': 39, 'wyckoff_sequence': 15, 'local_index': 123, 'label': 'Si1'}, {'site': PeriodicSite: Na+ (-1.2831, -2.6519, 9.2421) [-0.3063, -0.3333, 0.4167], 'image': (-1, -1, 0), 'weight': 3.4778161424304046, 'site_index': 23, 'wyckoff_sequence': 17, 'local_index': 35, 'label': 'Na2'}, {'site': ...}]
        
        for nasicon, wrong neighbor sequence contains 6 neighbors of Si and 6 neighbors of Na
        """
        
        complete_list_for_iteration=[]
        
        dict_by_label={}
        
        for neighbor_information in wrong_neighbor_sequence:# neighbor information: dictionary: {'site': PeriodicSite: Si4+ (-3.2361, -0.3015, 9.2421) [-0.3712, -0.0379, 0.4167], 'image': (-1, -1, 0), 'weight': 3.7390091507903174, 'site_index': 39, 'wyckoff_sequence': 15, 'local_index': 123, 'label': 'Si1'}
            # sort the input neighbor sequence by label
            if neighbor_information["label"] not in dict_by_label:
                dict_by_label[neighbor_information["label"]]=[neighbor_information]
            else:
                dict_by_label[neighbor_information["label"]].extend([neighbor_information])

                
        """
        dict_by_label={"Na2":[{"site":xxx,"index":xxx,...},{"site":xxx,"index":xxx,...},...],
        
                        "Si1:[]...
        }
        
        dict_by_label["label"]=list
        dict_by_label["label"][0]=dic, with key=site,image, wyckoff_sequence,...
        """
        
        for label in dict_by_label:
            """
            complete_list_for_iteration=List[itertools.permutation()]
            every element in the list are permutations of neighbors with 1 kind of label
            
            for nasicon
            
            complete_list_for_iteration=[itertools.permutation($"sites with label Na2"),itertools.permutation($"sites with label Si2")]
            
            for example , if there is 2 Na2 and 3 Si2 
            
            complete_list_for_iteration=[(Na2_1,Na2_2 or Na2_2, Na2_1), (Si1Si2Si3,Si1Si3Si2,Si2Si1Si3,Si2Si3Si1,Si3Si2Si1,Si3Si1Si2)]
            """
            complete_list_for_iteration.append(itertools.permutations(dict_by_label[label]))
        


        print_divider()
        print(" trying to search for a correct neighbor sequence.  progress is stored in logfile. As long as the array is changing, it should be working properly. It take around 90s for searching 12 neighbors. For more neighbors, the searching time increase in O(n!). ")        
        for re_sorted_neighbors_tuple in (itertools.product(*complete_list_for_iteration)):

            """re_sorted_neighbors_tuple is tuple format from itertools.product
            
            need to convert it to list 
            
            it looks like this :
            
            re_sorted_neighbors_tuple=(  (Na2-1,Na2-3,Na2-2) , (Si1-5,Si1-2) ... )

            convert them to plain list
            
            the itertools.product is the product of the permutations 
    
            for example , if there is 2 Na2 and 3 Si2 
            
            complete_list_for_iteration=[(Na2_1,Na2_2 or Na2_2, Na2_1), (Si1Si2Si3,Si1Si3Si2,Si2Si1Si3,Si2Si3Si1,Si3Si2Si1,Si3Si1Si2)]
                        
            for Na2_1 Na2_2 this sequence, try all permutations of Si  (Si1Si2Si3,Si1Si3Si2,Si2Si1Si3,Si2Si3Si1,Si3Si2Si1,Si3Si1Si2)
            then for Na2_2 Na2_1 this sequence ,try all permutations of Si again (Si1Si2Si3,Si1Si3Si2,Si2Si1Si3,Si2Si3Si1,Si3Si2Si1,Si3Si1Si2)
            then if there is any more Na2 permutation , try all Si permutation
            ....
            
            """
            re_sorted_neighbors_list=[]
        
            if verbose:
                with open("searching.log","a+") as f:
                    # this is the progress for impatient people
                    for neighbor in re_sorted_neighbors_tuple:
                        re_sorted_neighbors_list.extend(list(neighbor))
                    sequences=[]
                    for neighbor in re_sorted_neighbors_list:
                        sequences.append(str(neighbor["wyckoff_sequence"]))
                    f.write(",".join(sequences))
                    
            else:
                for neighbor in re_sorted_neighbors_tuple:

                    re_sorted_neighbors_list.extend(list(neighbor))                

            if np.allclose(build_distance_matrix_from_getnninfo_output(cutoffdnn_output=re_sorted_neighbors_list),reference_distance_matrix,rtol=rtol_for_neighbor):
                print_divider()
                print("Reorganized neighbors found. In this sequence of neighbor, the distance matrix should be same to the reference matrix ")
                print([(neighbor["wyckoff_sequence"],neighbor["label"]) for neighbor in re_sorted_neighbors_list])
                print("distance matrix is ")
                print(build_distance_matrix_from_getnninfo_output(cutoffdnn_output=re_sorted_neighbors_list))
                print(" use this as the new sequence of neighbor.")
                return re_sorted_neighbors_list
        raise ValueError("Unfortunately, generate_events2.brutely_rearrange_neighbors() cannot find any way to rearrange the local environment to get a same distance matrix! Probably the tolerance of np.allclose is too small? This shouldn't happen. Please also check if you are working on a correct local cluster....")


    new_hacking_arg={}# brutally rearrange neighbor cost time. Hacking arg = rearranged neighbor sequence of which the distance matrix is the same as reference distance matrix

    for center_atom_index in center_atom_indices:
        
        # try to find the sorted neighbors by wyckoff sequence and label
        # if the distance matrix is the same as the reference matrix (distance matrix of the 1st center atom), then we are good
        # if not the same, we have to modify the sequence of neighbors around this center atom in order to get the same matrix
        # now, using the brutal force to find the new sequence
        """hacking_arg:

            hacking arg is from a already completed calculation where you after hours of computing, finally use the brute force to find the correct sequence of neighbors for each center atom
            
            then will arrange following the hacking_arg
    
        """
        
        if not hacking_arg:
            this_neighbor_sequence=sorted(sorted(local_env_finder.get_nn_info(primitive_cell,center_atom_index),key=lambda x:x["wyckoff_sequence"]),key = lambda x:x["label"])
        else:
            if center_atom_index in hacking_arg:
                if verbose: 
                    print_divider()
                    print("hacking arg triggered")
                this_neighbor_sequence=[]
                for local_index in hacking_arg[center_atom_index]:
                    for unsorted_neighbor in local_env_finder.get_nn_info(primitive_cell,center_atom_index):
                        if unsorted_neighbor["local_index"]==local_index:
                            this_neighbor_sequence.append(unsorted_neighbor)
            else:
                this_neighbor_sequence=sorted(sorted(local_env_finder.get_nn_info(primitive_cell,center_atom_index),key=lambda x:x["wyckoff_sequence"]),key = lambda x:x["label"])
        
        
        
        if np.allclose(build_distance_matrix_from_getnninfo_output(cutoffdnn_output=this_neighbor_sequence),reference_distance_matrix,rtol=rtol_for_neighbor):
            # the distance matrix is correct, means that the neighbors are arranging in  a correct way 
            print_divider()
            print("neighbors of center atom at",primitive_cell[center_atom_index]," is arranging correctly. Carry on next center atom")
            print([(neighbor["wyckoff_sequence"],neighbor["label"],neighbor["image"]) for neighbor in this_neighbor_sequence])
        else: 
            print_divider()
            print("neighbors of ",primitive_cell[center_atom_index]," is not arranging correctly\n the distance matrix is:")      
    
            print(build_distance_matrix_from_getnninfo_output(cutoffdnn_output=this_neighbor_sequence))
            
            print("the difference matrix is :")

            print(build_distance_matrix_from_getnninfo_output(cutoffdnn_output=this_neighbor_sequence)-reference_distance_matrix)
            
            this_neighbor_sequence=brutely_rearrange_neighbors(wrong_neighbor_sequence=this_neighbor_sequence,corect_neighbor_sequence=reference_neighbor_sequences,reference_distance_matrix=reference_distance_matrix)
            
            hacking_arg_for_this_center_atom=[]
            
            for sorted_neighbor in this_neighbor_sequence:
                hacking_arg_for_this_center_atom.append(sorted_neighbor["local_index"])
                
            new_hacking_arg[center_atom_index]=hacking_arg_for_this_center_atom

        
        local_env_info_dict[primitive_cell[center_atom_index].properties["wyckoff_sequence"]]=this_neighbor_sequence

    if len(new_hacking_arg)>0:
        print_divider()
        print("there are different sorting patterns of neighbors observed in this structure. Please record this hacking args and pass it this function for accelerating repeating calculation! iT is also saved to hacking_arg.json")
        print(new_hacking_arg)
        with open("hacking_arg.json",'w') as fhandle:
            jsonStr = json.dumps(new_hacking_arg,indent=4,default=convert)
            fhandle.write(jsonStr)        
  
    if verbose:    
        #print(local_env_info_dict)    
        print_divider()
        print("finish searching of neighbors. Revalidating the distance matrix. Make sure all of the distance matrix are the same!")
        for center_atom_index in center_atom_indices:
            print("finding local environment of",primitive_cell[center_atom_index],"the local info is ",local_env_info_dict[primitive_cell[center_atom_index].properties["wyckoff_sequence"]])

            print("different matrix: (should be 0)\n",build_distance_matrix_from_getnninfo_output(local_env_info_dict[primitive_cell[center_atom_index].properties["wyckoff_sequence"]])-reference_distance_matrix)
                     
    
    # finish trying to align all other clusters in the primitive cell to the reference_neighbor_sequences ------------------  
    #----------------------------------------------
    

    #----------------------------------------------   
    # create the supercell
    
    
    supercell=primitive_cell.make_kmc_supercell(supercell_shape)
    # this special make_kmc_supercell will keep the informations as in site_properties``
    if verbose:
        print_divider()
        print("build the supercell:\n",supercell)

    
    
    # find the center atom indices of supercell
    # each primitive cell has 2 Na1 center atoms. For supercell, each "image" has 2 center atoms
    supercell_center_atom_indices=[]
    for i in range(0,len(supercell)):
        if supercell[i].properties["label"]==center_atom_label:
            supercell_center_atom_indices.append(i)    
            
    events = []
    events_dict = []
    
    #----------------------------------------------   
    # map the neighbors to the supercell    
    def _equivalent_position_in_periodic_supercell(site_belongs_to_supercell=[5,1,7],image_of_site=(0,-1,1),supercell_shape=[5,6,7],additional_input=False,verbose=False):
        """finding the equivalent position in periodic supercell considering the periodic boundary condition
        input:
        site_belongs_to_supercell: site belongs to which supercell

        Returns:
            _type_: _description_
        """
        if verbose:
            print ("equivalent position",site_belongs_to_supercell,image_of_site)
        # 5 1 7 with image 0 -1 1 -> 5 0 8 -> in periodic 567 supercell should change to 561, suppose supercell start with index1
        
        temp=np.array(site_belongs_to_supercell)+np.array(image_of_site)
        # 517+(0-11)=508
        
        
        # 508-1=4-17 mod: 4 5 0 
        #+1 : 561
        temp=np.mod(temp,supercell_shape)
        
        temp=temp.tolist()
        if additional_input is not False:
            temp.append(additional_input)
        return tuple(temp)    
    

    indices_dict_from_identifier=supercell.kmc_build_dict(skip_check=False)#a dictionary. Key is the tuple with same format as class.kmc_info_to_tuple, Value is the global indices
    
    for supercell_center_atom_index in supercell_center_atom_indices:
        
        # for center atoms of newly generated supercell, find the neighbors
        
        
        this_center_atom_belongs_to_supercell=supercell[supercell_center_atom_index].properties["supercell"]
        wyckoff_sequence_of_this_center_atom=supercell[supercell_center_atom_index].properties["wyckoff_sequence"]
        local_env_info=[]# list of integer / indices of local environment
        
        
        for neighbor_site_in_primitive_cell in local_env_info_dict[wyckoff_sequence_of_this_center_atom]:
            """
            
             local_env_info_dict[wyckoff_sequence_of_this_center_atom]
            
            In primitive cell, the center atom has 1 unique identifier: The "wyckoff sequence inside the primitive cell"
            
            IN supercell, the center atom has an additional unique identifier: belongs to which supercell
            
            However, as long as the "wyckoff sequence inside the primitive cell" is the same, no matter which supercell this center atom belongs to, the sequence of "wyckoff sequence" of its neighbor sites are the same. In primitive cell, center atom with index 1 has neighbor arranged in 1,3,2,4,6,5, then for every center atom with index 1 in supercell, the neighbor is arranged in 1,3,2,4,6,5
            
            Mapping the sequence in primitive cell to supercell!
            
            
            In order to accelerate the speed

            use the dictionary to store the index of atoms
            
            indices_dict_from_identifier is a dictionary by pymatgen_structure.kmc_build_dict()
            
            the key to dictionary is kmc_info_to_tuple
            
            This loop build the local_env_info list, [supercell_neighbor_index1, supercell_neighbor_index2, .....]
            
            """
            
            local_env_info.append(indices_dict_from_identifier[supercell.kmc_info_to_tuple(wyckoff_sequence=neighbor_site_in_primitive_cell["wyckoff_sequence"],label=neighbor_site_in_primitive_cell["label"],supercell=_equivalent_position_in_periodic_supercell(site_belongs_to_supercell=this_center_atom_belongs_to_supercell,image_of_site=neighbor_site_in_primitive_cell["image"],supercell_shape=supercell_shape))])

        
        if verbose:
            """
            In this case , will do the validation of distance matrix
            
            Should greatly decrease the speed
            
            """
            # center atom distance matrix
            # this is distance matrix between center atom and neighbor atom

            print_divider()
            center_distance_matrix=np.array([])
            for local_env_index in local_env_info:
                if verbose==2:
                    print("distance from center to environment:",supercell[supercell_center_atom_index].properties,supercell[local_env_index].properties,supercell[supercell_center_atom_index].distance(supercell[local_env_index]))#debug
                center_distance_matrix=np.append(center_distance_matrix,supercell[supercell_center_atom_index].distance(supercell[local_env_index]))
            print("center atom distance matrix: ",center_distance_matrix)
            
            # distance matrix of supercell neighbors
            print_divider()
            
            
            #print(build_distance_matrix_from_getnninfo_output(local_env_info))

            print("finding local environment of",supercell[supercell_center_atom_index],"the local info is ",[(supercell[global_index].properties["label"],supercell[global_index].properties["wyckoff_sequence"],supercell[global_index].properties["supercell"]) for global_index in local_env_info])
            
            unsorted_supercell_neighbors=local_env_finder.get_nn_info(supercell,supercell_center_atom_index)
            
            print("unsorted neighbors of this center atom by cutoffdnn is ",unsorted_supercell_neighbors)
            
            sorted_supercell_neighbors=[]
            
            for supercell_neighbor_index in local_env_info:
                
                # add the neighbor to sorted_supercell_neighbor in sequence
                
                for unsorted_neighbor in unsorted_supercell_neighbors:
                    
                    # problem here!!!
                    
                    if supercell[supercell_neighbor_index].properties["label"] == unsorted_neighbor["label"] and supercell[supercell_neighbor_index].properties["wyckoff_sequence"] == unsorted_neighbor["wyckoff_sequence"] and supercell[supercell_neighbor_index].properties["supercell"] == unsorted_neighbor["supercell"]:
                        sorted_supercell_neighbors.append(unsorted_neighbor)
                        break
            

            this_supercell_distance_matrix=build_distance_matrix_from_getnninfo_output(cutoffdnn_output=sorted_supercell_neighbors)

            print("rearrange it to",sorted_supercell_neighbors, "the distance matrix is ",this_supercell_distance_matrix)

            if not np.allclose(this_supercell_distance_matrix,reference_distance_matrix,rtol=rtol_for_neighbor):
                print("warning: The supercell distance matrix is different from reference distance matrix by:", this_supercell_distance_matrix-reference_distance_matrix)
                raise ValueError("Supercell distance matrix is different from reference distance matrix")
            else:
                
                print("this supercell distance matrix is validated to be same with reference distance matrix within tolerance of ", rtol_for_neighbor)
            

        for local_env in local_env_info:
            # generate event
            if supercell[local_env].properties["label"] == diffuse_to_atom_label:
                # or for understanding, if any site in local environment, its label== "Na2"
                # initialize the event
                this_event = Event()
                this_event.initialization2(supercell_center_atom_index,local_env,local_env_info)
                events.append(this_event)
                events_dict.append(this_event.as_dict())            
    
    if len(events)==0:
        raise ValueError("There is no events generated. This is probably caused by wrong input parameters. Probably check the diffuse_to_atom_label?")
            


    print('Saving:',event_fname)
    with open(event_fname,'w') as fhandle:
        jsonStr = json.dumps(events_dict,indent=4,default=convert)
        fhandle.write(jsonStr)
    
    events_site_list = []
    for event in events:
        # sublattice indices: local site index for each site
        events_site_list.append(event.sorted_sublattice_indices)
    
    np.savetxt('./events_site_list.txt',np.array(events_site_list,dtype=int),fmt="%i")
    generate_event_kernal(len(supercell),np.array(events_site_list),event_kernal_fname=event_kernal_fname)
    pass
    

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
        
def generate_events3(prim_cif_name="210.cif",convert_to_primitive_cell=False,local_env_cutoff_dict={("Li+","Cl-"):4.0,("Li+","Li+"):3.0},atom_identifier_type="specie",center_atom_identifier="Li+",diffuse_to_atom_identifier="Li+",species_to_be_removed=["O2-","O"],distance_matrix_rtol=0.01,distance_matrix_atol=0.01,find_nearest_if_fail=True,export_local_env_structure=True,supercell_shape=[2,1,1],event_fname="events.json",event_kernal_fname='event_kernal.csv',verbosity="INFO"):
    """
    220603 XIE WEIHANG    
    3rd version of generate events, using the x coordinate and label as the default sorting criteria for neighbors in local environment therefore should behave similar as generate_events1. Comparing generate_events1, this implementation accelerate the speed of finding neighbors and add the capability of looking for various kind of center atoms (not only Na1 in generate_events1). In addtion, generate events3 is also capable of identifying various kind of local environment, which can be used in grain boundary models. Although the _generate_event_kernal is not yet capable of identifying different types of environment. The speed is improved a lot comparing with version2 

    Args:
        prim_cif_name (str, optional): the file name of primitive cell of KMC model. Strictly limited to cif file because only cif parser is capable of taking label information of site. Defaults to "210.cif".
        convert_to_primitive_cell (bool, optional): whether convert to primitive cell. For rhombohedral, if convert_to_primitive_cell, will use the rhombohedral primitive cell, otherwise use the hexagonal primitive cell. Defaults to False.
        local_env_cutoff_dict (dict, optional): cutoff dictionary for finding the local environment. This will be passed to local_env.cutoffdictNN`. Defaults to {("Li+","Cl-"):4.0,("Li+","Li+"):3.0}.
        atom_identifier_type (str, optional): atom identifier type, choose from ["specie", "label"].. Defaults to "specie".
        center_atom_identifier (str, optional): identifier for center atom. Defaults to "Li+".
        diffuse_to_atom_identifier (str, optional): identifier for the atom that center atom will diffuse to . Defaults to "Li+".
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
    
    # build the logger
    event_generator_logger=logging.getLogger("event generator")
    event_generator_logger.setLevel(verbosity)
    event_generator_logger.addHandler(logging.StreamHandler())
    event_generator_logger.addHandler(logging.FileHandler("debug.log"))
    event_generator_logger.warning("Extracting clusters from primitive cell structure. This primitive cell should be bulk structure, grain boundary model not implemented yet.")
    
    
    # generate primitive cell
    primitive_cell=Structure.from_cif(prim_cif_name,primitive=convert_to_primitive_cell)
    primitive_cell.add_oxidation_state_by_guess()
    primitive_cell.remove_species(species_to_be_removed)
    
    event_generator_logger.warning("primitive cell composition after adding oxidation state and removing uninvolved species: ")
    event_generator_logger.info(str(primitive_cell.composition))
    event_generator_logger.warning("building center atom index list")
    

    center_atom_indices=find_atom_indices(primitive_cell,atom_identifier_type=atom_identifier_type,atom_identifier=center_atom_identifier)  
        
    #--------
    
    local_env_finder = CutOffDictNN(local_env_cutoff_dict)
    
    reference_local_env_dict={}
    
    local_env_info_dict = {}
   
   
    reference_local_env_type=0
    
    event_generator_logger.info("start finding the neighboring sequence of center atoms")
    event_generator_logger.info("total number of center atoms:"+str(len(center_atom_indices)))
    
    neighbor_has_been_found=0
    
    for center_atom_index in center_atom_indices:
        
        unsorted_neighbor_sequences=sorted(sorted(local_env_finder.get_nn_info(primitive_cell,center_atom_index),key=lambda x:x["site"].coords[0]),key = lambda x:x["site"].specie)      
                    
        this_nninfo=neighbor_info_matcher.from_neighbor_sequences(unsorted_neighbor_sequences)
        
        if this_nninfo.neighbor_species not in reference_local_env_dict:
            
            # then take this as the reference neighbor info sequence
            
            if export_local_env_structure:
                
                reference_local_env_sites=[primitive_cell[center_atom_index]]
                
                for i in unsorted_neighbor_sequences:
                    
                    reference_local_env_sites.append(i["site"])
                    reference_local_env_structure=Structure.from_sites(sites=reference_local_env_sites)
                    
                reference_local_env_structure.to("cif",str(reference_local_env_type)+"th_reference_local_env.cif") 
                reference_local_env_type+=1
                
                event_generator_logger.info(str(reference_local_env_type)+"th type of reference local_env structure cif file is created. please check")
            
            reference_local_env_dict[this_nninfo.neighbor_species]=this_nninfo

            local_env_info_dict[primitive_cell[center_atom_index].properties['local_index']]=this_nninfo.neighbor_sequence
            
            
            event_generator_logger.warning("a new type of cluster is recognized with the species "+str(this_nninfo.neighbor_species)+" \nthe distance matrix are \n"+str(this_nninfo.distance_matrix))

            
        else:
            event_generator_logger.info("a cluster is created with the species "+str(this_nninfo.neighbor_species)+" \nthe distance matrix are \n"+str(this_nninfo.distance_matrix))
            
            

            sorted_neighbor_sequence=reference_local_env_dict[this_nninfo.neighbor_species].brutal_match(this_nninfo.neighbor_sequence,rtol=distance_matrix_rtol,atol=distance_matrix_atol,find_nearest_if_fail=find_nearest_if_fail)
            
            local_env_info_dict[primitive_cell[center_atom_index].properties['local_index']]=sorted_neighbor_sequence
            
        neighbor_has_been_found+=1
        
        event_generator_logger.warning(str(neighbor_has_been_found)+" out of "+str(len(center_atom_indices))+" neighboring sequence has been found")
        

  
    supercell=primitive_cell.make_kmc_supercell(supercell_shape)
    event_generator_logger.warning("supercell is created")
    event_generator_logger.info(str(supercell))
    
    supercell_center_atom_indices=find_atom_indices(supercell,atom_identifier_type=atom_identifier_type,atom_identifier=center_atom_identifier)

            
    events = []
    events_dict = []
    
    def _equivalent_position_in_periodic_supercell(site_belongs_to_supercell=[5,1,7],image_of_site=(0,-1,1),supercell_shape=[5,6,7],additional_input=False,verbose=False):
        """finding the equivalent position in periodic supercell considering the periodic boundary condition
        input:
        site_belongs_to_supercell: site belongs to which supercell

        Returns:
            _type_: _description_
        """
        if verbose:
            print ("equivalent position",site_belongs_to_supercell,image_of_site)
        # 5 1 7 with image 0 -1 1 -> 5 0 8 -> in periodic 567 supercell should change to 561, suppose supercell start with index1
        
        temp=np.array(site_belongs_to_supercell)+np.array(image_of_site)
        # 517+(0-11)=508
        
        
        # 508-1=4-17 mod: 4 5 0 
        #+1 : 561
        temp=np.mod(temp,supercell_shape)
        
        temp=temp.tolist()
        if additional_input is not False:
            temp.append(additional_input)
        return tuple(temp)    
    
    
    indices_dict_from_identifier=supercell.kmc_build_dict3(skip_check=False)#a dictionary. Key is the tuple with same format as class.kmc_info_to_tuple, Value is the global indices   

    
    for supercell_center_atom_index in supercell_center_atom_indices:
        
        # for center atoms of newly generated supercell, find the neighbors

        this_center_atom_belongs_to_supercell=supercell[supercell_center_atom_index].properties["supercell"]
        
        local_index_of_this_center_atom=supercell[supercell_center_atom_index].properties["local_index"]
        
        local_env_info=[]# list of integer / indices of local environment
        
        
        for neighbor_site_in_primitive_cell in local_env_info_dict[local_index_of_this_center_atom]:
            """
            
             local_env_info_dict[local_index_of_this_center_atom]
            
            In primitive cell, the center atom has 1 unique identifier: The "local index inside the primitive cell"
            
            IN supercell, the center atom has an additional unique identifier: belongs t o which supercell
            
            However, as long as the "local index inside the primitive cell" is the same, no matter which supercell this center atom belongs to, the sequence of "local index" of its neighbor sites are the same. In primitive cell, center atom with index 1 has neighbor arranged in 1,3,2,4,6,5, then for every center atom with index 1 in supercell, the neighbor is arranged in 1,3,2,4,6,5
            
            In the loop. I'm mapping the sequence in primitive cell to supercell!
            
            In order to accelerate the speed

            use the dictionary to store the index of atoms
            
            indices_dict_from_identifier is a dictionary by pymatgen_structure.kmc_build_dict()
            
            the key to dictionary is kmc_info_to_tuple
            
            This loop build the local_env_info list, [supercell_neighbor_index1, supercell_neighbor_index2, .....]
            
            """
            
            neighbor_site_around_supercell_center_atom_belongs_to_supercell=_equivalent_position_in_periodic_supercell(site_belongs_to_supercell=this_center_atom_belongs_to_supercell,image_of_site=neighbor_site_in_primitive_cell["image"],supercell_shape=supercell_shape)
            
            tuple_key_of_such_neighbor_site=supercell.kmc_info_to_tuple3(local_index=neighbor_site_in_primitive_cell["local_index"],label=neighbor_site_in_primitive_cell["label"],supercell=neighbor_site_around_supercell_center_atom_belongs_to_supercell)
            
            local_env_info.append(indices_dict_from_identifier[tuple_key_of_such_neighbor_site])

        for local_env in local_env_info:
            # generate event
            
            if atom_identifier_type=="specie":
                if diffuse_to_atom_identifier in supercell[local_env].species  :
                    # initialize the event
                    this_event = Event()
                    this_event.initialization2(supercell_center_atom_index,local_env,local_env_info)
                    events.append(this_event)
                    events_dict.append(this_event.as_dict())   
                        
            elif atom_identifier_type=="label":

                if supercell[local_env].properties["label"] == diffuse_to_atom_identifier:
                    # or for understanding, if any site in local environment, its label== "Na2"
                    # initialize the event
                    this_event = Event()
                    this_event.initialization2(supercell_center_atom_index,local_env,local_env_info)
                    events.append(this_event)
                    events_dict.append(this_event.as_dict())   
                        
            elif atom_identifier_type=="list":
                raise NotImplementedError("how to do this.... atom identifier_type=list is not implemented in finding neighbors") 
                                   
            else:
                raise ValueError('unrecognized atom_identifier_type. Please select from: ["specie","label","list"] ')
            
    if len(events)==0:
        raise ValueError("There is no events generated. This is probably caused by wrong input parameters. Probably check the diffuse_to_atom_label?")
    
    print('Saving:',event_fname)
    with open(event_fname,'w') as fhandle:
        jsonStr = json.dumps(events_dict,indent=4,default=convert)
        fhandle.write(jsonStr)
    
    events_site_list = []

    for event in events:
        # sublattice indices: local site index for each site
        events_site_list.append(event.sorted_sublattice_indices)
    
    #np.savetxt('./events_site_list.txt',np.array(events_site_list,dtype=int),fmt="%i") # dimension not equal error
    generate_event_kernal(len(supercell),np.array(events_site_list),event_kernal_fname=event_kernal_fname)       
    
def generate_events1(prim_fname="prim.json",supercell_shape=[2,1,1],event_fname="events.json",event_kernal_fname='event_kernal.csv'):
    """generate_events() looks for all possible swaps by given a primitive cell as defined in prim_fname(prim.json) with a supercell shape of [2,1,1] as default.

    Args:
        prim_fname (_type_): _description_
        supercell_shape (_type_): _description_
        event_fname (_type_): _description_
    """
    import json
    from kmcpy.external.pymatgen_structure import Structure
    from pymatgen.analysis.local_env import CutOffDictNN
    from pymatgen.core.lattice import Lattice
    import multiprocessing
    from joblib import Parallel, delayed
    
    print('Generating events ...')
    shape = supercell_shape
    
    print('Initializing model with pirm.json at',prim_fname,'...')
    with open(prim_fname,'r') as f:
        prim = json.load(f)
        prim_coords = [site['coordinate'] for site in prim['basis']]
        prim_specie_cases = [site['occupant_dof'] for site in prim['basis']]
        prim_lattice = Lattice(prim['lattice_vectors'])
        prim_species = [s[0] for s in prim_specie_cases]# need to change: now from [Na,Li] it will choose Na
        
    supercell_shape_matrix = np.diag(supercell_shape)
    print('Supercell Shape:\n',supercell_shape_matrix)
    structure = Structure(prim_lattice,prim_species,prim_coords)
    print('Converting supercell ...')
    structure.remove_species(['Zr','O'])
    structure.make_supercell(supercell_shape_matrix)
    structure.to(fmt='cif',filename='supercell.cif')
    # np.savetxt('frac_coords.txt',np.array(structure.frac_coords))
    # np.savetxt('cart_coords.txt',np.array(structure.cart_coords))
    # print(structure.lattice.matrix)
    # np.savetxt('latt.txt',np.array(structure.lattice.matrix))
    
    # change the parameter name to center_site (center_atom). 
    # this information  can be loaded from input (future feature)
    # currently, the sodium1 is the first two sites in prim.json
    # need to generalize to arbitrary sequence
    # future code: the input shall be a list, telling the index of center atom / Na1
    # current value 2, equivalent [0,1] (len([0,1])=2)
    #  future input [0,1] or any indices
    
    n_na1_per_prim = 2# number of Na1 per primitive cell
    
    
    n_na1 = n_na1_per_prim*np.prod(shape) # total number of Na1 in the supercell. This should be equal to the empty lines in the event_kernal.csv
    
    
    local_env_cutoff_dict = {('Na','Na'):4,('Na','Si'):4}
    ncpus = multiprocessing.cpu_count()
    print('Looking for all possible swaps with ncpus =',ncpus,' (this might take a while) ...')
    na_1_sites = structure[0:n_na1]
    local_env_finder = CutOffDictNN(local_env_cutoff_dict)
    
    # get_nn_info; return the neiarest neighbor (or local environment)
    local_env_info_list = list(Parallel(n_jobs=ncpus)(delayed(local_env_finder.get_nn_info)(structure,i) for i in np.arange(0,n_na1)))
    # change to scipy.spatial.cKDTtree in the future
    # get_nn_info return list of these thing:
    # local_env_info_list[0]: local environmental information of 0th Na1
    # local_env_info_list[0] includes (or get_nn_info return) list of neighbors, with each neighbor as dictionary of: {site, image(periodic structure index),weight,site_index}     
    
    events = []
    events_dict = []
    for (na1_index,(na1_site,local_env_info)) in enumerate(zip(na_1_sites,local_env_info_list)):
        for local_env in local_env_info:
            # only consider the environment of Na, but not S or P.
            # need generalization for other system.
            # "Na" -> specie_to_diffuse
            # need to be input.
            # probably become a list.
            if "Na" in local_env['site'].specie.symbol:
                this_event = Event()
                this_event.initialization(na1_index,local_env['site_index'],local_env_info)
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
    generate_event_kernal(len(structure),np.array(events_site_list),event_kernal_fname=event_kernal_fname)

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
        is_Na1=False
        event_index = 0
        for event in events_site_list:
            if site in event:
                row.append(event_index)
            event_index+=1
            if len(row)==0:
                is_Na1=True
        results.append(row)
    return results


def _generate_event_kernal3(len_structure,events_site_list):
    """to be called by generate_event_kernal3 for generating the event_kernal.csv
    
    This is a very slow version that do not use the numba
    
    NEED FURTHER ALGO IMPROVEMENT FOR LARGER SUPERCELL
    
    for  a event and find all other event that include the site of this event
    

    Args:
        len_structure (int): _description_
        events_site_list (list): list[ list[]  ]

    Returns:
        list[ list[]  ]: _description_
    """
    n_sites  = len_structure
    all_site_list = np.arange(n_sites)
    results = []
    for site in all_site_list:
        # print('Looking for site:',site)
        row = []
        #is_Na1=False
        event_index = 0
        for event in events_site_list:
            if site in event:
                row.append(event_index)
            event_index+=1
            #if len(row)==0:
                #is_Na1=True
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

def generate_event_kernal3(len_structure,events_site_list,event_kernal_fname='event_kernal.csv'):
    """
    event_kernal.csv: 
        event_kernal[i] tabulates the index of sites that have to be updated after event[i] has been executed
        
        
    
    """
    print('Generating event kernal ...')
    event_kernal = _generate_event_kernal3(len_structure,events_site_list)
    with open(event_kernal_fname, 'w') as f:
        print('Saving into:',event_kernal_fname)
        for row in event_kernal:
            for item in row:
                f.write('%5d ' % item)
            f.write('\n')
    return event_kernal

# def generate_event_kernal(len_structure):
#     import subprocess, os
#     print('Computing site_event_list  ...')
#     home = os.getcwd()
#     subprocess.run(["rm","-f","generate_list.x"])
#     subprocess.run(["g++","-std=c++11","generate_list.cpp","-o","generate_list.x"])
#     subprocess.run(["./generate_list.x",str(len_structure)])
#     os.rename("results.csv",home+"/"+"event_kernal.csv")
#     os.chdir(home)

if __name__=="__main__":
    import os
    os.chdir("examples/inputs")
    generate_events2(prim_fname="prim.json",supercell_shape=[2,1,1],event_fname="events.json",event_kernal_fname='event_kernal.csv')