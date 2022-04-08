#!/usr/bin/env python

from kmcpy.event import Event
import numpy as np
from numba.typed import List
import numba as nb
from kmcpy.io import convert


def print_divider():
    print("\n\n-------------------------------------------\n\n")

def generate_events(api=1,**kwargs):
    
    if api==1:
        return generate_events1(**kwargs)
    elif api>=2:
        return generate_events2(**kwargs)
    else:
        raise NotImplementedError("debug information from event_generator.generate_events. Unsupport API value: api=",api)




def generate_events2(prim_cif_name="EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",supercell_shape=[2,1,1],local_env_cutoff_dict = {('Na+','Na+'):4,('Na+','Si4+'):4},event_fname="events.json",event_kernal_fname='event_kernal.csv',center_atom_label_or_indices="Na1",diffuse_to_atom_label="Na2",species_to_be_removed=['Zr4+','O2-','O','Zr'],verbose=False,hacking_arg={1:[27,29,28,30,32,31,117,119,118,120,122,121],2:[21,22,23,32,30,31,111,112,113,122,120,121],3:[18,20,19,34,33,35,108,110,109,124,123,125],5:[21,23,22,24,26,25,111,113,112,114,116,115]}):
    """2nd version of generate_events2
    
    

    Args:
        prim_cif_name (str, optional): Path to the cif file. ONLY CIF FILE IS ACCEPTED! REMEMBER TO CHANGE THE label of different sites. Defaults to "EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif".
        supercell_shape (list, optional): list of supercell shape with len=3. Defaults to [2,1,1].
        
        local_env_cutoff_dict (dict, optional): cutoff dict for finding neighbors. Must include the charge otherwise cannot find. Defaults to {('Na+','Na+'):4,('Na+','Si4+'):4}.
        
        event_fname (str, optional): output filename of event.json. Defaults to "events.json".
        
        event_kernal_fname (str, optional): output csv file of event kernel, this is : which events need to be updated if event with index=column is updated. Defaults to 'event_kernal.csv'.
        
        center_atom_label_or_indices (str / list , optional): string or list of integers . If passing a string, the program will find the indices of all sites in the cif file which has the label=string. if passing a list, then will treat it as the list of indices of center atom. (center of local cluster). Defaults to "Na1".
        diffuse_to_atom_label (str, optional): diffuse to what atom? . Defaults to "Na2".
        species_to_be_removed (list, optional): the species that do not participate in the local cluster expansion calculation. Defaults to ['Zr4+','O2-','O','Zr'].
        verbose (bool or int, optional): if False, then only limited output. If true, then standard verbose output. If verbose=2, a lot of annoying output. Defaults to False.
        
        hacking_arg (dict, optional): dictionary of hacking arg. The key is the index of center atom, the value is a list, the element of the list is the "local_index" of neighbors. If hacking_ar is present, then when looking for the neighbors and trying to sort them, if the center_atom_index is in the hacking_arg_dict, then the neighbor will be arranged by the sequence of hacking_arg[center_atom_index] The example here is for the NASICON. Defaults to {1:[27,29,28,30,32,31,117,119,118,120,122,121],2:[21,22,23,32,30,31,111,112,113,122,120,121],3:[18,20,19,34,33,35,108,110,109,124,123,125],5:[21,23,22,24,26,25,111,113,112,114,116,115]}.
        
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

    
    print('Generating events ...')
    shape = supercell_shape
    
    print('Initializing model with input cif file at',prim_cif_name,'...')

    # I only modify the from_cif!
    primitive_cell=Structure.from_cif(prim_cif_name)
    primitive_cell.remove_species(species_to_be_removed)
    
    
    # find the indices of center atom, if not assigned explicitly
    if type(center_atom_label_or_indices) is str:
        center_atom_label=center_atom_label_or_indices
        center_atom_indices=[]
        print_divider()
        print("receive string type in 'center atom label' parameter. Trying to find the indices of center atom")        
        for i in range(0,len(primitive_cell)):
            if primitive_cell[i].properties["label"]==center_atom_label_or_indices:
                center_atom_indices.append(i)
                print("please check if this is a center atom:",primitive_cell[i],primitive_cell[i].properties)
        print_divider()
        print("all center atom indices:",center_atom_indices)
        
    elif type(center_atom_label_or_indices) is list:
        center_atom_indices=center_atom_label_or_indices
        center_atom_label=primitive_cell[center_atom_label_or_indices[0]].properties["label"]
        
    else:
        raise TypeError("center_atom_label_or_indices is either string or list")
                
                


    #-------------------------------------------
    # find nearest neighbor in primitive cell    
    local_env_finder = CutOffDictNN(local_env_cutoff_dict)

    # Due to the problem that the wyckoff sequence also fail in identifying the structures
    
    # I set this reference_neighbor_sequences, set the sequence of 1st center atom index as the reference
    
    reference_neighbor_sequences=sorted(sorted(local_env_finder.get_nn_info(primitive_cell,center_atom_indices[0]),key=lambda x:x["wyckoff_sequence"]),key = lambda x:x["label"])
    
    
    def build_distance_matrix_from_getnninfo_output(cutoffdnn_output=reference_neighbor_sequences):
        """build a distance matrix from the output of CutOffDictNN.get_nn_info

        nn_info looks like: 
        [{'site': PeriodicSite: Si4+ (-3.2361, -0.3015, 9.2421) [-0.3712, -0.0379, 0.4167], 'image': (-1, -1, 0), 'weight': 3.7390091507903174, 'site_index': 39, 'wyckoff_sequence': 15, 'local_index': 123, 'label': 'Si1'}, {'site': PeriodicSite: Na+ (-1.2831, -2.6519, 9.2421) [-0.3063, -0.3333, 0.4167], 'image': (-1, -1, 0), 'weight': 3.4778161424304046, 'site_index': 23, 'wyckoff_sequence': 17, 'local_index': 35, 'label': 'Na2'}, {'site': ...]
        
        or say:
        
        nn_info is a list, the elements of list is dictionary, the keys of dictionary are: "site":pymatgen.site, "wyckoff_sequence": ....
        
        Use the site.distance function to build matrix
        

        Args:
            cutoffdnn_output (nn_info, optional): nninfo. Defaults to reference_neighbor_sequences.

        Returns:
            np.2darray: 2d distance matrix, in format of numpy.array
        """
    
        distance_matrix=np.zeros(shape=(len(cutoffdnn_output),len(cutoffdnn_output)))
        
        for sitedictindex1 in range(0,len(cutoffdnn_output)):
            for sitedictindex2 in range(0,len(cutoffdnn_output)):
                # get distance between two sites. Build the distance matrix
                distance_matrix[sitedictindex1][sitedictindex2]=cutoffdnn_output[sitedictindex1]["site"].distance(cutoffdnn_output[sitedictindex2]["site"])
        
        return distance_matrix
    
    
    reference_distance_matrix=build_distance_matrix_from_getnninfo_output(cutoffdnn_output=reference_neighbor_sequences)
            
    if verbose:
        print_divider()
        print("finding neighbors in primitive cell")
        print("looking for the neighbors of 1st center atom for reference. The 1st center atom is ",primitive_cell[center_atom_indices[0]])
        print("neighbor is arranged in this way")

        print(reference_neighbor_sequences)
        print_divider()
        print("the reference distance matrix is ")
        print(reference_distance_matrix)
        print("all other center atoms should arrange in this way. If not, then will try to rearrange the sequence of their neighbor")
        print("starting the validation process")
        

    # try to align all other local structure to the reference_neighbor_sequences
    
    local_env_info_dict = {}
    """0: [{'site': PeriodicSite: Si4+ (-3.2361, -0.3015, 9.2421) [-0.3712, -0.0379, 0.4167], 'image': (-1, -1, 0), 'weight': 3.7390091507903174, 'site_index': 39, 'wyckoff_sequence': 15, 'local_index': 123, 'label': 'Si1'}, {'site': PeriodicSite: Na+ (-1.2831, -2.6519, 9.2421) [-0.3063, -0.3333, 0.4167], 'image': (-1, -1, 0), 'weight': 3.4778161424304046, 'site_index': 23, 'wyckoff_sequence': 17, 'local_index': 35, 'label': 'Na2'}, {'site': ...

    
    """
    
    
    def brutely_rearrange_neighbors(wrong_neighbor_sequence=reference_neighbor_sequences,corect_neighbor_sequence=reference_neighbor_sequences,reference_distance_matrix=reference_distance_matrix):
        import itertools
        
        complete_list_for_iteration=[]
        
        dict_by_label={}
        
        for neighbor_information in wrong_neighbor_sequence:
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
            complete_list_for_iteration.append(itertools.permutations(dict_by_label[label]))


        print_divider()
        print(" trying to search for a correct neighbor sequence.  progress is stored in logfile. As long as the array is changing, it should be working properly. Be patient.")        
        for re_sorted_neighbors_tuple in (itertools.product(*complete_list_for_iteration)):

            """re_sorted_neighbors_tuple is tuple format from itertools.product
            
            need to convert it to list 
            
            it looks like this :
            
            re_sorted_neighbors_tuple=(  (Na2-1,Na2-3,Na2-2) , (Si1-5,Si1-2) ... )

            convert them to plain list
            """
            re_sorted_neighbors_list=[]
            
            
            if verbose:
                with open("searching.log","a+") as f:


                    for neighbor in re_sorted_neighbors_tuple:
                        re_sorted_neighbors_list.extend(list(neighbor))
                    sequences=[]
                    for neighbor in re_sorted_neighbors_list:
                        sequences.append(str(neighbor["wyckoff_sequence"]))
                    f.write(",".join(sequences))
                    
            else:
                for neighbor in re_sorted_neighbors_tuple:

                    re_sorted_neighbors_list.extend(list(neighbor))                

            

            if np.allclose(build_distance_matrix_from_getnninfo_output(cutoffdnn_output=re_sorted_neighbors_list),reference_distance_matrix,rtol=0.001):
                print_divider()
                print("Reorganized neighbors found. In this sequence of neighbor, the distance matrix should be same to the reference matrix ")
                print(re_sorted_neighbors_list)
                print("distance matrix is ")
                print(build_distance_matrix_from_getnninfo_output(cutoffdnn_output=re_sorted_neighbors_list))
                print(" use this as the new sequence of neighbor.")
                return re_sorted_neighbors_list
        raise ValueError("Unfortunately, generate_events2.brutely_rearrange_neighbors() cannot find any way to rearrange the local environment to get a same distance matrix! Probably the tolerance of np.allclose is too small? This shouldn't happen. Please also check if you are working on a correct local cluster....")


    new_hacking_arg={}

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
        
        
        
        if np.allclose(build_distance_matrix_from_getnninfo_output(cutoffdnn_output=this_neighbor_sequence),reference_distance_matrix,rtol=0.001):
            # the distance matrix is correct, means that the neighbors are arranging in  a correct way 
            print_divider()
            print("neighbors of center atom at",primitive_cell[center_atom_index]," is arranging correctly. Carry on next center atom")
        else: 
            print_divider()
            print("neighbors of ",primitive_cell[center_atom_index]," is not arranging correctly\n the distance matrix is:")      
            np.set_printoptions(precision=3)      
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
            
            center_distance_matrix=np.array([])
            
            for neighbor_information in local_env_info_dict[primitive_cell[center_atom_index].properties["wyckoff_sequence"]]:
                for neighbor_information2 in local_env_info_dict[primitive_cell[center_atom_index].properties["wyckoff_sequence"]]: 
                    center_distance_matrix=np.append(center_distance_matrix,neighbor_information["site"].distance(neighbor_information2["site"]))
            print("center atom distance matrix: ",center_distance_matrix)    
    
    
    
    #----------------------------------------------
    
    
    
    # create the supercell
    supercell=primitive_cell.make_kmc_supercell(supercell_shape)
    if verbose:
        print_divider()
        print(supercell)

    
    
    # find the center atom indices of supercell
    supercell_center_atom_indices=[]
    for i in range(0,len(supercell)):
        if supercell[i].properties["label"]==center_atom_label:
            supercell_center_atom_indices.append(i)    
            
            
            
    events = []
    events_dict = []
    
    def _equivalent_position_in_periodic_supercell(site_belongs_to_supercell=[5,1,7],image_of_site=(0,-1,1),supercell_shape=[5,6,7],additional_input=False,verbose=False):
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
    
    
    for supercell_center_atom_index in supercell_center_atom_indices:
        
        # for center atoms of supercell, find the neighbors
        
        
        this_center_atom_belongs_to_supercell=supercell[supercell_center_atom_index].properties["supercell"]
        wyckoff_sequence_of_this_center_atom=supercell[supercell_center_atom_index].properties["wyckoff_sequence"]
        local_env_info=[]# list of integer / indices of local environment
        
        
        """_summary_
        
        In order to accelerate the speed
        
        Use a modified pymatgen structure
        
        use the dictionary to store the index of atoms
        
        
        
        """        
        indices_dict_from_identifier=supercell.kmc_build_dict(skip_check=False)

        
        for neighbor_site_in_primitive_cell in local_env_info_dict[wyckoff_sequence_of_this_center_atom]:
            #print(_equivalent_position_in_periodic_supercell(site_belongs_to_supercell=this_center_atom_belongs_to_supercell,image_of_site=neighbor_site_in_primitive_cell["image"],supercell_shape=supercell_shape))
            
            #local_env_info.append(supercell.find_site_by_wyckoff_sequence_label_and_supercell(wyckoff_sequence=neighbor_site_in_primitive_cell["wyckoff_sequence"],label=neighbor_site_in_primitive_cell["label"],supercell=_equivalent_position_in_periodic_supercell(site_belongs_to_supercell=this_center_atom_belongs_to_supercell,image_of_site=neighbor_site_in_primitive_cell["image"],supercell_shape=supercell_shape),return_index=True))# too slow!
            """_summary_
            
            In order to accelerate the speed

            use the dictionary to store the index of atoms
            
            indices_dict_from_identifier is a dictionary by pymatgen_structure.kmc_build_dict()
            
            the key to dictionary is kmc_info_to_tuple
            
            """
            
            local_env_info.append(indices_dict_from_identifier[supercell.kmc_info_to_tuple(wyckoff_sequence=neighbor_site_in_primitive_cell["wyckoff_sequence"],label=neighbor_site_in_primitive_cell["label"],supercell=_equivalent_position_in_periodic_supercell(site_belongs_to_supercell=this_center_atom_belongs_to_supercell,image_of_site=neighbor_site_in_primitive_cell["image"],supercell_shape=supercell_shape))])

        
        if verbose:
            
            np.set_printoptions(precision=3)
            print("finding local environment of",supercell[supercell_center_atom_index],"the local info is ",local_env_info)
            center_distance_matrix=np.array([])
            for local_env_index in local_env_info:
                if verbose==2:
                    print("distance from center to environment:",supercell[supercell_center_atom_index].properties,supercell[local_env_index].properties,supercell[supercell_center_atom_index].distance(supercell[local_env_index]))#debug
                center_distance_matrix=np.append(center_distance_matrix,supercell[supercell_center_atom_index].distance(supercell[local_env_index]))
            print("center atom distance matrix: ",center_distance_matrix)
                
                
                
                
            sites_distance_matrix=np.array([])
                
            for local_env_index1 in local_env_info:
                for local_env_index2 in local_env_info:
            #        pass
                    if verbose==2:
                        print("distance of two local environment site,index1:",supercell[local_env_index1].properties," index2:",supercell[local_env_index2].properties," distance:",supercell[local_env_index1].distance(supercell[local_env_index2]))
                    

                    
                    sites_distance_matrix=np.append(sites_distance_matrix,supercell[local_env_index1].distance(supercell[local_env_index2]))
                    

                    
            print("sites distance matrix:",sites_distance_matrix)
                    
            print("done for this center atom\n--------------------------------------\n\n\n")
        
        for local_env in local_env_info:
                    
            if supercell[local_env].properties["label"] == diffuse_to_atom_label:
                # or for understanding, if any site in local environment, its label== "Na2"
                # initialize the event
                this_event = Event()
                this_event.initialization2(supercell_center_atom_index,local_env,local_env_info)
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
    
    np.savetxt('./events_site_list.txt',np.array(events_site_list,dtype=int),fmt="%i")
    generate_event_kernal(len(supercell),np.array(events_site_list),event_kernal_fname=event_kernal_fname)
    pass
    
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