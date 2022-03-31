#!/usr/bin/env python

from kmcpy.event import Event
import numpy as np

from kmcpy.io import convert

def generate_events(api=1,**kwargs):
    
    if api==1:
        return generate_events1(**kwargs)
    elif api>=2:
        return generate_events2(**kwargs)
    else:
        raise NotImplementedError("debug information from event_generator.generate_events. Unsupport API value: api=",api)




def generate_events2(prim_cif_name="EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",supercell_shape=[2,1,1],local_env_cutoff_dict = {('Na+','Na+'):4,('Na+','Si4+'):4},event_fname="events.json",event_kernal_fname='event_kernal.csv',center_atom_label_or_indices="Na1",diffuse_to_atom_label="Na2",species_to_be_removed=['Zr4+','O2-','O','Zr'],verbose=False):

    import json
    from kmcpy.external.pymatgen_structure import Structure
    from pymatgen.analysis.local_env import CutOffDictNN
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
        print("receive string in center atom label parameter. Trying to find the indices of center atom")        
        for i in range(0,len(primitive_cell)):
            if primitive_cell[i].properties["label"]==center_atom_label_or_indices:
                center_atom_indices.append(i)
                print("please check if this is a center atom:",primitive_cell[i],primitive_cell[i].properties)
        print("all center atom indices:",center_atom_indices)
        
    elif type(center_atom_label_or_indices) is list:
        center_atom_indices=center_atom_label_or_indices
        center_atom_label=primitive_cell[center_atom_label_or_indices[0]].properties["label"]
        
    else:
        raise TypeError("center_atom_label_or_indices is either string or list")
                
                
                
    # find nearest neighbor in primitive cell    
    local_env_finder = CutOffDictNN(local_env_cutoff_dict)
    local_env_info_dict = {}
    """0: [{'site': PeriodicSite: Si4+ (-3.2361, -0.3015, 9.2421) [-0.3712, -0.0379, 0.4167], 'image': (-1, -1, 0), 'weight': 3.7390091507903174, 'site_index': 39, 'wyckoff_sequence': 15, 'local_index': 123, 'label': 'Si1'}, {'site': PeriodicSite: Na+ (-1.2831, -2.6519, 9.2421) [-0.3063, -0.3333, 0.4167], 'image': (-1, -1, 0), 'weight': 3.4778161424304046, 'site_index': 23, 'wyckoff_sequence': 17, 'local_index': 35, 'label': 'Na2'}, {'site': 
    """
    for center_atom_index in center_atom_indices:
    
        local_env_info_dict[primitive_cell[center_atom_index].properties["wyckoff_sequence"]]=sorted(sorted(local_env_finder.get_nn_info(primitive_cell,center_atom_index),key=lambda x:x["wyckoff_sequence"]),key = lambda x:x["label"])
    
    
    if verbose:    
        print(local_env_info_dict)    
    
    
    # create the supercell
    supercell=primitive_cell.make_kmc_supercell(supercell_shape)
    if verbose:
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
        this_center_atom_belongs_to_supercell=supercell[supercell_center_atom_index].properties["supercell"]
        wyckoff_sequence_of_this_center_atom=supercell[supercell_center_atom_index].properties["wyckoff_sequence"]
        local_env_info=[]# list of integer / indices of local environment
        
        for neighbor_site_in_primitive_cell in local_env_info_dict[wyckoff_sequence_of_this_center_atom]:
            #print(_equivalent_position_in_periodic_supercell(site_belongs_to_supercell=this_center_atom_belongs_to_supercell,image_of_site=neighbor_site_in_primitive_cell["image"],supercell_shape=supercell_shape))
            local_env_info.append(supercell.find_site_by_wyckoff_sequence_label_and_supercell(wyckoff_sequence=neighbor_site_in_primitive_cell["wyckoff_sequence"],label=neighbor_site_in_primitive_cell["label"],supercell=_equivalent_position_in_periodic_supercell(site_belongs_to_supercell=this_center_atom_belongs_to_supercell,image_of_site=neighbor_site_in_primitive_cell["image"],supercell_shape=supercell_shape),return_index=True))

        
        if verbose:
            print("finding local environment of",supercell[supercell_center_atom_index],"the local info is ",local_env_info)
            for local_env_index in local_env_info:
                print("distance from center to environment:",supercell[supercell_center_atom_index].distance(supercell[local_env_index]))
                
            for local_env_index1 in local_env_info:
                for local_env_index2 in local_env_info:
                    pass
                    #print("distance of two local environment site,index1:",local_env_index1," index2:",local_env_index2," distance:",supercell[local_env_index1].distance(supercell[local_env_index2]))
        
        for local_env in local_env_info:
                    
            if supercell[local_env].properties["label"] == diffuse_to_atom_label:
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
    
    np.savetxt('./events_site_list.txt',np.array(events_site_list,dtype=int))
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
    results = []
    for site in all_site_list:
        # print('Looking for site:',site)
        row = []
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