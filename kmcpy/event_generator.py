#!/usr/bin/env python

from kmcpy.event import Event
import numpy as np
from numba.typed import List
import numba as nb
from kmcpy.io import convert

def generate_events(prim_fname,supercell_shape,event_fname):
    import json
    from pymatgen.core.structure import Structure
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
        prim_species = [s[0] for s in prim_specie_cases]
        
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
    n_na1_per_prim = 2
    n_na1 = n_na1_per_prim*np.prod(shape) 
    local_env_cutoff_dict = {('Na','Na'):4,('Na','Si'):4}
    ncpus = multiprocessing.cpu_count()
    print('Looking for all possible swaps with ncpus =',ncpus,' (this might take a while) ...')
    na_1_sites = structure[0:n_na1]
    local_env_finder = CutOffDictNN(local_env_cutoff_dict)
    local_env_info_list = list(Parallel(n_jobs=ncpus)(delayed(local_env_finder.get_nn_info)(structure,i) for i in np.arange(0,n_na1)))
    events = []
    events_dict = []
    for (na1_index,(na1_site,local_env_info)) in enumerate(zip(na_1_sites,local_env_info_list)):
        for local_env in local_env_info:
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
    generate_event_kernal(len(structure),np.array(events_site_list))

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
        event_kernal[i] tabulates the index of events that have to be updated after event[i] has been executed
        
        
    
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

