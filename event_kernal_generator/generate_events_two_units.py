#!/usr/bin/env python
import sys
sys.path.append('../lib')
from kmc import KMC
from kmc_tools import load_project,save_project
from model_two_units import Event
import numpy as np
from numba.typed import List
import numba as nb
from scipy.spatial import cKDTree
import json
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
    
def generate_events(prim_fname,supercell_shape,template_na_center_idx,lce_fname='../inputs/lce.pkl'):

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
    lce_obj = load_project(lce_fname)

    n_na1_per_prim = 2
    n_na1 = n_na1_per_prim*np.prod(shape) 
    frac_coords = structure.frac_coords
    frac_coords[frac_coords>=1] -= 1
    print(np.where(frac_coords>=1))

    # This part only works well for NZSP! I don't know whether it works for your system or not! Please test this carefully!
    tree = cKDTree(boxsize = np.array([1,1,1]),data = frac_coords)
    query_data = tree.query(x = structure.frac_coords[0:n_na1],k = 16,p = 2)
    all_local_env_idx = query_data[1][:,0:13]
    print(all_local_env_idx)
    # This part only works well for NZSP! I don't know whether it works for your system or not! Please test this carefully!

    events = []
    
    for (na1_idx,local_env_idx) in enumerate(all_local_env_idx):
        for na2_idx in local_env_idx:
            if "Na" in structure[na2_idx].specie.symbol and na2_idx != na1_idx:
                na1_idx_with_na2 = np.where(all_local_env_idx==na2_idx)[0] # contains 2 na1 (event) indices with this na2
                local_env = set(all_local_env_idx[na1_idx_with_na2].flatten())
                # print(local_env)
                na1_other_idx = na1_idx_with_na2[na1_idx_with_na2 != na1_idx][0]
                event = Event(na1_idx,na2_idx,na1_other_idx)
                event.sort_local_env(structure,np.array(list(local_env)),template_na_center_idx) # <- this needs to be done properly
                events.append(event)
                print(event.__dict__)
                sublat_idx = event.sorted_env_global_indices
                # print(structure[sublat_idx[5]].species,structure[sublat_idx[11]].species,structure[sublat_idx[5]].distance(structure[sublat_idx[11]]))
                # event.show_info()
    save_project(events,'./events_double_units.pkl')
    
    
    # events_site_list = []
    # for event in events:
    #     events_site_list.append(event.sorted_env_global_indices)
    
    # np.savetxt('./events_site_list.txt',np.array(events_site_list,dtype=int))
    # na2_event_table = generate_event_kernal(len(structure),np.array(events_site_list)) # na2_event_table[na2_idx] is a list with all the event index associated with na2_idx

    # na2_event_table = load_site_event_list('./event_kernal.csv')



def load_site_event_list(fname="../event_kernal_generator/results.csv"):# workout the site_event_list -> site_event_list[site_index] will return a list of event index to update if a site_index is chosen
    print('Working at the site_event_list ...')
    print('Loading',fname)
    site_event_list = []
    with open(fname) as f:
        data = f.readlines()
    for x in data:
        if len(x.strip()) == 0:
            site_event_list.append([])
        else:
            site_event_list.append([int(y) for y in x.strip().split()])
    return site_event_list

# @nb.njit
# def _calc_dist_pbc(frac_coord_0,frac_coord_1,lattice_matrix):
#     dist_frac = frac_coord_0-frac_coord_1
#     j = 0
#     for i in dist_frac:
#         dist_frac[j] -= int(round(i))
#         j+=1
#     dist_cart = np.dot(lattice_matrix, dist_frac)
#     return dist_cart

# @nb.njit
# def get_sublat_index(glob_indices,glob_na1_idx,glob_na2_index):
#     """
#     get_sublat_index return a list of events
#     """
#     return 0

#results = [[event_index for event_index,event in enumerate(events) if s in event] for s in np.arange(len_structure)]
@nb.njit
def _generate_event_kernal(len_structure,events_site_list):
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
    
def generate_event_kernal(len_structure,events_site_list):
    event_kernal = _generate_event_kernal(len_structure,events_site_list)
    with open('event_kernal.csv', 'w') as f:
        for row in event_kernal:
            for item in row:
                f.write('%d ' % item)
            f.write('\n')
    return event_kernal

supercell_shape = (4,4,4)
lce_fname = '../local_cluster_expansion_new/local_cluster_expansion.pkl'
lce_site_fname = '../inputs/lce_site.pkl'

prim_fname = '../inputs/prim.json'

# generate_events(prim_fname=prim_fname,supercell_shape=(2,2,2),template_na_center_idx = [15,2],lce_fname=lce_fname)
# generate_events(prim_fname=prim_fname,supercell_shape=supercell_shape,template_na_center_idx = [828,251],lce_fname=lce_fname)
generate_events(prim_fname=prim_fname,supercell_shape=(4,4,4),template_na_center_idx = [126,45],lce_fname=lce_fname)
