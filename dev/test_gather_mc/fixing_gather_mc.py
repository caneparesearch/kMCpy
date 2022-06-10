#!/usr/bin/env python
import numpy as np
import pandas as pd
import glob2,os,json
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
import numba as nb
from numba.typed import List
from joblib import Parallel, delayed
import multiprocessing
from numba.types import float64, int64
from kmcpy.external.pymatgen_structure import Structure
def generate_supercell(prim_fname,supercell_shape):
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
    # structure.to(fmt='cif',filename='supercell.cif')
    return structure

def gather_data(path,template_structure):
    locations = glob2.glob(path)
    print(locations)
    get_occ(locations[0]+'/conditions.0/trajectory/POSCAR.final',template_structure)
    occ = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(get_occ)(location+'/conditions.0/trajectory/POSCAR.final',template_structure) for location in locations)
    data = []
    for index,location in enumerate(locations):
        x = float(location.split('_')[1])
        structure_index = int(location.split('_')[2])
        print(location,occ[index])
        data.append([x,structure_index,occ[index]])
    return pd.DataFrame(data,columns=['comp','structure_index','occ']).sort_values(by=['comp','structure_index'])

def isExist(s,structure):
    is_exist = False
    for site_in_structure in structure.sites:
        if s.species == site_in_structure.species:
            if site_in_structure.distance(s) < 1e-3:
                is_exist = True
                break
    return is_exist

"""
get_occ() works out the occupation vector by comparing the POSCAR generated from the Monte Carlo with the template structure using following steps:
1. get frac_coords and species of each sites in casm_structure and template_structure
2. take frac_coords difference by considering periodic boundary condition
3. for those norm(difference) = 0 (should be only one site), compare their species
4. if norm(difference) ==0 and species matches, then occ=-1 otherwise occ=1
"""
def get_occ(mc_poscar,template_structure):
    print(mc_poscar)
    casm_structure = Structure.from_file(mc_poscar)
    casm_structure.remove_species(['Zr','O'])
    occ = []
    template_frac_coords = np.array(template_structure.frac_coords)
    template_species = List([i.symbol for i in template_structure.species])
    casm_frac_coords = np.array(casm_structure.frac_coords)
    casm_species = List([i.symbol for i in casm_structure.species])
    lattice=template_structure.lattice.matrix

    occ = _get_occ(casm_frac_coords,template_frac_coords,casm_species,template_species,lattice)
    print(occ)
    print(len(np.where(occ==-1)[0]))

    print('Finished',mc_poscar)
    return np.array(occ)

@nb.njit
def _get_occ(casm_frac_coords,template_frac_coords,casm_species,template_species,lattice):
    occ = np.zeros(len(template_species),dtype=np.int64)
    for i,(template_frac_coord,template_specie) in enumerate(zip(template_frac_coords,template_species)):
        diff_frac = casm_frac_coords-template_frac_coord
        diff_frac = diff_frac-rnd(diff_frac) # for periodic condition
        diff_cart = diff_frac@lattice
        dists_cart = norm(diff_cart)
        matched_site_idx = np.where(dists_cart<1e-6)[0]
        if matched_site_idx.size> 0 and casm_species[matched_site_idx[0]] in template_specie:
            occ[i]=-1
        else:
            print(matched_site_idx.size)
            occ[i]=1
        # else:
        #     print(i,'is a Vacancy')
    return occ

@nb.njit
def rnd(x):
    out = np.empty_like(x)
    np.around(x,0,out)
    return out

@nb.njit
def norm(x):
    out = np.zeros(len(x))
    for j,i in enumerate(x):
        out[j] = (i[0]**2+i[1]**2+i[2]**2)**0.5
    return out

json_structure=generate_supercell("prim.json",[2,2,2])
cif_structure=Structure.from_cif("EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",primitive=True)
cif_structure.remove_species(['Zr','O',"Zr4+","O2-"])
cif_structure.remove_oxidation_states()
cif_structure=cif_structure.make_kmc_supercell([2,2,2])
print("json",json_structure)
print("cif",cif_structure)
cif2_structure=Structure.from_file("EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",primitive=True)
print("cif2",cif2_structure)
cif2_structure.remove_species(['Zr','O',"Zr4+","O2-"])
cif2_structure.remove_oxidation_states()
cif2_structure.make_supercell([2,2,2])
print("cif2",cif2_structure)
#exit()



template_structure = generate_supercell('./prim.json',(8,8,8))
#print(template_structure)
df = gather_data('comp*',template_structure)
#print(df)
df.to_hdf('mc_results.h5',key='df',complevel=9,mode='w')
df.to_json('mc_results.json',orient="index")
occ1=df["occ"]


structure_from_cif=Structure.from_cif("EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",primitive=True)
print("mark1")
cif_structure.remove_species(['Zr','O',"Zr4+","O2-"])
cif_structure.remove_oxidation_states()
structure_from_cif=structure_from_cif.make_kmc_supercell([8,8,8])



df2 = gather_data('comp*',structure_from_cif)
#print(df)
df2.to_hdf('mc_results_cif.h5',key='df',complevel=9,mode='w')
df2.to_json('mc_results_cif.json',orient="index")

occ2=df2["occ"]


for i in range(0,len(occ1)):
    if occ1[i]!=occ2[i]:
        print(i,occ1[i],occ2[i])