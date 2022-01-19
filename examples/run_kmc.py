#!/usr/bin/env python
"""
This code is used to run KMC

Author: Zeyu Deng
Email: dengzeyu@gmail.com

run this as:
python run_kmc.py comp structure_index T
"""
from kmcpy.kmc import KMC
import pandas as pd
import sys

v = 0.5*10**13
#extreme senerio -> 15*10**13
equ_pass= 1
kmc_pass = 3000
supercell_shape = (8,8,8)
fitting_results = './inputs/fitting_results.pkl'
fitting_results_site = './inputs/fitting_results_site.pkl'

prim_fname = './inputs/prim.json'
events = './inputs/events_888_site.pkl'
event_kernal = './inputs/event_kernal_888.csv'
mc_results = './mc_results.h5'

def main():
    comp = sys.argv[1]
    structure_index = sys.argv[2]
    T = int(sys.argv[3])
    df = pd.read_hdf(mc_results)
    occ = df.loc[(abs(df['comp']-float(comp))<1e-4) & (df['structure_index']==int(structure_index))].occ.to_numpy()[0]
    # step 1 initialize global occupation and conditions
    kmc = KMC()
    events_initialized = kmc.initialization(occ,prim_fname,fitting_results,fitting_results_site,events,supercell_shape,v,T) # v in 10^13 hz

    # # step 2 compute the site kernal (used for kmc run)
    kmc.load_site_event_list(event_kernal)

    # # step 3 run kmc
    kmc.run(int(kmc_pass),int(equ_pass), float(v), float(T),events_initialized,float(comp),int(structure_index))

if __name__ == "__main__":
    main()
