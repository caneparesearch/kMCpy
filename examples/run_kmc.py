#!/usr/bin/env python
"""
Example code to run kinetic Monte Carlo

Author: Zeyu Deng
Email: dengzeyu@gmail.com

run this as:
python run_kmc.py T
"""
from kmcpy.kmc import KMC
import sys
from kmcpy.io import load_occ

v = 0.5*10**13
equ_pass= 1
kmc_pass = 3000
supercell_shape = (2,1,1)
fitting_results = './inputs/fitting_results.json'
fitting_results_site = './inputs/fitting_results_site.json'
lce = './inputs/lce.json'
lce_site = './inputs/lce_site.json'
prim_fname = './inputs/prim.json'
events = './inputs/events.json'
event_kernal = './inputs/event_kernal.csv'
mc_results = './initial_state.json'

def main():
    T = int(sys.argv[1])
    occ = load_occ(mc_results,supercell_shape)
    # step 1 initialize global occupation and conditions
    kmc = KMC()
    events_initialized = kmc.initialization(occ,prim_fname,fitting_results,fitting_results_site,events,supercell_shape,v,T,lce,lce_site) # v in 10^13 hz

    # # step 2 compute the site kernal (used for kmc run)
    kmc.load_site_event_list(event_kernal)

    # # step 3 run kmc
    kmc.run(int(kmc_pass),int(equ_pass), float(v), float(T),events_initialized)

if __name__ == "__main__":
    main()
