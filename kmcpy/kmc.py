"""
Function and classes for running kMC

Author: Zeyu Deng
Email: dengzeyu@gmail.com
"""

import random
import numpy as np
from pymatgen.core import Structure,Lattice
from numba.typed import List
from pymatgen.core.structure import Structure
import numpy as np
import pandas as pd
from copy import copy
import json
from kmcpy.model import LocalClusterExpansion
from kmcpy.tracker import Tracker
from kmcpy.event import Event
class KMC:
    def __init__(self):
        pass

    def initialization(self,occ=np.array([-1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1]),prim_fname='./inputs/prim.json',fitting_results='./inputs/fitting_results.json',fitting_results_site='./inputs/fitting_results_site.json',event_fname="./inputs/events.json",supercell_shape=[2,1,1],v=5000000000000,T=298,lce_fname="./inputs/lce.json",lce_site_fname="./inputs/lce_site.json",**kwargs):
        print('Initializing kMC calculations with pirm.json at',prim_fname,'...')
        with open(prim_fname,'r') as f:
            prim = json.load(f)
            prim_coords = [site['coordinate'] for site in prim['basis']]
            prim_specie_cases = [site['occupant_dof'] for site in prim['basis']]
            prim_lattice = Lattice(prim['lattice_vectors'])
            prim_species = [s[0] for s in prim_specie_cases]

        supercell_shape_matrix = np.diag(supercell_shape)
        print('Supercell Shape:\n',supercell_shape_matrix)
        self.structure = Structure(prim_lattice,prim_species,prim_coords)
        print('Converting supercell ...')
        self.structure.remove_species(['Zr','O'])
        self.structure.make_supercell(supercell_shape_matrix)
        print('Loading fitting results ...')
        fitting_results = pd.read_json(fitting_results,orient='index').sort_values(by=['time_stamp'],ascending=False).iloc[0]
        self.keci = fitting_results.keci
        self.empty_cluster = fitting_results.empty_cluster

        print('Loading fitting results (site energy) ...')
        fitting_results_site = pd.read_json(fitting_results_site,orient='index').sort_values(by=['time_stamp'],ascending=False).iloc[0]
        self.keci_site = fitting_results_site.keci
        self.empty_cluster_site = fitting_results_site.empty_cluster

        print('Loading occupation:',occ)
        self.occ_global = copy(occ)
        print('Fitted time and error (LOOCV,RMS)')
        print(fitting_results.time,fitting_results.loocv,fitting_results.rmse)
        print('Fitted KECI and empty cluster')
        print(self.keci,self.empty_cluster)

        print('Fitted time and error (LOOCV,RMS) (site energy)')
        print(fitting_results_site.time,fitting_results_site.loocv,fitting_results_site.rmse)
        print('Fitted KECI and empty cluster (site energy)')
        print(self.keci_site,self.empty_cluster_site)
        local_cluster_expansion = LocalClusterExpansion.from_json(lce_fname)
        local_cluster_expansion_site = LocalClusterExpansion.from_json(lce_site_fname)
        sublattice_indices = _convert_list(local_cluster_expansion.sublattice_indices)
        sublattice_indices_site = _convert_list(local_cluster_expansion_site.sublattice_indices)
        print('Initializing correlation matrix and ekra for all events ...')
        events_site_list = []
        
        print('Loading events from:',event_fname)
        with open(event_fname,'rb') as fhandle:
            events_dict = json.load(fhandle)
        
        events = []
        for event_dict in events_dict:
            event = Event.from_dict(event_dict)
            event.set_sublattice_indices(sublattice_indices,sublattice_indices_site)
            event.initialize_corr()
            event.update_event(self.occ_global,v,T,self.keci,self.empty_cluster,self.keci_site,self.empty_cluster_site)
            events_site_list.append(event.sorted_sublattice_indices)
            events.append(event)
            
        self.prob_list = [event.probability for event in events]
        self.prob_cum_list = np.cumsum(self.prob_list)
        return events
        
    def load_site_event_list(self,fname="../event_kernal_generator/results.csv"):# workout the site_event_list -> site_event_list[site_index] will return a list of event index to update if a site_index is chosen
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
        self.site_event_list =site_event_list

    def show_project_info(self):
        try:
            print('Probabilities:')
            print(self.prob_list)
            print('Cumultative probability list:')
            print(self.prob_cum_list/sum(self.prob_list))
        except:
            pass

    def propose(self,events): # propose() will propose an event to be updated by update()
        random_seed = random.random()
        random_seed_2 = random.random()
        proposed_event_index = np.searchsorted(self.prob_cum_list/(self.prob_cum_list[-1]),random_seed,side='right')
        time_change = (-1.0/self.prob_cum_list[-1])*np.log(random_seed_2)

        event = events[proposed_event_index]

        return event, time_change

    def update(self,event,events):
        self.occ_global[event.na1_index]*=(-1)
        self.occ_global[event.na2_index]*=(-1)
        events_to_be_updated = copy(self.site_event_list[event.na2_index])
        for e_index in events_to_be_updated:
            events[e_index].update_event(self.occ_global,self.v,self.T,self.keci,self.empty_cluster,self.keci_site,self.empty_cluster_site)
            self.prob_list[e_index] = copy(events[e_index].probability)
        self.prob_cum_list = np.cumsum(self.prob_list)

    def run_from_database(self,kmc_pass=1000,equ_pass=1,v=5000000000000,T=298,events="./inputs/events.json",comp=1,structure_idx=1,**kwargs):
        print('Simulation condition: v =',v,'T = ',T)
        self.v = v
        self.T = T
        pass_length = len([el.symbol for el in self.structure.species if 'Na' in el.symbol])
        print('============================================================')
        print('Start running kMC ... ')
        print('\nInitial occ_global, prob_list and prob_cum_list')
        print('Starting Equilbrium ...')
        for current_pass in np.arange(equ_pass):
            for this_kmc in np.arange(pass_length):
                event,time_change = self.propose(events)
                self.update(event,events)

        print('Start running kMC ...')
        tracker = Tracker()
        tracker.initialization(self.occ_global,self.structure,self.T,self.v)
        print('Pass\tTime\t\tMSD\t\tD_J\t\tD_tracer\tConductivity\tH_R\t\tf\t\tOccNa(1)\tOccNa(2)')
        for current_pass in np.arange(kmc_pass):
            for this_kmc in np.arange(pass_length):
                event,time_change = self.propose(events)
                tracker.update(event,self.occ_global,time_change)
                self.update(event,events)

            previous_conduct = tracker.summary(comp,current_pass)
            tracker.show_current_info(comp,current_pass)

        tracker.write_results(comp,structure_idx,current_pass,self.occ_global)

    # def run(self,kmc_pass,equ_pass,v,T,events):
    #     print('Simulation condition: v =',v,'T = ',T)
    #     self.v = v
    #     self.T = T
    #     pass_length = len([el.symbol for el in self.structure.species if 'Na' in el.symbol])
    #     print('============================================================')
    #     print('Start running kMC ... ')
    #     print('\nInitial occ_global, prob_list and prob_cum_list')
    #     print('Starting Equilbrium ...')
    #     for current_pass in np.arange(equ_pass):
    #         for this_kmc in np.arange(pass_length):
    #             event,time_change = self.propose(events)
    #             self.update(event,events)

    #     print('Start running kMC ...')
    #     tracker = Tracker()
    #     tracker.initialization(self.occ_global,self.structure,self.T,self.v)
    #     print('Pass\tTime\t\tMSD\t\tD_J\t\tD_tracer\tConductivity\tH_R\t\tf\t\tOccNa(1)\tOccNa(2)')
    #     for current_pass in np.arange(kmc_pass):
    #         for this_kmc in np.arange(pass_length):
    #             event,time_change = self.propose(events)
    #             tracker.update(event,self.occ_global,time_change)
    #             self.update(event,events)

    #         previous_conduct = tracker.summary(comp,current_pass)
    #         tracker.show_current_info(current_pass)

    #     tracker.write_results(None,None,current_pass,self.occ_global)

    def as_dict(self):
        d = {"@module":self.__class__.__module__,
        "@class": self.__class__.__name__,
        "structure":self.structure.as_dict(),
        "keci":self.keci,
        "empty_cluster":self.empty_cluster,
        "keci_site":self.keci_site,
        "empty_cluster_site":self.empty_cluster_site,
        "occ_global":self.occ_global}
        return d
    
    def to_json(self,fname):
        print('Saving:',fname)
        with open(fname,'w') as fhandle:
            d = self.as_dict()
            jsonStr = json.dumps(d,indent=4,default=convert) # to get rid of errors of int64
            fhandle.write(jsonStr)
    
    @classmethod
    def from_json(self,fname):
        print('Loading:',fname)
        with open(fname,'rb') as fhandle:
            objDict = json.load(fhandle)
        obj = KMC()
        obj.__dict__ = objDict
        print("load complete")
        return obj

def _convert_list(list_to_convert):
    converted_list = List([List(List(j) for j in i ) for i in list_to_convert])
    return converted_list
def convert(o):
    if isinstance(o, np.int64): return int(o)
    elif isinstance(o, np.int32): return int(o)  
    raise TypeError
