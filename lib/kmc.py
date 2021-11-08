#!/usr/bin/env python
"""
Function and classes for KMC running

Author: Zeyu Deng
Email: dengzeyu@gmail.com
"""

import pandas as pd
import random
import json
import numpy as np
from pymatgen.core import Structure,Lattice

from copy import copy
import sys
sys.path.append('./')
from model import Event
import pickle
from numba.typed import List

class KMC:
    def __init__(self):
        pass

    def initialization(self,occ,prim_fname,fitting_results,fitting_results_site,event_fname,supercell_shape,v,T,
    lce_fname='../inputs/lce.pkl',lce_site_fname='../inputs/lce_site.pkl'):
        print('Initializing KMC calculations with pirm.json at',prim_fname,'...')
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
        fitting_results = pd.read_pickle(fitting_results).sort_values(by=['time_stamp'],ascending=False).iloc[0]
        self.keci = fitting_results.keci
        self.empty_cluster = fitting_results.empty_cluster

        print('Loading fitting results (site energy) ...')
        fitting_results_site = pd.read_pickle(fitting_results_site).sort_values(by=['time_stamp'],ascending=False).iloc[0]
        self.keci_site = fitting_results_site.keci
        self.empty_cluster_site = fitting_results_site.empty_cluster

        print('Loading occupation:',occ)
        self.occ_global = copy(occ)
        print('Fitted time and error (LOOCV,RMS)')
        print(fitting_results.time_stamp,fitting_results.loocv,fitting_results.rmse)
        print('Fitted KECI and empty cluster')
        print(self.keci,self.empty_cluster)

        print('Fitted time and error (LOOCV,RMS) (site energy)')
        print(fitting_results_site.time_stamp,fitting_results_site.loocv,fitting_results_site.rmse)
        print('Fitted KECI and empty cluster (site energy)')
        print(self.keci_site,self.empty_cluster_site)
        local_cluster_expansion = load_project(lce_fname)
        local_cluster_expansion_site = load_project(lce_site_fname)
        sublattice_indices = _convert_list(local_cluster_expansion.sublattice_indices)
        sublattice_indices_site = _convert_list(local_cluster_expansion_site.sublattice_indices)
        print('Initializing correlation matrix and ekra for all events ...')
        events_site_list = []
        events = load_project(event_fname)
        for event in events:
            event.set_sublattice_indices(sublattice_indices,sublattice_indices_site)
            event.initialize_corr()
            event.update_event(self.occ_global,v,T,self.keci,self.empty_cluster,self.keci_site,self.empty_cluster_site)
            events_site_list.append(event.sorted_sublattice_indices)

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

    def run(self,kmc_pass,equ_pass,v,T,events,comp,structure_idx):
        print('Simulation condition: v =',v,'T = ',T)
        self.v = v
        self.T = T
        pass_length = len([el.symbol for el in self.structure.species if 'Na' in el.symbol])
        print('============================================================')
        print('Start running KMC ... ')
        print('\nInitial occ_global, prob_list and prob_cum_list')
        print('Starting Equilbrium ...')
        for current_pass in np.arange(equ_pass):
            for this_kmc in np.arange(pass_length):
                event,time_change = self.propose(events)
                self.update(event,events)


        print('Start KMC runing ...')
        tracker = Tracker(self.occ_global,self.structure,self.T,self.v)
        print('Pass\tTime\t\tMSD\t\tD_J\t\tD_tracer\tConductivity\tH_R\t\tf\t\tOccNa(1)\tOccNa(2)')
        for current_pass in np.arange(kmc_pass):
            for this_kmc in np.arange(pass_length):
                event,time_change = self.propose(events)
                tracker.update(event,self.occ_global,time_change)
                self.update(event,events)

            previous_conduct = tracker.summary(current_pass,comp,structure_idx)
            tracker.show_current_info(current_pass,comp)

        tracker.write_results(comp,structure_idx,current_pass,self.occ_global)

class Tracker:
    """
    Tracker has a data structure of tracker[na_si_idx]
    """
    def __init__(self,occ_initial,structure,T,v):
        print('Initializing Tracker ...')
        self.T = T
        self.v = v
        self.occ_initial = copy(occ_initial)
        self.frac_coords = structure.frac_coords
        self.latt = structure.lattice
        self.volume = structure.volume
        self.n_na_sites = len([el.symbol for el in structure.species if 'Na' in el.symbol])
        self.n_si_sites = len([el.symbol for el in structure.species if 'Si' in el.symbol])
        self.na_locations = np.where(self.occ_initial[0:self.n_na_sites]==-1)[0] # na_si_site_indices[na_si_indices]
        print('Initial Na locations =',self.na_locations)
        self.n_na = len(self.na_locations)
        self.n_si = len(np.where(self.occ_initial[self.n_na_sites:]==-1)[0])
        print('n_Na =',self.n_na,'n_Na_sites = ',self.n_na_sites)
        self.frac_na_at_na1 = [np.count_nonzero(self.na_locations < self.n_na_sites/4)/self.n_na]
        print('n_Na% @ Na(1) =',self.frac_na_at_na1[0])
        self.displacement = np.zeros((len(self.na_locations),3)) # displacement stores the displacement vector for each ion
        self.hop_counter = np.zeros(len(self.na_locations),dtype=np.int64) 
        self.time = 0
       # self.barrier = []
        self.results = {'time':[],'D_J':[],'D_tracer':[],'conductivity':[],'f':[],'H_R':[],
        'final_na_at_na1':[],'average_na_at_na1':[],'msd':[]}
        
        print('Center of mass (Na):',np.mean(self.frac_coords[self.na_locations]@self.latt.matrix,axis=0))
        self.r0 = self.frac_coords[self.na_locations]@self.latt.matrix

    def update(self,event,current_occ,time_change): # this should be called after update() of KMC run
        na_1_coord = copy(self.frac_coords[event.na1_index])
        na_2_coord = copy(self.frac_coords[event.na2_index])
        na_1_occ = current_occ[event.na1_index]
        na_2_occ = current_occ[event.na2_index]

        # print('---------------------Tracker info---------------------')
        # print('Na(1) at',na_1_coord,'with idx:',event.na1_index,na_1_occ)
        # print('Na(2) at',na_2_coord,'with idx:',event.na2_index,na_2_occ)
        # print('Time now:',self.time)
        # print('Hop counter: ',self.hop_counter)
        # print(event.probability)
        # print('Before update Na locations =',self.na_locations)
        # print('Occupation before update: ',current_occ)
        direction = int((na_2_occ - na_1_occ)/2) # na1 -> na2 direction = 1; na2 -> na1 direction = -1
        displacement_frac = copy(direction*(na_2_coord - na_1_coord))
        displacement_frac -= np.array([int(round(i)) for i in displacement_frac]) # for periodic condition
        displacement_cart = copy(self.latt.get_cartesian_coords(displacement_frac))
        if direction == -1: # Na(2) -> Na(1)
            # print('Diffuse direction: Na(2) -> Na(1)')
            na_to_diff = np.where(self.na_locations==event.na2_index)[0][0]
            self.na_locations[na_to_diff] = event.na1_index
        elif direction == 1: # Na(1) -> Na(2)
            # print('Diffuse direction: Na(1) -> Na(2)')
            na_to_diff = np.where(self.na_locations==event.na1_index)[0][0]
            self.na_locations[na_to_diff] = event.na2_index
        else:
            print('Proposed a wrong event! Please check the code!')
        self.displacement[na_to_diff] += copy(np.array(displacement_cart))
        self.hop_counter[na_to_diff] += 1
        self.time+=time_change
        self.frac_na_at_na1.append(np.count_nonzero(self.na_locations < self.n_na_sites/4)/self.n_na)


    def calc_D_J(self,d=3):
        displacement_vector_tot = np.linalg.norm(np.sum(self.displacement,axis=0))
        D_J = displacement_vector_tot**2/(2*d*self.time*self.n_na)*10**(-16) # to cm^2/s
        return D_J
    
    def calc_D_tracer(self,d=3):
        D_tracer = np.mean(np.linalg.norm(self.displacement,axis=1)**2)/(2*d*self.time)*10**(-16)

        return D_tracer
    
    def calc_corr_factor(self,a=3.47782): # a is the hopping distance in Angstrom
        corr_factor = np.linalg.norm(self.displacement,axis=1)**2/(self.hop_counter*a**2)
        corr_factor = np.nan_to_num(corr_factor,nan=0)
        return np.mean(corr_factor)

    def calc_conductivity(self,D_J,D_tracer,q=1, T = 300):
        n = (self.n_na)/self.volume # e per Angst^3 vacancy is the carrier
        k = 8.617333262145*10**(-2) # unit in meV/K
        conductivity = D_J*n*q**2/(k*T)*1.602*10**11 # to mS/cm
        # print('Conductivty: sigma = ',conductivity,'mS/cm')
        return conductivity

    def show_current_info(self,current_pass,comp):
        print('%d\t%.3E\t%.3E\t%.3E\t%.3E\t%.3E\t%.3E\t%.3E\t%4.2f\t%4.2f' % (current_pass,self.time,self.results['msd'][-1],self.results['D_J'][-1],self.results['D_tracer'][-1],self.results['conductivity'][-1],self.results['H_R'][-1],self.results['f'][-1],
        (4-3*comp)*self.frac_na_at_na1[-1],(4-3*comp)/3*(1-self.frac_na_at_na1[-1])))
        # print('Center of mass (Na):',np.mean(self.frac_coords[self.na_locations]@self.latt.matrix,axis=0))
        # print('MSD = ',np.linalg.norm(np.sum(self.displacement,axis=0))**2,'time = ',self.time)

    def summary(self,current_pass,comp,structure_idx):
        # print('\nTracker Summary:')
        # print('comp =',comp)
        # print('structure_idx =',structure_idx)
        # print('Time elapsed: ',self.time)
        # print('Current pass: ',current_pass)
        # print('T = ',self.T,'K','v = ',self.v)
        # print('Na ratio (Na/(Na+Va)) =',self.n_na/self.n_na_sites)
        # print('Si ratio (Si/(Si+P)) =',self.n_si/self.n_si_sites)
        # print('Displacement vectors r_i = ')
        # print(self.displacement)
        # print('Hopping counts n_i = ')
        # print(self.hop_counter)
        # print('average n_Na% @ Na(1) =',sum(self.frac_na_at_na1)/len(self.frac_na_at_na1))
        # print('final n_Na% @ Na(1) =',self.frac_na_at_na1[-1])
        # print('final Occ Na(1):',(4-3*comp)*self.frac_na_at_na1[-1])
        # print('final Occ Na(2):',(4-3*comp)/3*(1-self.frac_na_at_na1[-1]))

        D_J = self.calc_D_J()
        D_tracer = self.calc_D_tracer()
        f = self.calc_corr_factor()
        conductivity = self.calc_conductivity(D_J=D_J,D_tracer=D_tracer,q = 1,T=self.T)
        H_R = D_tracer/D_J
        
        msd =  np.mean(np.linalg.norm(self.displacement,axis=1)**2) # MSD = sum_i(|r_i|^2)/N

        # print('Haven\'s ratio H_R =',H_R)

        self.results['D_J'].append(D_J)
        self.results['D_tracer'].append(D_tracer)
        self.results['f'].append(f)
        self.results['H_R'].append(H_R)
        self.results['conductivity'].append(conductivity)
        self.results['time'].append(copy(self.time))
        self.results['final_na_at_na1'].append(copy(self.frac_na_at_na1[-1]))
        self.results['average_na_at_na1'].append(copy(sum(self.frac_na_at_na1)/len(self.frac_na_at_na1)))
        self.results['msd'].append(msd)

        return conductivity
    
    def write_results(self,comp,structure_idx,current_pass,current_occupation):
        np.savetxt('displacement_'+str(comp)+'_'+str(structure_idx)+'_'+str(current_pass)+'.csv.gz',self.displacement,delimiter=',')
        np.savetxt('hop_counter_'+str(comp)+'_'+str(structure_idx)+'_'+str(current_pass)+'.csv.gz',self.hop_counter,delimiter=',')
    #    np.savetxt('barrier_'+str(comp)+'_'+str(structure_idx)+'_'+str(current_pass)+'.csv.gz',self.barrier,delimiter=',')
        np.savetxt('current_occ_'+str(comp)+'_'+str(structure_idx)+'_'+str(current_pass)+'.csv.gz',current_occupation,delimiter=',')

        df = pd.DataFrame(self.results)
        df.to_csv('results_'+str(comp)+'_'+str(structure_idx)+'.csv.gz',compression='gzip')
        

def _convert_list(list_to_convert):
    converted_list = List([List(List(j) for j in i ) for i in list_to_convert])
    return converted_list


def save_project(project,fname):
    print('Saving:',fname)
    with open(fname,'wb') as fhandle:
        pickle.dump(project,fhandle)

def load_project(fname):
    print('Loading:',fname)
    with open(fname,'rb') as fhandle:
        obj = pickle.load(fhandle)
    return obj