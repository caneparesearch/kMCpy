"""
Tracker is an object to track trajectories of diffusing ions

Author: Zeyu Deng
Email: dengzeyu@gmail.com
"""
import numpy as np
import numpy as np
import pandas as pd
from copy import copy
import json
from kmcpy.io import convert

class Tracker:
    """
    Tracker has a data structure of tracker[na_si_idx]
    """
    def __init__(self):
        pass

    def initialization(self,occ_initial,structure,T,v):
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
        return conductivity

    def show_current_info(self,comp,current_pass):
        print('%d\t%.3E\t%.3E\t%.3E\t%.3E\t%.3E\t%.3E\t%.3E\t%4.2f\t%4.2f' % (current_pass,self.time,self.results['msd'][-1],self.results['D_J'][-1],self.results['D_tracer'][-1],self.results['conductivity'][-1],self.results['H_R'][-1],self.results['f'][-1],
        (4-3*comp)*self.frac_na_at_na1[-1],(4-3*comp)/3*(1-self.frac_na_at_na1[-1])))
        # print('Center of mass (Na):',np.mean(self.frac_coords[self.na_locations]@self.latt.matrix,axis=0))
        # print('MSD = ',np.linalg.norm(np.sum(self.displacement,axis=0))**2,'time = ',self.time)
        
    def return_current_info(self):
        return (self.time,self.results['msd'][-1],self.results['D_J'][-1],self.results['D_tracer'][-1],self.results['conductivity'][-1],self.results['H_R'][-1],self.results['f'][-1])

    def summary(self,comp,current_pass):
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
    
    def as_dict(self):
        d = {"@module":self.__class__.__module__,
        "@class": self.__class__.__name__,
        "T":self.T,
        "occ_initial":self.occ_initial,
        "frac_coords":self.frac_coords,
        "latt":self.latt.as_dict(),
        "volume":self.volume,
        "n_na_sites":self.n_na_sites,
        "n_si_sites":self.n_si_sites,
        "na_locations":self.na_locations,
        "n_na":self.n_na,
        "n_si":self.n_si,
        "frac_na_at_na1":self.frac_na_at_na1,
        "displacement":self.displacement,
        "hop_counter":self.hop_counter,
        "time":self.time,
        "results":self.results,
        "r0":self.r0}
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
        obj = Tracker()
        obj.__dict__ = objDict
        return obj
