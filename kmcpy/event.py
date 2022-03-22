"""
Event is a database storing site and cluster info for each diffusion event

Author: Zeyu Deng
Email: dengzeyu@gmail.com
"""
import numpy as np

from copy import deepcopy
import json
from kmcpy.io import convert

class Event: 
    """
    na1_index
    na2_index
    sorted_sublattice_indices
    """
    def __init__(self):
        pass

    def initialization(self,na1_index,na2_index,local_env_info):
        self.na1_index = na1_index
        self.na2_index = na2_index
        self.sorted_sublattice_indices = self.analyze_local_structure(local_env_info) # this is the sublattice indices that matches with the local cluster expansion
        self.local_env_indices_list = [i['site_index'] for i in local_env_info]
        self.local_env_indices_list_site = [i['site_index'] for i in local_env_info]
        
    def initialization2(self,center_atom=12,diffuse_to=15,sorted_sublattice_indices=[1,2,3,4,5]):
        self.na1_index = center_atom
        self.na2_index = diffuse_to
        self.sorted_sublattice_indices = sorted_sublattice_indices # this is the sublattice indices that matches with the local cluster expansion
        self.local_env_indices_list = sorted_sublattice_indices
        self.local_env_indices_list_site = sorted_sublattice_indices
        
    def set_sublattice_indices(self,sublattice_indices,sublattice_indices_site):
        self.sublattice_indices = sublattice_indices# this stores the site indices from local_cluster_expansion object
        self.sublattice_indices_site = sublattice_indices_site # this stores the site indices from local_cluster_expansion object

    def show_info(self):
        print('Event: Na(1)[',self.na1_index,']<--> Na(2)[',self.na2_index,']')
        # print('Global sites indices are (excluding O and Zr):',self.sorted_sublattice_indices)
        # print('Local template structure:')
        # print(self.sorted_local_structure)

        try:
            print('occ_sublat\tE_KRA\tProbability')
            print(self.occ_sublat,'\t',self.ekra,'\t',self.probability)
        except:
            pass
    
    def clear_property(self):
        pass

    def analyze_local_structure(self,local_env_info):
        # 
        indices_sites_group = [(s['site_index'],s['site']) for s in local_env_info]
        
        # this line is to sort the neighbors. First sort by x coordinate, and then sort by specie (Na, then Si/P)
        # the sorted list, store the sequence of hash.
        # for other materials, need to find another method to sort.
        # this sort only works for the NaSICON!
        indices_sites_group_sorted = sorted(sorted(indices_sites_group,key=lambda x:x[1].coords[0]),key = lambda x:x[1].specie)
        
        
        sorted_sublattice_indices = [s[0] for s in indices_sites_group_sorted]
        return sorted_sublattice_indices
    
    # @profile
    def set_occ(self,occ_global):
        self.occ_sublat = deepcopy(occ_global[self.sorted_sublattice_indices]) # occ is an 1D numpy array
    
    # @profile
    def initialize_corr(self):
        self.corr = np.empty(shape=len(self.sublattice_indices))
        self.corr_site = np.empty(shape=len(self.sublattice_indices_site))

    # @profile
    def set_corr(self):
        _set_corr(self.corr,self.occ_sublat, self.sublattice_indices)
        _set_corr(self.corr_site,self.occ_sublat,self.sublattice_indices_site)
        
    # @profile
    def set_ekra(self,keci,empty_cluster,keci_site,empty_cluster_site):# input is the keci and empty_cluster; ekra = corr*keci + empty_cluster
        self.ekra = np.inner(self.corr,keci)+empty_cluster
        self.esite = np.inner(self.corr_site,keci_site)+empty_cluster_site

    # @profile
    def set_probability(self,occ_global,v,T): # calc_probability() will evaluate diffusion probability for this event, should be updated everytime when change occupation
        k = 8.617333262145*10**(-2) # unit in meV/K
        direction = (occ_global[self.na2_index] - occ_global[self.na1_index])/2 # 1 if na1 -> na2, -1 if na2 -> na1
        self.barrier = self.ekra+direction*self.esite/2 # ekra 
        self.probability = abs(direction)*v*np.exp(-1*(self.barrier)/(k*T))

    # @profile
    def update_event(self,occ_global,v,T,keci,empty_cluster,keci_site,empty_cluster_site):
        self.set_occ(occ_global) # change occupation and correlation for this unit
        self.set_corr()
        self.set_ekra(keci,empty_cluster,keci_site,empty_cluster_site)    #calculate ekra and probability
        self.set_probability(occ_global,v,T)

    def as_dict(self):
        d = {"na1_index":self.na1_index,
        "na2_index":self.na2_index,
        "sorted_sublattice_indices":self.sorted_sublattice_indices,
        "local_env_indices_list":self.local_env_indices_list,
        "local_env_indices_list_site":self.local_env_indices_list_site}
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
        obj = Event()
        obj.__dict__ = objDict
        return obj
    
    @classmethod
    def from_dict(self,event_dict): # convert dict into event object
        event = Event()
        event.__dict__ = event_dict
        return event


def _set_corr(corr,occ_latt,sublattice_indices):
    i = 0
    for sublat_ind_orbit in sublattice_indices:
        corr[i]=0
        for sublat_ind_cluster in sublat_ind_orbit:
            corr_cluster = 1
            for occ_site in sublat_ind_cluster:
                corr_cluster*=occ_latt[occ_site]
            corr[i]+= corr_cluster
        i+=1