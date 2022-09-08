"""
Event is a database storing site and cluster info for each migration event

Author: Zeyu Deng
Email: dengzeyu@gmail.com
"""
import numpy as np
import numba as nb
from copy import deepcopy
import json
from kmcpy.io import convert

class Event: 
    """
    mobile_ion_specie_1_index
    mobile_ion_specie_2_index
    local_env_indices_list
    """
    def __init__(self):
        pass

   
    def initialization3(self,mobile_ion_specie_1_index=12,mobile_ion_specie_2_index=15,local_env_indices_list=[1,2,3,4,5]):
        """3rd version of initialization. The input local_env_indices_list is already sorted. Center atom is equivalent to the Na1 in the 1st version and mobile_ion_specie_2_index is equivalent to the Na2 in the 1st version

        Args:
            mobile_ion_specie_1_index (int, optional): the global index (index in supercell) of the center atom. Defaults to 12.
            mobile_ion_specie_2_index (int, optional): the global index of the atom that the center atom is about to diffuse to. Defaults to 15.
            local_env_indices_list (list, optional): list of integers, which is a list of indices of the neighboring sites in supercell, and is already sorted. Defaults to [1,2,3,4,5].
        """
        self.mobile_ion_specie_1_index = mobile_ion_specie_1_index
        self.mobile_ion_specie_2_index = mobile_ion_specie_2_index

        self.local_env_indices_list = local_env_indices_list 
        self.local_env_indices_list_site = local_env_indices_list
        
    def set_sublattice_indices(self,sublattice_indices,sublattice_indices_site):
        self.sublattice_indices = sublattice_indices# this stores the site indices from local_cluster_expansion object
        self.sublattice_indices_site = sublattice_indices_site # this stores the site indices from local_cluster_expansion object

    def show_info(self):
        print('Event: mobile_ion(1)mobile_ion(1)[',self.mobile_ion_specie_1_index,']<--> mobile_ion(2)[',self.mobile_ion_specie_2_index,']')
        # print('Global sites indices are (excluding O and Zr):',self.local_env_indices_list)
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
        
        
        local_env_indices_list = [s[0] for s in indices_sites_group_sorted]
        return local_env_indices_list
    
    # @profile
    def set_occ(self,occ_global):
        self.occ_sublat = deepcopy(occ_global[self.local_env_indices_list]) # occ is an 1D numpy array
    
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
    def set_probability(self,occ_global,v,T): # calc_probability() will evaluate migration probability for this event, should be updated everytime when change occupation
        k = 8.617333262145*10**(-2) # unit in meV/K
        direction = (occ_global[self.mobile_ion_specie_2_index] - occ_global[self.mobile_ion_specie_1_index])/2 # 1 if na1 -> na2, -1 if na2 -> na1
        self.barrier = self.ekra+direction*self.esite/2 # ekra 
        self.probability = abs(direction)*v*np.exp(-1*(self.barrier)/(k*T))

    # @profile
    def update_event(self,occ_global,v,T,keci,empty_cluster,keci_site,empty_cluster_site):
        self.set_occ(occ_global) # change occupation and correlation for this unit
        self.set_corr()
        self.set_ekra(keci,empty_cluster,keci_site,empty_cluster_site)    #calculate ekra and probability
        self.set_probability(occ_global,v,T)

    def as_dict(self):
        d = {"mobile_ion_specie_1_index":self.mobile_ion_specie_1_index,
        "mobile_ion_specie_2_index":self.mobile_ion_specie_2_index,
        "local_env_indices_list":self.local_env_indices_list,
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

@nb.njit
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