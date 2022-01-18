#!/usr/bin/env python
"""
Functions to build the model
"""
from pymatgen.core.structure import Molecule,Structure
import numpy as np
from itertools import combinations
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
import glob
import sys
from copy import deepcopy
sys.path.append('./')
import numba as nb
import pickle,json

class LocalClusterExpansion:
    """
    LocalClusterExpansion will be initialized with a template structure where all the sites are occupied
    cutoff_cluster is the cutoff for pairs and triplet
    cutoff_region is the cutoff for generating local cluster region
    """
    def __init__(self):
        pass

    def initialization(self,center_Na1_index=0,cutoff_cluster=[6,6,6],cutoff_region=4,template_cif_fname='EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif',is_write_basis=False):
        template_structure = Structure.from_file(template_cif_fname)
        template_structure.remove_oxidation_states()
        self.center_Na1 = template_structure[center_Na1_index]
        template_structure.remove_sites([center_Na1_index])
        template_structure.remove_species(['Zr4+','O2-','O','Zr'])
        print('Searching local env around',self.center_Na1,'...')
        self.diffusion_unit_structure = self.get_cluster_structure(structure = template_structure,cutoff = cutoff_region, center_site = self.center_Na1,is_write_basis = is_write_basis)
        # List all possible point, pair and triplet clusters
        atom_index_list = np.arange(0,len(self.diffusion_unit_structure))
        cluster_indexes = list(combinations(atom_index_list,1))+list(combinations(atom_index_list,2))+list(combinations(atom_index_list,3))+list(combinations(atom_index_list,4))
        print(len(cluster_indexes),'clusters will be generated ...')
        self.clusters = self.clusters_constructor(cluster_indexes,[10]+cutoff_cluster)
        self.orbits = self.orbits_constructor(self.clusters)
        self.sublattice_indices = [[cluster.site_indices for cluster in orbit.clusters] for orbit in self.orbits] # sublattice_indices[orbit,cluster,site]
        print('Type','Index','max_length','min_length','Point Group','Multiplicity',sep='\t')
        for orbit in self.orbits:
            orbit.show_representative_cluster()

    def get_cluster_structure(self,structure,center_site,cutoff = 4,is_write_basis=False): # return a molecule structure centeret center_site
        local_env_structure = [s[0] for s in structure.get_sites_in_sphere(center_site.coords,cutoff)]
        local_env_list_sorted = sorted(sorted(local_env_structure,key=lambda x:x.coords[0]),key = lambda x:x.specie)
        local_env_structure = Molecule.from_sites(local_env_list_sorted)
        local_env_structure.translate_sites(np.arange(0,len(local_env_structure),1),-1*center_site.coords)
        if is_write_basis:
            print('Local environemnt: ')
            print(local_env_structure)
            local_env_structure.to('xyz','local_env.xyz')
            print('The point group of local environment is: ',PointGroupAnalyzer(local_env_structure).sch_symbol)
        return local_env_structure

    def get_occupation_neb_cif(self,other_cif_name): # input is a cif structure
        occupation = []
        other_structure = Structure.from_file(other_cif_name)
        other_structure.remove_oxidation_states()
        other_structure.remove_species(['Zr4+','O2-','O','Zr'])
        other_structure_mol = self.get_cluster_structure(other_structure,self.center_Na1)
        for this_site in self.diffusion_unit_structure:
            if self.is_exists(this_site,other_structure_mol):# Chebyshev basis is used here: ±1
                occu = -1
            else:
                occu = 1
            occupation.append(occu)
        return occupation
    
    def get_correlation_matrix_neb_cif(self,other_cif_names):
        correlation_matrix = []
        occupation_matrix = []
        for other_cif_name in sorted(glob.glob(other_cif_names)):
            occupation = self.get_occupation_neb_cif(other_cif_name)
            correlation = [(orbit.multiplicity)*orbit.get_cluster_function(occupation) for orbit in self.orbits]
            occupation_matrix.append(occupation)
            correlation_matrix.append(correlation)
            print(other_cif_name,occupation)
        self.correlation_matrix = correlation_matrix

        print(np.round(correlation_matrix,decimals=3))
        np.savetxt(fname='occupation.txt',X=occupation_matrix,fmt='%5d')
        np.savetxt(fname='correlation_matrix.txt',X=correlation_matrix,fmt='%.8f')
        self.correlation_matrix = correlation_matrix
        # print(other_cif_name,occupation,np.around(correlation,decimals=3),sep='\t')
        
    def is_exists(self,this_site,other_structure):
        # 2 things to compare: 1. cartesian coords 2. species at each site
        is_exists = False
        for s_other in other_structure:
            if (np.linalg.norm(this_site.coords - s_other.coords) < 1e-3) and (this_site.species == s_other.species):
                is_exists = True
        return is_exists

    def clusters_constructor(self,indexes,cutoff): # return a list of Cluster
        clusters = []
        print('\nGenerating possible clusters within this diffusion unit...')
        print('Cutoffs: pair =',cutoff[1],'Angst, triplet =',cutoff[2],'Angst, quadruplet =',cutoff[3],'Angst')
        for site_indices in indexes:
            sites = [self.diffusion_unit_structure[s] for s in site_indices]
            cluster = Cluster(site_indices,sites)
            if cluster.max_length < cutoff[len(cluster.site_indices)-1]:
                clusters.append(cluster)
        return clusters

    def orbits_constructor(self,clusters):
        """
        return a list of Orbit

        For each orbit, loop over clusters
            for each cluster, check if this cluster exists in this orbit
                if not, attach the cluster to orbit
                else, 
        """
        orbit_clusters= []
        grouped_clusters = []
        for i in clusters:
            if i not in orbit_clusters:
                orbit_clusters.append(i)
                grouped_clusters.append([i])
            else:
                grouped_clusters[orbit_clusters.index(i)].append(i)
        orbits = []
        for i in grouped_clusters:
            orbit = Orbit()
            for cl in i:
                orbit.attach_cluster(cl)
            orbits.append(orbit)
        return orbits

    def __str__(self):
        print('\nGLOBAL INFORMATION')
        print('Number of orbits =',len(self.orbits))
        print("Number of clusters =",len(self.clusters))

    def write_representative_clusters(self,path='.'):
        print('Writing representative structures to xyz files to',path,'...')
        for i,orbit in enumerate(self.orbits):
            orbit.clusters[0].to_xyz(path+'/orbit_'+str(i)+'.xyz')

    def to_json(self,fname):
        print('Saving:',fname)
        with open(fname,'w') as fhandle:
            d = self.as_dict()
            jsonStr = json.dumps(d,indent=4,default=convert) # to get rid of errors of int64
            fhandle.write(jsonStr)

    def as_dict(self):
        d = {"@module":self.__class__.__module__,
        "@class": self.__class__.__name__,
        "center_Na1":self.center_Na1.as_dict(),
        "diffusion_unit_structure":self.diffusion_unit_structure.as_dict(),
        "clusters":[],
        "orbits":[],
        "sublattice_indices":self.sublattice_indices}

        for cluster in self.clusters:
            d["clusters"].append(cluster.as_dict())
        for orbit in self.orbits:
            d["orbits"].append(orbit.as_dict())
        return d


    @classmethod
    def from_json(self,fname):
        print('Loading:',fname)
        with open(fname,'rb') as fhandle:
            objDict = json.load(fhandle)
        obj = LocalClusterExpansion()
        obj.__dict__ = objDict
        return obj



class Orbit:# orbit is a collection of symmetry equivalent clusters
    def __init__(self):
        self.clusters = []
        self.multiplicity = 0

    def attach_cluster(self,cluster):
        self.clusters.append(cluster)
        self.multiplicity+=1

    def get_cluster_function(self,occupancy): # Phi[orbit] = 1/multiplicity * sum(prod(cluster))
        cluster_function  = (1/self.multiplicity)*sum([c.get_cluster_function(occupancy) for c in self.clusters])
        return cluster_function

    def __str__(self):
        try:
            for i,cluster in enumerate(self.clusters):
                print("Cluster[",i,"]: {0:5s}\t{1:10s}\t{2:8.3f}\t{3:8.3f}\t{4:5s}\t{5:5d}".format(cluster.type,str(cluster.site_indices),cluster.max_length,cluster.min_length,cluster.sym,self.multiplicity))
        except TypeError:
            print('No cluster in this orbit!')

    def to_xyz(self,fname):
        self.clusters[0].to_xyz(fname)
    
    def show_representative_cluster(self):
        print("{0:5s}\t{1:10s}\t{2:8.3f}\t{3:8.3f}\t{4:5s}\t{5:5d}".format(self.clusters[0].type,str(self.clusters[0].site_indices),self.clusters[0].max_length,self.clusters[0].min_length,self.clusters[0].sym,self.multiplicity))

    def as_dict(self):
        d = {"@module":self.__class__.__module__,
        "@class": self.__class__.__name__,
        "clusters":[],
        "multiplicity":self.multiplicity}
        for cluster in self.clusters:
            d["clusters"].append(cluster.as_dict())
        return d

class Cluster:
    def __init__(self,site_indices,sites):
        cluster_type = {1:'point',2:'pair',3:'triplet',4:'quadruplet'}
        self.site_indices = site_indices
        self.type = cluster_type[len(site_indices)]
        self.structure = Molecule.from_sites(sites)
        self.sym = PointGroupAnalyzer(self.structure).sch_symbol
        if self.type == 'point':
            self.max_length = 0
            self.min_length = 0
            self.bond_distances = []
        else:
            self.max_length,self.min_length,self.bond_distances = self.get_bond_distances()

    def __eq__(self,other): # to compare 2 clusters and check if they're the same by comparing atomic distances
        if self.type != other.type:
            return False
        elif self.type == 'point' and other.type == 'point':
            return self.structure.species == other.structure.species
        else:
            return np.linalg.norm(self.bond_distances-other.bond_distances) < 1e-3

    def get_site(self):
        return [self.diff_unit_structure[s] for s in self.index]

    def get_bond_distances(self):
        indices_combination = list(combinations(np.arange(0,len(self.structure)),2))
        bond_distances = np.array([self.structure.get_distance(*c) for c in indices_combination])
        bond_distances.sort()
        max_length = max(bond_distances)
        min_length = min(bond_distances)
        return max_length, min_length , bond_distances
    
    def get_cluster_function(self,occupation):
        cluster_function = np.prod([occupation[i] for i in self.site_indices])
        return cluster_function
    
    def to_xyz(self,fname):
        local_structure_no_oxidation = self.structure.copy()
        local_structure_no_oxidation.remove_oxidation_states()
        local_structure_no_oxidation.to('xyz',fname)
    
    def __str__(self):
        print('==============================================================')
        print('This cluster is a',self.type,', constructed by site',self.site_indices)
        print(f'max length = {self.max_length:.3f} Angst ,min_length = {self.min_length:.3f} Angst')
        print('Point Group: ',self.sym)
        try:
            print('Cluster function = ',self.cluster_function_string)
        except:
            pass
        print('==============================================================\n')
    
    def as_dict(self):
        d = {"@module":self.__class__.__module__,
        "@class": self.__class__.__name__,
        "site_indices":self.site_indices,
        "type":self.type,
        "structure":self.structure.as_dict(),
        "sym":self.sym,
        "max_length":self.max_length,
        "min_length":self.min_length}
        if type(self.bond_distances) is list:
            d['bond_distances']=self.bond_distances
        else:
            d['bond_distances']=self.bond_distances.tolist()
        return d


class Event: # Event is a database storing site and cluster info for each diffusion unit
    """
    na1_index
    na2_index
    sorted_sublattice_indices
    """
    def __init__(self,na1_index,na2_index,local_env_info):
        self.na1_index = na1_index
        self.na2_index = na2_index
        self.sorted_sublattice_indices = self.analyze_local_structure(local_env_info) # this is the sublattice indices that matches with the local cluster expansion
        self.local_env_indices_list = [i['site_index'] for i in local_env_info]
        self.local_env_indices_list_site = [i['site_index'] for i in local_env_info]

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
        indices_sites_group = [(s['site_index'],s['site']) for s in local_env_info]
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
    def set_ekra(self,keci,empty_cluster,keci_site,empty_cluster_site):# input is the keci and empty_cluster; ekra = corr*keci + empty_cluster=
        self.ekra = np.inner(self.corr,keci)+empty_cluster
        self.esite = np.inner(self.corr_site,keci_site)+empty_cluster_site

    # @profile
    def set_probability(self,occ_global,v,T): # calc_probability() will evaluate diffusion probability for this event, should be updated everytime when change occupation
        k = 8.617333262145*10**(-2) # unit in meV/K
        direction = (occ_global[self.na2_index] - occ_global[self.na1_index])/2 # 1 if na1 -> na2, -1 if na2 -> na1
        self.barrier = self.ekra+direction*self.esite/2 # ekra 
        self.probability = abs(direction)*v*np.exp(-1*(self.barrier)/(k*T))
        # if self.barrier<0:
        #     print(self.barrier)
        # if direction>0:
        #     print('Na(1)[',self.na1_index,'] -> Na(2)[',self.na2_index,'] with Eb =',self.barrier)
        # elif direction <0:
        #     print('Na(2)[',self.na2_index,'] -> Na(1)[',self.na1_index,'] with Eb =',self.barrier)
        # else:
        #     print('Na not moving')


    
    # @profile

    def update_event(self,occ_global,v,T,keci,empty_cluster,keci_site,empty_cluster_site):
        self.set_occ(occ_global) # change occupation and correlation for this unit
        self.set_corr()
        self.set_ekra(keci,empty_cluster,keci_site,empty_cluster_site)    #calculate ekra and probability
        self.set_probability(occ_global,v,T)

    
    def as_dict(self):
        d = {"time_stamp":self.time_stamp,
        "weight":self.weight,
        "alpha":self.alpha,
        "keci":self.keci,
        "empty_cluster":self.empty_cluster,
        "rmse":self.rmse,
        "loocv":self.loocv}

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
        obj = Fitting()
        obj.__dict__ = objDict
        return obj


    
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

class Fitting:
    def __init__(self):
        pass

    def add_data(self,time_stamp,keci,empty_cluster,weight,alpha,rmse,loocv):
        self.time_stamp = time_stamp
        self.weight = weight
        self.alpha = alpha
        self.keci = keci
        self.empty_cluster = empty_cluster
        self.rmse = rmse
        self.loocv =loocv
    
    def save_file(self):
        save_project(self,'fitting_results.pkl')
    
    def as_dict(self):
        d = {"time_stamp":self.time_stamp,
        "weight":self.weight,
        "alpha":self.alpha,
        "keci":self.keci,
        "empty_cluster":self.empty_cluster,
        "rmse":self.rmse,
        "loocv":self.loocv}
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
        obj = Fitting()
        obj.__dict__ = objDict
        return obj

def save_project(project,fname):
    print('Saving:',fname)
    with open(fname,'wb') as fhandle:
        pickle.dump(project,fhandle)


def load_project(fname):
    print('Loading:',fname)
    with open(fname,'rb') as fhandle:
        obj = pickle.load(fhandle)
    return obj


def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError