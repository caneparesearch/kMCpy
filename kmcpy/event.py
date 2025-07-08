#!/usr/bin/env python
"""
This module defines the Event class, which encapsulates the information and methods required to represent and analyze a migration event in a lattice-based simulation, such as those used in kinetic Monte Carlo (kMC) studies. The Event class manages indices of mobile ions, local environments, sublattice information, and provides methods for initializing, updating, and serializing event data. It also includes methods for calculating occupation, correlation, migration barriers, and probabilities associated with migration events.
"""

import numpy as np
import numba as nb
from copy import deepcopy
import json
from kmcpy.io import convert
import logging
from abc import ABC
from numba.typed import List

logger = logging.getLogger(__name__) 
logging.getLogger('numba').setLevel(logging.WARNING)

class Event:
    """
    mobile_ion_specie_1_index
    mobile_ion_specie_2_index
    local_env_indices_list
    """

    def __init__(self):
        pass

    def initialization(
        self,
        mobile_ion_specie_1_index=12,
        mobile_ion_specie_2_index=15,
        local_env_indices_list=[1, 2, 3, 4, 5],
    ):
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

    def set_sublattice_indices(self, sublattice_indices, sublattice_indices_site):
        self.sublattice_indices = sublattice_indices  # this stores the site indices from local_cluster_expansion object
        self.sublattice_indices_site = sublattice_indices_site  # this stores the site indices from local_cluster_expansion object

    def show_info(self):
        logger.info(
            "Event: mobile_ion(1)[%s]<--> mobile_ion(2)[%s]",
            self.mobile_ion_specie_1_index,
            self.mobile_ion_specie_2_index,
        )
        logger.debug('Global sites indices are (excluding O and Zr): %s', self.local_env_indices_list)
        logger.debug('Local template structure: %s', getattr(self, 'sorted_local_structure', None))

        try:
            logger.info("occ_sublat\tE_KRA\tProbability")
            logger.info("%s\t%s\t%s", self.occ_sublat, self.ekra, self.probability)
        except Exception:
            pass

    def clear_property(self):
        pass

    def analyze_local_structure(self, local_env_info):
        #
        indices_sites_group = [(s["site_index"], s["site"]) for s in local_env_info]

        # this line is to sort the neighbors. First sort by x coordinate, and then sort by specie (Na, then Si/P)
        # the sorted list, store the sequence of hash.
        # for other materials, need to find another method to sort.
        # this sort only works for the NaSICON!
        indices_sites_group_sorted = sorted(
            sorted(indices_sites_group, key=lambda x: x[1].coords[0]),
            key=lambda x: x[1].specie,
        )

        local_env_indices_list = [s[0] for s in indices_sites_group_sorted]
        return local_env_indices_list

    # @profile
    def set_occ(self, occ_global):
        self.occ_sublat = deepcopy(
            occ_global[self.local_env_indices_list]
        )  # occ is an 1D numpy array

    # @profile
    def initialize_corr(self):
        self.corr = np.empty(shape=len(self.sublattice_indices))
        self.corr_site = np.empty(shape=len(self.sublattice_indices_site))

    # @profile
    def set_corr(self):
        _set_corr(self.corr, self.occ_sublat, self.sublattice_indices)
        _set_corr(self.corr_site, self.occ_sublat, self.sublattice_indices_site)

    # @profile
    def set_ekra(
        self, keci, empty_cluster, keci_site, empty_cluster_site
    ):  # input is the keci and empty_cluster; ekra = corr*keci + empty_cluster
        self.ekra = np.inner(self.corr, keci) + empty_cluster
        self.esite = np.inner(self.corr_site, keci_site) + empty_cluster_site

    # @profile
    def set_probability(
        self, occ_global, v, T
    ):  # calc_probability() will evaluate migration probability for this event, should be updated everytime when change occupation
        k = 8.617333262145 * 10 ** (-2)  # unit in meV/K
        direction = (
            occ_global[self.mobile_ion_specie_2_index]
            - occ_global[self.mobile_ion_specie_1_index]
        ) / 2  # 1 if na1 -> na2, -1 if na2 -> na1
        self.barrier = self.ekra + direction * self.esite / 2  # ekra
        self.probability = abs(direction) * v * np.exp(-1 * (self.barrier) / (k * T))

    # @profile
    def update_event(
        self, occ_global, v, T, keci, empty_cluster, keci_site, empty_cluster_site
    ):
        self.set_occ(occ_global)  # change occupation and correlation for this unit
        self.set_corr()
        self.set_ekra(
            keci, empty_cluster, keci_site, empty_cluster_site
        )  # calculate ekra and probability
        self.set_probability(occ_global, v, T)

    def as_dict(self):
        d = {
            "mobile_ion_specie_1_index": self.mobile_ion_specie_1_index,
            "mobile_ion_specie_2_index": self.mobile_ion_specie_2_index,
            "local_env_indices_list": self.local_env_indices_list,
            "local_env_indices_list": self.local_env_indices_list,
            "local_env_indices_list_site": self.local_env_indices_list_site,
        }
        return d

    def to_json(self, fname):
        logger.info("Saving: %s", fname)
        with open(fname, "w") as fhandle:
            d = self.as_dict()
            jsonStr = json.dumps(
                d, indent=4, default=convert
            )  # to get rid of errors of int64
            fhandle.write(jsonStr)

    @classmethod
    def from_json(self, fname):
        logger.info("Loading: %s", fname)
        with open(fname, "rb") as fhandle:
            objDict = json.load(fhandle)
        obj = Event()
        obj.__dict__ = objDict
        return obj
    @classmethod
    def from_dict(self, event_dict):  # convert dict into event object
        event = Event()
        event.__dict__ = event_dict
        return event


@nb.njit
def _set_corr(corr, occ_latt, sublattice_indices):
    i = 0
    for sublat_ind_orbit in sublattice_indices:
        corr[i] = 0
        for sublat_ind_cluster in sublat_ind_orbit:
            corr_cluster = 1
            for occ_site in sublat_ind_cluster:
                corr_cluster *= occ_latt[occ_site]
            corr[i] += corr_cluster
        i += 1


@nb.njit
def _generate_event_dependency_matrix(events_site_list):
    """
    Generate the event dependency matrix using numba for performance.
    
    Optimized algorithm that avoids expensive set operations:
    1. Pre-converts site lists to sorted arrays for faster intersection
    2. Uses binary search-like approach for finding overlaps
    3. Reduces memory allocations
    
    Args:
        events_site_list: List of lists, where each inner list contains 
                         the site indices involved in that event
    
    Returns:
        List of Lists: event_dependencies[i] contains indices of events that 
                      depend on event i (need to be updated when event i is executed)
    """
    n_events = len(events_site_list)
    event_dependencies = List()
    
    # Pre-process: convert each event's sites to sorted arrays for faster intersection
    sorted_event_sites = List()
    for event_sites in events_site_list:
        # Convert to numpy array and sort
        sites_array = np.array([site for site in event_sites])
        sites_array.sort()
        sorted_event_sites.append(sites_array)
    
    # Build dependency matrix with optimized intersection checking
    for event_i in range(n_events):
        dependent_events = List()
        sites_i = sorted_event_sites[event_i]
        
        for event_j in range(n_events):
            sites_j = sorted_event_sites[event_j]
            
            # Check if sorted arrays have any common elements
            # This is much faster than set operations
            if _arrays_intersect(sites_i, sites_j):
                dependent_events.append(event_j)
        
        event_dependencies.append(dependent_events)
    
    return event_dependencies


@nb.njit
def _arrays_intersect(arr1, arr2):
    """
    Check if two sorted arrays have any common elements.
    
    Uses two-pointer technique for O(n+m) complexity instead of O(n*m).
    
    Args:
        arr1: Sorted numpy array
        arr2: Sorted numpy array
    
    Returns:
        bool: True if arrays have at least one common element
    """
    i, j = 0, 0
    len1, len2 = len(arr1), len(arr2)
    
    while i < len1 and j < len2:
        if arr1[i] == arr2[j]:
            return True
        elif arr1[i] < arr2[j]:
            i += 1
        else:
            j += 1
    
    return False


class EventLib(ABC):
    """
    A library of events, which can be used to store and manage multiple Event objects.
    
    Attributes:
        events (list): List of Event objects
        event_dependencies (list): 2D list where event_dependencies[i] contains indices of events 
                                 that depend on event i (need to be updated when event i is executed).
                                 Dependencies are determined based on shared global site indices.
        site_to_events (dict): Mapping from global site index to list of event indices 
                              that involve that site
                              
    Note: All indices (mobile_ion_specie_1_index, mobile_ion_specie_2_index, and 
    local_env_indices_list) are global site indices, despite the misleading name 
    of local_env_indices_list.
    """

    def __init__(self):
        self.events = []
        self.event_dependencies = None
        self.site_to_events = {}

    def add_event(self, event):
        """Add an event to the library and update site mappings."""
        event_index = len(self.events)
        self.events.append(event)
        
        # Update site to events mapping using global site indices
        # mobile_ion_specie_1_index, mobile_ion_specie_2_index, and local_env_indices_list 
        # all contain global indices (despite the misleading name of local_env_indices_list)
        sites_involved = set()
        sites_involved.add(event.mobile_ion_specie_1_index)
        sites_involved.add(event.mobile_ion_specie_2_index)
        sites_involved.update(event.local_env_indices_list)
        
        for site in sites_involved:
            if site not in self.site_to_events:
                self.site_to_events[site] = []
            self.site_to_events[site].append(event_index)

    def get_event(self, index):
        return self.events[index]

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        return self.get_event(index)
    
    def __str__(self):
        return f"EventLib with {len(self.events)} events"
    
    def generate_event_dependencies(self):
        """
        Generate the event dependency matrix and store it in the class.
        
        For each event, find all other events that share global sites and thus have dependencies.
        When an event is executed, all dependent events need to be updated.
        The dependency matrix is stored as self.event_dependencies.
        
        Note: All indices (mobile_ion_specie_1_index, mobile_ion_specie_2_index, and 
        local_env_indices_list) are global site indices, despite the misleading name 
        of local_env_indices_list.
        
        Returns:
            list: 2D list where event_dependencies[i] contains indices of events that 
                  depend on event i (need to be updated when event i is executed).
        """
        logger.info("Generating and storing event dependency matrix in class...")
        
        # Build events_site_list for numba function using all global site indices
        # mobile_ion_specie_1_index, mobile_ion_specie_2_index, and local_env_indices_list
        # all contain global indices (despite the misleading name of local_env_indices_list)
        events_site_list = List()
        for event in self.events:
            sites_involved = List()
            sites_involved.append(event.mobile_ion_specie_1_index)
            sites_involved.append(event.mobile_ion_specie_2_index)
            for site in event.local_env_indices_list:
                sites_involved.append(site)
            events_site_list.append(sites_involved)
        
        # Generate and store the dependency matrix using optimized numba function
        self.event_dependencies = _generate_event_dependency_matrix(events_site_list)
        
        logger.info("Event dependency matrix generated and stored with %d events", len(self.event_dependencies))
        return self.event_dependencies
    
    def save_event_dependencies_to_file(self, filename="event_dependencies.csv"):
        """Save the event dependency matrix to a CSV file."""
        if self.event_dependencies is None:
            logger.warning("Event dependencies not generated yet. Call generate_event_dependencies() first.")
            return
            
        logger.info("Saving event dependencies to: %s", filename)
        with open(filename, "w") as f:
            for row in self.event_dependencies:
                for item in row:
                    f.write("%5d " % item)
                f.write("\n")
    
    def get_dependent_events(self, event_index):
        """
        Get all event indices that depend on the given event (need to be updated when it's executed).
        
        Args:
            event_index (int): Index of the event that was executed.
            
        Returns:
            list: List of event indices that depend on the given event.
        """
        if self.event_dependencies is None:
            logger.warning("Event dependencies not generated. Call generate_event_dependencies() first.")
            return []
        
        if event_index >= len(self.event_dependencies):
            logger.error("Event index %d out of range", event_index)
            return []
            
        return list(self.event_dependencies[event_index])
    
    def update_dependent_events(self, executed_event_index, occ_global, v, T, 
                               keci, empty_cluster, keci_site, empty_cluster_site):
        """
        Update all events that depend on the execution of a specific event.
        
        Args:
            executed_event_index (int): Index of the event that was executed
            occ_global: Global occupation array
            v: Attempt frequency
            T: Temperature
            keci: Cluster expansion coefficients
            empty_cluster: Empty cluster contribution
            keci_site: Site cluster expansion coefficients  
            empty_cluster_site: Site empty cluster contribution
        """
        dependent_events = self.get_dependent_events(executed_event_index)
        
        for event_idx in dependent_events:
            self.events[event_idx].update_event(
                occ_global, v, T, keci, empty_cluster, keci_site, empty_cluster_site
            )
        
        logger.debug("Updated %d events dependent on event %d", 
                    len(dependent_events), executed_event_index)

    def as_dict(self):
        """Convert EventLib to dictionary for serialization."""
        # Convert numba Lists to regular Python lists for JSON serialization
        event_dependencies_serializable = None
        if self.event_dependencies is not None:
            event_dependencies_serializable = [[int(idx) for idx in row] for row in self.event_dependencies]
        
        return {
            "events": [event.as_dict() for event in self.events],
            "event_dependencies": event_dependencies_serializable,
            "site_to_events": {str(k): v for k, v in self.site_to_events.items()}  # Convert keys to strings for JSON
        }
    
    def to_json(self, fname):
        """Save EventLib to JSON file."""
        logger.info("Saving EventLib to: %s", fname)
        with open(fname, "w") as fhandle:
            d = self.as_dict()
            jsonStr = json.dumps(d, indent=4, default=convert)
            fhandle.write(jsonStr)
    
    @classmethod
    def from_dict(cls, data):
        """Create EventLib from dictionary."""
        event_lib = cls()
        
        # Load events
        for event_dict in data["events"]:
            event = Event.from_dict(event_dict)
            event_lib.events.append(event)
        
        # Load event dependencies if present and convert back to numba Lists
        if data.get("event_dependencies"):
            event_lib.event_dependencies = List()
            for row in data["event_dependencies"]:
                numba_row = List()
                for idx in row:
                    numba_row.append(int(idx))
                event_lib.event_dependencies.append(numba_row)
        
        # Load site to events mapping (convert string keys back to integers)
        site_to_events_data = data.get("site_to_events", {})
        event_lib.site_to_events = {int(k): v for k, v in site_to_events_data.items()}
        
        return event_lib

    
    @classmethod
    def from_json(cls, fname):
        """
        Load EventLib from a JSON file.
        
        Args:
            fname: The name of the file to load the EventLib from.
        """
        logger.info("Loading EventLib from: %s", fname)
        with open(fname, "r") as fhandle:
            data = json.load(fhandle)
            
        # Handle both old format (list of events) and new format (dict with events and kernel)
        if isinstance(data, list):
            # Old format - just a list of event dictionaries
            event_lib = cls()
            for event_dict in data:
                event = Event.from_dict(event_dict)
                event_lib.add_event(event)
            return event_lib
        else:
            # New format - use from_dict method
            return cls.from_dict(data)
    
    def clear_event_dependencies(self):
        """Clear the event dependency matrix cache. Call this if events are modified."""
        self.event_dependencies = None
        logger.debug("Event dependency matrix cache cleared")
    
    def has_event_dependencies(self):
        """Check if event dependency matrix is generated and stored."""
        return self.event_dependencies is not None
    
    def get_event_dependencies_info(self):
        """Get information about the stored event dependency matrix."""
        if not self.has_event_dependencies():
            return {"status": "not_generated", "message": "Event dependency matrix not generated"}
        
        return {
            "status": "generated",
            "num_events": len(self.event_dependencies),
            "total_dependencies": sum(len(row) for row in self.event_dependencies),
            "memory_usage": "stored in class as numba.typed.List"
        }
    
    def get_events_involving_site(self, site_index):
        """
        Get all events that involve a specific site.
        
        Args:
            site_index (int): The site index to search for.
            
        Returns:
            list: List of event indices that involve the given site.
        """
        return self.site_to_events.get(site_index, [])
    
    def get_dependency_statistics(self):
        """
        Get statistics about the event dependency matrix for analysis.
        
        Returns:
            dict: Dictionary containing dependency matrix statistics.
        """
        if self.event_dependencies is None:
            return {"error": "Event dependency matrix not generated"}
        
        total_dependencies = sum(len(row) for row in self.event_dependencies)
        max_dependencies = max(len(row) for row in self.event_dependencies) if self.event_dependencies else 0
        min_dependencies = min(len(row) for row in self.event_dependencies) if self.event_dependencies else 0
        avg_dependencies = total_dependencies / len(self.event_dependencies) if self.event_dependencies else 0
        
        return {
            "total_events": len(self.events),
            "total_dependencies": total_dependencies,
            "max_dependencies_per_event": max_dependencies,
            "min_dependencies_per_event": min_dependencies,
            "avg_dependencies_per_event": avg_dependencies,
            "dependency_density": total_dependencies / (len(self.events) ** 2) if self.events else 0
        }
