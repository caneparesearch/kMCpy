#!/usr/bin/env python
"""
This module defines the Event class, which encapsulates the information required to represent 
a migration event in a lattice-based simulation, such as those used in kinetic Monte Carlo (kMC) studies. 

The Event class focuses purely on defining which sites are involved in the hop and providing 
the local environment indices. All energy calculations are now handled by the model classes.
"""

import numpy as np
import numba as nb
import json
from kmcpy.io import convert
import logging
from abc import ABC
from numba.typed import List

logger = logging.getLogger(__name__) 
logging.getLogger('numba').setLevel(logging.WARNING)

class Event:
    """
    Represents a migration event in a lattice-based simulation.
    
    The Event class focuses purely on defining which sites are involved in the hop
    and providing the local environment indices. All calculations are now handled
    by the model classes.
    
    Attributes:
        mobile_ion_indices (tuple): Global indices of the mobile ions involved in the event
        local_env_indices (tuple): Global indices of the neighboring sites in the supercell
    """

    def __init__(self, mobile_ion_indices: tuple, local_env_indices: tuple):
        """Initialize the Event object with the indices of the mobile ions and their local environment.

        Args:
            mobile_ion_indices (tuple): A tuple containing the two global indices 
                of the mobile ions involved in the event.
            local_env_indices (tuple): A tuple of integers representing the sorted 
                indices of the neighboring sites in the supercell.
        """
        self.mobile_ion_indices = mobile_ion_indices
        self.local_env_indices = local_env_indices

    def show_info(self):
        """Display information about the event."""
        logger.info(
            "Event: mobile_ion(1)[%s] <--> mobile_ion(2)[%s]",
            self.mobile_ion_indices[0],
            self.mobile_ion_indices[1],
        )
        logger.debug('Local environment indices: %s', self.local_env_indices)

    def as_dict(self):
        d = {
            "mobile_ion_indices": self.mobile_ion_indices,
            "local_env_indices": self.local_env_indices,
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
    def from_dict(cls, event_dict):
        """Create Event from dictionary."""
        mobile_ion_indices = event_dict.get("mobile_ion_indices", (12, 15))
        local_env_indices = event_dict.get("local_env_indices", (1, 2, 3, 4, 5))
        return cls(mobile_ion_indices, local_env_indices)


@nb.njit
def _generate_event_dependency_matrix(events_site_list):
    """
    Generate the event dependency matrix.
    
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
        sites_array = np.array([site for site in event_sites], dtype=np.int64)
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
                              
    Note: All indices (mobile_ion_indices and local_env_indices) are global site indices, 
    despite the misleading name of local_env_indices.
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
        # mobile_ion_indices and local_env_indices 
        # all contain global indices
        sites_involved = []
        # Add mobile ion indices
        for site in event.mobile_ion_indices:
            sites_involved.append(site)
        # Add local environment indices
        for site in event.local_env_indices:
            sites_involved.append(site)
        
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
        
        Note: All indices (mobile_ion_indices and 
        local_env_indices) are global site indices.
        
        Returns:
            list: 2D list where event_dependencies[i] contains indices of events that 
                  depend on event i (need to be updated when event i is executed).
        """
        logger.info("Generating and storing event dependency matrix in class...")
        
        # Build events_site_list for numba function using all global site indices
        # mobile_ion_indices and local_env_indices
        # all contain global indices
        events_site_list = List()
        for event in self.events:
            sites_involved = List()
            # Add mobile ion indices individually to avoid extend() deprecation warning
            for site in event.mobile_ion_indices:
                sites_involved.append(site)
            # Add local environment indices individually
            for site in event.local_env_indices:
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
