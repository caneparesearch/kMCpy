#!/usr/bin/env python
"""
This module provides utilities for handling KMC simulation I/O operations.
"""

import numpy as np
import json
from kmcpy.external.structure import StructureKMCpy
import logging
import pandas as pd
import warnings

logger = logging.getLogger(__name__)

# Model registry for extensible model types
MODEL_REGISTRY = {
    "lce": "lce",  # Maps model type to task type for backward compatibility
    "composite_lce": "lce"  # CompositeLCEModel still uses lce task type
}

# Task registry for different simulation types
TASK_REGISTRY = {
    "kmc": "kmc",
    "model": "lce", 
    "generate_event": "generate_event"
} 

def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    elif isinstance(o, np.int32):
        return int(o)
    raise TypeError

class Results:
    """
    Results class for storing and manipulating simulation results from the Tracker.

    Attributes
    ----------
    data : dict
        Dictionary containing lists for each tracked property:
        - "time": Simulation time steps.
        - "D_J": Jump diffusion coefficients.
        - "D_tracer": Tracer diffusion coefficients.
        - "conductivity": Ionic conductivities.
        - "f": Correlation factors.
        - "H_R": Haven ratios.
        - "msd": Mean squared displacements.

    Methods
    -------
    add(time, D_J, D_tracer, conductivity, f, H_R, msd)
        Add a new set of results to the data.

    to_dataframe()
        Convert the stored results to a pandas DataFrame.

    merge(other)
        Merge results from another Results instance into this one.

    clear()
        Clear all stored results.

    __getitem__(key)
        Get the list of values for a given property.

    __setitem__(key, value)
        Set the list of values for a given property.
    """
    def __init__(self):
        self.data = {
            "time": [],
            "D_J": [],
            "D_tracer": [],
            "conductivity": [],
            "f": [],
            "H_R": [],
            "msd": [],
        }

    def add(self, time, D_J, D_tracer, conductivity, f, H_R, msd):
        self.data["time"].append(time)
        self.data["D_J"].append(D_J)
        self.data["D_tracer"].append(D_tracer)
        self.data["conductivity"].append(conductivity)
        self.data["f"].append(f)
        self.data["H_R"].append(H_R)
        self.data["msd"].append(msd)

    def to_dataframe(self):
        return pd.DataFrame(self.data)

    def merge(self, other):
        for key in self.data:
            self.data[key].extend(other.data.get(key, []))

    def clear(self):
        for key in self.data:
            self.data[key] = []

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value