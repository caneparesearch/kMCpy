#!/usr/bin/env python
"""
This module provides utilities for handling KMC simulation I/O operations.
"""

import logging
import pandas as pd

from kmcpy.io.registry import MODEL_TASK_REGISTRY
from kmcpy.io.serialization import to_json_compatible

logger = logging.getLogger(__name__)

# Backward-compatible alias for existing code importing MODEL_REGISTRY.
MODEL_REGISTRY = MODEL_TASK_REGISTRY

def convert(o):
    return to_json_compatible(o)

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
