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


def _load_occ(
    fname: str,
    shape: list,
    select_sites: list
):
    """load occupation data

    Args:
        fname (str): initial occupation (json format) that also includes immutable site (for example, Zr, O in NASICON). E.g.: "./initial_state.json".
        shape (list): supercell shape. E.g.: [2,1,1].
        select_sites (list): all the sites included in kinetic monte carlo process, i.e., this is the list include only the indices of Na, Si, P (no Zr and O) in the Na1+xZr2P3-xSixO12. E.g.: [0,1,2,3,4,5,6,7,32,33,34,35,36,37].

    Raises:
        ValueError:

    Returns:
        chebyshev occupation: list of 1 and -1 states, the initial occupation data of sites included in KMC, for example, Na, Si, P initial states in NZSP
    """
    with open(fname, "r") as f:

        # read the occupation from json
        occupation_raw_data = np.array(json.load(f)["occupation"])

        # check if the occupation is compatible with the shape.
        # for example. if there is 20 occupation data and supercell is [3,1,1], it is incompatible because 20/3 is not integer
        if len(occupation_raw_data) % (shape[0] * shape[1] * shape[2]) != 0:
            logger.error(
                f"The length of occupation data {len(occupation_raw_data)} is incompatible with the supercell shape, please check the {fname} "
            )
            raise ValueError(
                f"The length of occupation data {len(occupation_raw_data)} is incompatible with the supercell shape,please check the {fname} "
            )

        # this is the total sites
        site_nums = int(len(occupation_raw_data) / (shape[0] * shape[1] * shape[2]))

        # this is the dimension of global occupation array
        convert_to_dimension = (site_nums, shape[0], shape[1], shape[2])

        occupation = occupation_raw_data.reshape(convert_to_dimension)[
            select_sites
        ].flatten(
            "C"
        )  # the global occupation array in the format of (site,x,y,z). Now it only contain the selected sites.

        occupation_chebyshev = np.where(
            occupation == 0, -1, occupation
        )  # replace 0 with -1 for Chebyshev basis

        logger.debug(f"Selected sites are {select_sites}")
        logger.debug(
            f"Converting the occupation raw data to dimension: {convert_to_dimension}"
        )
        logger.debug(f"Occupation (chebyshev basis after removing immutable sites): {occupation_chebyshev}")

        return occupation_chebyshev


class Results:
    """
    Class to store and manipulate results from the Tracker.
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