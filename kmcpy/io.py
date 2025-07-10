#!/usr/bin/env python
"""
This module provides the InputSet class for reading and managing input parameters for KMC simulations.
"""

import numpy as np
import json
from kmcpy.external.structure import StructureKMCpy
import logging
import pandas as pd
import warnings

logger = logging.getLogger(__name__) 

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

class InputSet:
    """
    a flexible input set class for running KMC
    just a dictionary
    """
    def __init__(self, _parameters={}) -> None:
        # Handle parameter name transitions for backward compatibility
        if 'event_kernel' in _parameters and 'event_dependencies' not in _parameters:
            # Map old parameter name to new one
            _parameters = _parameters.copy()  # Don't modify the original
            _parameters['event_dependencies'] = _parameters['event_kernel']
            logger.warning("Parameter 'event_kernel' is deprecated. Use 'event_dependencies' instead.")
        
        self._parameters = _parameters

    @classmethod
    def from_json(
        cls, input_json_path=r"examples/test_input.json", 
    ):
        """
        input_reader takes input (a json file with all parameters as shown in run_kmc.py in examples folder)
        return a dictionary with all input parameters
        """
        _parameters = json.load(open(input_json_path))
        logger.debug(_parameters)

        input_set = cls(_parameters)

        # check parameters
        input_set.parameter_checker()

        # load occupation data
        input_set.load_occ()
        return input_set

    @classmethod
    def from_yaml(
        cls, input_yaml_path=r"examples/test_input.yaml",
    ):
        """
        input_reader takes input (a yaml file with all parameters as shown in run_kmc.py in examples folder)
        return a dictionary with all input parameters
        """
        import yaml

        _parameters = yaml.safe_load(open(input_yaml_path))
        logger.debug(_parameters)

        input_set = cls(_parameters)

        # check parameters
        input_set.parameter_checker()

        # load occupation data
        input_set.load_occ()
        return input_set
    
    @classmethod
    def from_dict(cls, input_dict):
        """
        input_reader takes input (a dictionary with all parameters as shown in run_kmc.py in examples folder)
        return a dictionary with all input parameters
        """
        # print the input_dict for debugging
        logger.debug(input_dict)

        input_set = cls(input_dict)

        # check parameters
        input_set.parameter_checker()

        # load occupation data only if template structure file exists
        import os
        template_file = input_set._parameters.get('template_structure_fname', '')
        if template_file and os.path.exists(template_file):
            input_set.load_occ()
        elif template_file:
            # File specified but doesn't exist - this is an error
            input_set.load_occ()  # This will raise FileNotFoundError
        else:
            # No file specified - set minimal structure attributes for testing
            logger.warning("No template structure file specified")
            input_set.structure = None
            input_set.occupation = []
            input_set.n_sites = 0
            
        return input_set

    def __str__(self):
        return self._parameters.__str__()

    def set_parameter(self, key, value):
        """set a parameter in the input set"""
        if key in self._parameters:
            self._parameters[key] = value
        else:
            logger.error(f"{key} is not a valid parameter in the InputSet")
            raise KeyError(f"{key} is not a valid parameter in the InputSet")
        
    def enumerate(self, **kwargs):
        """generate a new InputSet from the input kwargs

        Inputs:
            for example: InputSet.enumerate(T=298.15)

        Returns:
            InputSet: a InputSet class with modified parameters
        """

        new_InputSet = InputSet(self._parameters.copy())
        for key_to_change in kwargs:
            new_InputSet.set_parameter(key_to_change, kwargs[key_to_change])
        return new_InputSet

    def change_key_name(self, oldname="lce", newname="lce_fname"):
        """change the key name from old name to new name for self._parameters

        Args:
            oldname (str): defined name in self._parameters
            newname (str): new name
        """
        self._parameters[newname] = self._parameters[oldname]


    def parameter_checker(self):
        """
        Checks all keys in self._parameters against valid_params, case-insensitively.

        Returns:
            dict: {parameter: True/False} for each parameter in self._parameters.
        """
        available_tasks = ["kmc", "lce","generate_event"]
        # Convert all keys in self._parameters to lower case
        self._parameters = {k.lower(): v for k, v in self._parameters.items()}

        if self._parameters["task"] == "kmc":
            # True -> positional parameters, False -> optional parameters
            parameters = {
                "task": True,
                "v": True,
                "equ_pass": True,
                "kmc_pass": True,
                "supercell_shape": True,
                "fitting_results": True,
                "fitting_results_site": True,
                "lce_fname": True,
                "lce_site_fname": True,
                "template_structure_fname": True,
                "event_fname": True,
                "event_dependencies": True,  # New parameter name
                "event_kernel": False,  # Old parameter name (optional for backward compatibility)
                "initial_state": True,
                "temperature": True,
                "dimension": True,
                "q": True,
                "elem_hop_distance": True,
                "convert_to_primitive_cell": False,
                "immutable_sites": False,
                "mobile_ion_specie": True,
                "random_seed": False,  # Optional parameter for random number generation
                "name": False  # Optional parameter for simulation name
            }
        elif self._parameters["task"] == "lce":
            parameters = {
                "task": True,
                "center_frac_coord": True,
                "mobile_ion_identifier_type": True,
                "mobile_ion_specie_1_identifier": True,
                "cutoff_cluster": True,
                "cutoff_region": True,
                "template_structure_fname": True,
                "is_write_basis": True,
                "species_to_be_removed": True,
                "convert_to_primitive_cell": False,
                "exclude_site_with_identifier": True,
            }
        elif self._parameters["task"] == "generate_event":
            parameters = {
                "task": True,
                "template_structure_fname": True,
                "convert_to_primitive_cell": False,
                "local_env_cutoff_dict": True,
                "mobile_ion_identifier_type": True,
                "mobile_ion_specie_1_identifier": True,
                "mobile_ion_specie_2_identifier": True,
                "species_to_be_removed": True,
                "distance_matrix_rtol": False,
                "distance_matrix_atol": False,
                "find_nearest_if_fail": False,
                "export_local_env_structure": False,
                "supercell_shape": True,
                "event_fname": True,
                "event_dependencies_fname": True,
            }
        else:
            raise ValueError(f"Unknown task {self._parameters['task']}. Please set task to {available_tasks}.")
        valid_params_lower = {param.lower() for param in parameters}
        positional_params = {param.lower() for param, required in parameters.items() if required}
        result = {}
        missing_params = []
        for key in self._parameters:
            result[key] = key.lower() in valid_params_lower

        # Check for missing positional parameters
        for param in positional_params:
            if param not in self._parameters:
                missing_params.append(param)
        if missing_params:
            logger.error(f"Missing required parameters: {missing_params} for task {self._parameters['task']}")
            raise ValueError(f"Missing required parameters: {missing_params} for task {self._parameters['task']}")

        # Warn about ignored parameters
        ignored_params = [key for key, valid in result.items() if not valid]
        if ignored_params:
            warnings.warn(f"Ignored parameters: {ignored_params} for task {self._parameters['task']}", UserWarning)

    def __getattr__(self, name):
        # Handle backward compatibility for parameter names
        if name == 'event_kernel' and 'event_kernel' not in self._parameters:
            # Map old parameter name to new one
            if 'event_dependencies' in self._parameters:
                return self._parameters['event_dependencies']
        
        try:
            return self._parameters[name]
        except KeyError:
            raise AttributeError(f"'InputSet' object has no attribute '{name}'")
        
    def load_occ(self):
        """load the occupation data from the input json file
        """
        
        # workout the sites to be selected
        self.structure = StructureKMCpy.from_cif(
            self._parameters['template_structure_fname'], primitive=self._parameters['convert_to_primitive_cell']
        )

        immutable_sites_idx = []
        for index,site in enumerate(self.structure.sites):
            if site.specie.symbol not in self._parameters['immutable_sites']:
                immutable_sites_idx.append(index)
        logger.debug(f"Immutable sites are {immutable_sites_idx}")

        self._parameters['initial_occ'] = _load_occ(
            self._parameters["initial_state"],
            self._parameters["supercell_shape"],
            select_sites=immutable_sites_idx,
        )

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