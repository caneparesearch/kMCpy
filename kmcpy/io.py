"""
IO takes dictionary like object and convert them into json writable string

Author: Zeyu Deng
Email: dengzeyu@gmail.com
"""

import numpy as np
import json
from kmcpy.external.structure import StructureKMCpy
import logging

logger = logging.getLogger(__name__) 

# class IO:
#     def __init__(self):
#         pass

#     def to_json(self,fname):
#         print('Saving:',fname)
#         with open(fname,'w') as fhandle:
#             d = self.as_dict()
#             jsonStr = json.dumps(d,indent=4,default=convert) # to get rid of errors of int64
#             fhandle.write(jsonStr)


def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    elif isinstance(o, np.int32):
        return int(o)
    raise TypeError


def _load_occ(
    fname: str,
    shape: list,
    select_sites: list,
    **kwargs
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


# to be developed


class InputSet:
    """
    a flexible input set class for running KMC
    just a dictionary
    """

    def __init__(self, _parameters={}) -> None:

        self._parameters = _parameters
        pass

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
        return cls(_parameters)

    def get_parameters_str(self, format="equation"):
        """
        return the parameters of this input set.

        Args:
            format (str, optional): "equation" or "dict". If format=dict, then return a python dict. 
            format=equation: return equations that is capable for kwargs. Defaults to "equation".
        Returns:
            str or dict: The parameter string or dictionary.
        """
        if format == "dict":
            return self._parameters.__str__()
        if format == "equation":
            equation_strs = []
            for i in self._parameters:
                if isinstance(self._parameters[i], str):
                    equation_strs.append(f"{i}='{self._parameters[i]}'")
                elif isinstance(self._parameters[i], np.ndarray):
                    equation_strs.append(f"{i}={self._parameters[i].tolist()}")
                else:
                    equation_strs.append(f"{i}={self._parameters[i]}")
            return ",".join(equation_strs)

    def set_parameter(self, key_to_change="T", value_to_change=273.15):
        """_summary_

        Args:
            key_to_change (str, optional): the key to change in the parameters. Defaults to "T".
            value_to_change (any, optional): any type that json can read. Defaults to 273.15.
        """
        self._parameters[key_to_change] = value_to_change

    def enumerate(self, **kwargs):
        """generate a new InputSet from the input kwargs

        Inputs:
            for example: InputSet.enumerate(T=298.15)

        Returns:
            InputSet: a InputSet class with modified parameters
        """

        new_InputSet = InputSet(self._parameters)
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
        """a rough parameter checker to make sure that there is enough parameters to run a job

        Raises:
            ValueError: In case that parameter is not defined in the self._parameters
        """
        for i in [
                "v",
                "equ_pass",
                "kmc_pass",
                "supercell_shape",
                "fitting_results",
                "fitting_results_site",
                "lce_fname",
                "lce_site_fname",
                "prim_fname",
                "event_fname",
                "event_kernel",
                "mc_results",
                "T",
                "comp",
                "dimension",
                "q",
            ]:
                if i not in self._parameters:
                    logger.error(f"{i} is not defined yet in the parameters!")
                    raise ValueError(
                        "This program is exploding due to undefined parameter. Please check input json file"
                    )

    def load_occ(self):
        """load the occupation data from the input json file
        """
        
        # workout the sites to be selected
        self.structure = StructureKMCpy.from_cif(
            self._parameters['prim_fname'], primitive=self._parameters['convert_to_primitive_cell']
        )

        immutable_sites_idx = []
        for index,site in enumerate(self.structure.sites):
            if site.specie.symbol not in self._parameters['immutable_sites']:
                immutable_sites_idx.append(index)
        logger.debug(f"Immutable sites are {immutable_sites_idx}")

        self._parameters['occ'] = _load_occ(
            self._parameters["mc_results"],
            self._parameters["supercell_shape"],
            select_sites=immutable_sites_idx,
        )