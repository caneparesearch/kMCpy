#!/usr/bin/env python

from kmcpy.io import InputSet
from kmcpy.kmc import KMC
import kmcpy
import argparse

def main()->None:
    """
    Entry point for the kMCpy command-line interface to run kinetic Monte Carlo (kMC) simulations.
    This function parses command-line arguments for running a kMC simulation using kMCpy. It supports
    two modes of input:
      1. Providing a single JSON/YAML file containing all simulation parameters.
      2. Providing individual arguments for each required parameter.
    If a JSON/YAML input file is provided, all other parameters are read from this file. Otherwise, the user
    must specify all required arguments individually.
    Command-line Arguments:
        input (str, optional): Path to the input JSON/YAML file for kMC simulation. If provided, all other
            parameters are read from this file.
        --supercell_shape (str): Shape of the supercell as a list of integers (e.g., [2, 2, 2]).
            Required if input file is not provided.
        --fitting_results (str): Path to the JSON file containing the fitting results for E_kra.
            Required if input file is not provided.
        --fitting_results_site (str): Path to the JSON file containing the fitting results for site energy difference.
            Required if input file is not provided.
        --lce_fname (str): Path to the JSON file containing the Local Cluster Expansion (LCE) model.
            Required if input file is not provided.
        --lce_site_fname (str): Path to the JSON file containing the site LCE model for computing site energy differences.
            Required if input file is not provided.
        --template_structure_fname (str): Path to the CIF file of the template structure (with all sites filled).
            Required if input file is not provided.
        --event_fname (str): Path to the JSON file containing the list of events.
            Required if input file is not provided.
        --event_kernel (str): Path to the event kernel file.
            Required if input file is not provided.
        -v, --v (float, optional): Attempt frequency (prefactor) for hopping events. Defaults to 1e13 Hz.
        --temperature (float, optional): Simulation temperature in Kelvin. Defaults to 300 K.
        --convert_to_primitive_cell (bool, optional): Whether to convert the structure to its primitive cell.
            Defaults to False.
        --immutable_sites (str, optional): List of sites to be treated as immutable and removed from the simulation
            (as a JSON string, default: []).
    Returns:
        None
    """

    parser = argparse.ArgumentParser(
        description=kmcpy.get_logo(),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=str, help="Path to the input JSON/YAML file for kMC simulation. If provided, all other parameters are read from this file.")
    # Always show all arguments in help
    parser.add_argument("--supercell_shape", type=str, help='Shape of the supercell as a list of integers (e.g., [2, 2, 2]). This should be consistent with events.')
    parser.add_argument("--fitting_results", type=str, help='Path to the JSON file containing the fitting results for E_kra.')
    parser.add_argument("--fitting_results_site", type=str, help='Path to the JSON file containing the fitting results for site energy difference.')
    parser.add_argument("--lce_fname", type=str, help='Path to the JSON file containing the Local Cluster Expansion (LCE) model.')
    parser.add_argument("--lce_site_fname", type=str, help='Path to the JSON file containing the site LCE model for computing site energy differences.')
    parser.add_argument("--template_structure_fname", type=str, help='Path to the CIF file of the template structure (with all sites filled).')
    parser.add_argument("--event_fname", type=str, help='Path to the JSON file containing the list of events.')
    parser.add_argument("--event_kernel", type=str, help='Path to the event kernel file.')
    parser.add_argument("-v", "--v", type=float, default=1e13, help='Attempt frequency (prefactor) for hopping events. Defaults to 1e13 Hz.')
    parser.add_argument("--temperature", type=float, default=300, help='Simulation temperature in Kelvin. Defaults to 300 K.')
    parser.add_argument("--convert_to_primitive_cell", action='store_true', help='Whether to convert the structure to its primitive cell (default: False).')
    parser.add_argument("--immutable_sites", type=str, default="[]", help='List of sites to be treated as immutable and removed from the simulation (as a JSON string, default: []).')
    args = parser.parse_args()
    run_kmc(args)

def run_kmc(args)-> None:
    """
    Runs the kinetic Monte Carlo (KMC) simulation based on the provided arguments.

    This function initializes the simulation input either from a JSON/YAML file or directly from
    the command-line arguments, constructs the KMC simulation object, and executes the simulation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments. If 'input' is provided,
            it should be a path to a JSON/YAML file containing the simulation parameters.
            Otherwise, parameters are taken directly from the other attributes of 'args'.

    Returns:
        None
    """

    if args.input:
        if args.input.endswith(".yaml") or args.input.endswith(".yml"):
            inputset = InputSet.from_yaml(args.input)
        else:
            inputset = InputSet.from_json(args.input)
    else:
        # Build a dictionary from the argparse Namespace, excluding None values and 'input'
        input_dict = {k: v for k, v in vars(args).items() if k != "input" and v is not None}
        inputset = InputSet.from_dict(input_dict)

    # initialize global occupation and conditions
    kmc = KMC.from_inputset(inputset = inputset)

    # run kmc
    kmc.run(inputset = inputset)


if __name__ == "__main__":
    main()
