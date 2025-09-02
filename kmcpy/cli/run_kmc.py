#!/usr/bin/env python

import warnings
from kmcpy.io.config_io import SimulationConfigIO  # Use modern IO system
from kmcpy.simulator.config import SimulationConfig
from kmcpy.simulator.kmc import KMC
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
        --fitting_results_file (str): Path to the JSON file containing the fitting results for E_kra.
            Required if input file is not provided.
        --fitting_results_site_file (str): Path to the JSON file containing the fitting results for site energy difference.
            Required if input file is not provided.
        --cluster_expansion_file (str): Path to the JSON file containing the Local Cluster Expansion (LCE) model.
            Required if input file is not provided.
        --cluster_expansion_site_file (str): Path to the JSON file containing the site LCE model for computing site energy differences.
            Required if input file is not provided.
        --structure_file (str): Path to the CIF file of the template structure (with all sites filled).
            Required if input file is not provided.
        --event_file (str): Path to the JSON file containing the list of events.
            Required if input file is not provided.
        --attempt_frequency (float, optional): Attempt frequency (prefactor) for hopping events. Defaults to 1e13 Hz.
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
    # Always show all arguments in help - using modern parameter names
    parser.add_argument("--supercell_shape", type=str, help='Shape of the supercell as a list of integers (e.g., [2, 2, 2]). This should be consistent with events.')
    parser.add_argument("--fitting_results_file", type=str, help='Path to the JSON file containing the fitting results for E_kra.')
    parser.add_argument("--fitting_results_site_file", type=str, help='Path to the JSON file containing the fitting results for site energy difference.')
    parser.add_argument("--cluster_expansion_file", type=str, help='Path to the JSON file containing the Local Cluster Expansion (LCE) model.')
    parser.add_argument("--cluster_expansion_site_file", type=str, help='Path to the JSON file containing the site LCE model for computing site energy differences.')
    parser.add_argument("--structure_file", type=str, help='Path to the CIF file of the template structure (with all sites filled).')
    parser.add_argument("--event_file", type=str, help='Path to the JSON file containing the list of events.')
    parser.add_argument("--attempt_frequency", type=float, default=1e13, help='Attempt frequency (prefactor) for hopping events. Defaults to 1e13 Hz.')
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
    config = None
    
    print("Starting KMC simulation...")
    
    if args.input:
        # Load modern SimulationConfig format only
        try:
            print(f"Loading configuration from {args.input}")
            from kmcpy.io.config_io import SimulationConfigIO
            raw_data = SimulationConfigIO._load_yaml_section(args.input, "kmc", "default")
            config = SimulationConfig.from_dict(raw_data)
            print(f"✓ Configuration loaded: {config.runtime_config.name}")
        except Exception as e:
            # Provide clear error message for legacy formats
            raise ValueError(
                f"Unable to load configuration from {args.input}. "
                f"Legacy InputSet format is no longer supported. "
                f"Please convert your configuration to the modern SimulationConfig format. "
                f"Use SimulationConfigIO to create new configuration files. "
                f"Original error: {e}"
            )
    else:
        # Build a dictionary from the argparse Namespace, excluding None values and 'input'
        input_dict = {k: v for k, v in vars(args).items() if k != "input" and v is not None}
        config = SimulationConfig(**input_dict)

    print("Configuration loaded, initializing KMC...")
    print(f"  Structure file: {config.system_config.structure_file}")
    print(f"  Temperature: {config.runtime_config.temperature} K")
    
    # initialize global occupation and conditions using modern API
    kmc = KMC.from_config(config)
    print("KMC initialized, starting simulation...")

    # run kmc with config
    tracker = kmc.run(config)
    
    # Optionally save results
    print("KMC simulation completed successfully!")


if __name__ == "__main__":
    main()
