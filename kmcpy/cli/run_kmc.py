#!/usr/bin/env python

from kmcpy.simulator.config import Configuration
from kmcpy.simulator.kmc import KMC
import kmcpy
import argparse
import ast


def _parse_sequence(value):
    if value is None or isinstance(value, (list, tuple)):
        return value
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = [item.strip() for item in str(value).split(",") if item.strip()]
    if not isinstance(parsed, (list, tuple)):
        raise argparse.ArgumentTypeError("value must be a list or tuple")
    return parsed


def _parse_supercell_shape(value):
    parsed = _parse_sequence(value)
    if parsed is None:
        return None
    try:
        supercell_shape = tuple(int(item) for item in parsed)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "supercell_shape must contain integers"
        ) from exc
    if len(supercell_shape) != 3:
        raise argparse.ArgumentTypeError("supercell_shape must contain 3 integers")
    return supercell_shape


def _parse_mapping(value):
    if value is None or isinstance(value, dict):
        return value
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be a dictionary") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a dictionary")
    return parsed



def main()->None:
    """
    Entry point for the kMCpy command-line interface to run kinetic Monte Carlo (kMC) simulations.
    
    This function parses command-line arguments for running a kMC simulation using kMCpy. It supports
    two modes of input:
    
    1. Providing a single JSON/YAML file containing all simulation parameters.
    2. Providing individual arguments for each required parameter.
    
    If a JSON/YAML input file is provided, all other parameters are read from this file. Otherwise, the user
    must specify all required arguments individually.
    
    Args:
        input (str, optional): Path to the input JSON/YAML file for kMC simulation. If provided, all other
            parameters are read from this file.
        supercell_shape (str): Shape of the supercell as a list of integers (e.g., [2, 2, 2]).
            Required if input file is not provided.
        model_file (str): Path to model JSON file.
            Files written by model.to(...) include class metadata.
            Required if input file is not provided.
        structure_file (str): Path to the CIF file of the template structure (with all sites filled).
            Required if input file is not provided.
        event_file (str): Path to the JSON file containing the list of events.
            Required if input file is not provided.
        attempt_frequency (float, optional): Attempt frequency (prefactor) for hopping events. Defaults to 1e13 Hz.
        temperature (float, optional): Simulation temperature in Kelvin. Defaults to 300 K.
        convert_to_primitive_cell (bool, optional): Whether to convert the structure to its primitive cell.
            Defaults to False.
    
    Returns:
        None
    """

    parser = argparse.ArgumentParser(
        description=kmcpy.get_logo(),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=str,
        help=(
            "Path to the input JSON/YAML file for kMC simulation. If provided, "
            "all other parameters are read from this file."
        ),
    )
    # Always show all arguments in help - using modern parameter names
    parser.add_argument(
        "--supercell_shape",
        type=_parse_supercell_shape,
        help=(
            "Shape of the supercell as a list of integers (e.g., [2, 2, 2]). "
            "This should be consistent with events."
        ),
    )
    parser.add_argument(
        "--model_file",
        type=str,
        help="Path to model JSON file.",
    )
    parser.add_argument(
        "--structure_file",
        type=str,
        help="Path to the CIF file of the template structure (with all sites filled).",
    )
    parser.add_argument(
        "--event_file",
        type=str,
        help="Path to the JSON file containing the list of events.",
    )
    parser.add_argument(
        "--site_mapping",
        type=_parse_mapping,
        help=(
            "Site mapping that defines active and fixed sites "
            "(for example: {'Na': ['Na', 'X'], 'O': 'O'})."
        ),
    )
    parser.add_argument(
        "--attempt_frequency",
        type=float,
        default=1e13,
        help="Attempt frequency (prefactor) for hopping events. Defaults to 1e13 Hz.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300,
        help="Simulation temperature in Kelvin. Defaults to 300 K.",
    )
    parser.add_argument(
        "--convert_to_primitive_cell",
        action="store_true",
        help="Whether to convert the structure to its primitive cell (default: False).",
    )
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
        # Load modern Configuration format only
        try:
            print(f"Loading configuration from {args.input}")
            config = Configuration.from_file(args.input)
            print(f"✓ Configuration loaded: {config.runtime_config.name}")
        except Exception as e:
            # Provide clear error message for legacy formats
            raise ValueError(
                f"Unable to load configuration from {args.input}. "
                f"Legacy InputSet format is no longer supported. "
                f"Please convert your configuration to the modern Configuration format. "
                f"Use `kmcpy init` to create a new configuration file. "
                f"Original error: {e}"
            )
    else:
        # Build a dictionary from the argparse Namespace, excluding None values and 'input'
        input_dict = {k: v for k, v in vars(args).items() if k != "input" and v is not None}
        if "supercell_shape" in input_dict:
            input_dict["supercell_shape"] = _parse_supercell_shape(
                input_dict["supercell_shape"]
            )
        if "site_mapping" in input_dict:
            input_dict["site_mapping"] = _parse_mapping(
                input_dict["site_mapping"]
            )
        config = Configuration(**input_dict)

    print("Configuration loaded, initializing KMC...")
    print(f"  Structure file: {config.system_config.structure_file}")
    print(f"  Model file: {config.system_config.model_file}")
    print(f"  Temperature: {config.runtime_config.temperature} K")
    
    # initialize global occupation and conditions using modern API
    kmc = KMC.from_config(config)
    print("KMC initialized, starting simulation...")

    # run kmc
    tracker = kmc.run()
    
    # Optionally save results
    print("KMC simulation completed successfully!")


if __name__ == "__main__":
    main()
