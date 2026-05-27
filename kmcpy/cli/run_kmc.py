#!/usr/bin/env python

from kmcpy.simulator.config import Configuration
from kmcpy.simulator.kmc import KMC
import kmcpy
import argparse
import ast


RUN_HELP_EPILOG = """
Recommended workflow:
  kmcpy init --output input_template.yaml
  # or: kmcpy sample all --output-dir kmcpy_sample
  run_kmc --input input_template.yaml

The input-file workflow is preferred for research runs because it records the
full Configuration in one place. Direct flags are kept for quick checks and
simple scripts. For less common fields such as property callbacks and built-in
property schedules, use a YAML or JSON input file.
"""


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


def _parse_int_sequence(value):
    parsed = _parse_sequence(value)
    if parsed is None:
        return None
    try:
        return [int(item) for item in parsed]
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "initial_occupations must contain integers"
        ) from exc


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add run arguments to ``parser`` and return it."""
    parser.add_argument(
        "--input",
        type=str,
        help=(
            "Preferred. Path to a modern Configuration YAML/JSON input file. "
            "Generate one with `kmcpy init` or `kmcpy sample all`."
        ),
    )

    system_group = parser.add_argument_group("common system fields")
    system_group.add_argument(
        "--supercell_shape",
        type=_parse_supercell_shape,
        help=(
            "Supercell replication factors [a, b, c], e.g. '[2, 2, 2]'. "
            "Must match the event library."
        ),
    )
    system_group.add_argument(
        "--model_file",
        type=str,
        help="Path to model JSON file.",
    )
    system_group.add_argument(
        "--structure_file",
        type=str,
        help="Path to the structure/CIF file containing all possible mobile-ion sites.",
    )
    system_group.add_argument(
        "--event_file",
        type=str,
        help="Path to the event library JSON file.",
    )
    system_group.add_argument(
        "--initial_state_file",
        type=str,
        help="Path to an initial State JSON file.",
    )
    system_group.add_argument(
        "--initial_occupations",
        type=_parse_int_sequence,
        help="Initial active-site occupation vector, e.g. '[0, 1, 0]' or '0,1,0'.",
    )
    system_group.add_argument(
        "--site_mapping",
        type=_parse_mapping,
        help=(
            "Site mapping that defines active and fixed sites "
            "(for example: {'Na': ['Na', 'X'], 'O': 'O'})."
        ),
    )
    system_group.add_argument(
        "--model_type",
        type=str,
        help="Model type for direct payloads, e.g. local_barrier or composite_lce.",
    )
    system_group.add_argument(
        "--dimension",
        type=int,
        choices=[1, 2, 3],
        help="Transport dimensionality.",
    )
    system_group.add_argument(
        "--mobile_ion_specie",
        type=str,
        help="Mobile-ion species label, e.g. Li or Na.",
    )
    system_group.add_argument(
        "--mobile_ion_charge",
        type=float,
        help="Mobile-ion charge in |e|.",
    )
    system_group.add_argument(
        "--elementary_hop_distance",
        type=float,
        help="Characteristic hop distance in Angstrom.",
    )
    system_group.add_argument(
        "--convert_to_primitive_cell",
        action="store_true",
        help="Convert the structure to its primitive cell before setup.",
    )

    runtime_group = parser.add_argument_group("common runtime fields")
    runtime_group.add_argument(
        "--attempt_frequency",
        type=float,
        help="Attempt frequency in Hz.",
    )
    runtime_group.add_argument(
        "--temperature",
        type=float,
        help="Simulation temperature in Kelvin.",
    )
    runtime_group.add_argument(
        "--equilibration_passes",
        type=int,
        help="Number of equilibration KMC passes.",
    )
    runtime_group.add_argument(
        "--kmc_passes",
        type=int,
        help="Number of production KMC passes.",
    )
    runtime_group.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for reproducible event selection.",
    )
    runtime_group.add_argument(
        "--name",
        type=str,
        help="Simulation label used in output filenames.",
    )
    return parser


def build_parser(prog: str = "run_kmc") -> argparse.ArgumentParser:
    """Build the parser shared by ``run_kmc`` and ``kmcpy run``."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=kmcpy.get_logo(),
        epilog=RUN_HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return configure_parser(parser)


def main(argv=None) -> None:
    """Entry point for running kinetic Monte Carlo simulations."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run_kmc(args)


def run_kmc(args) -> None:
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
        ignored_keys = {"command", "input"}
        input_dict = {
            k: v
            for k, v in vars(args).items()
            if k not in ignored_keys and v is not None
        }
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
