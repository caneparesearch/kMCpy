#!pythonw
import os
import numpy as np

try:
    from gooey import Gooey, GooeyParser
    HAS_GOOEY = True
except ImportError:
    HAS_GOOEY = False
    # Create dummy decorators for when gooey is not available
    def Gooey(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    import argparse
    GooeyParser = argparse.ArgumentParser

from kmcpy.simulator.kmc import KMC
from kmcpy.simulator.config import SimulationConfig
from kmcpy.event.generators import EventGenerator
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
import kmcpy._version

@Gooey(optional_cols=2, program_name="kMCpy GUI", default_size=(1024, 768))
def main():
    """
    Entry point for the kmcpy GUI wrapper CLI.
    
    This function sets up a Gooey-based command-line interface for various kmcpy tasks,
    including local cluster expansion generation, event generation, model fitting, and
    kinetic Monte Carlo (KMC) simulation. It parses user input, processes arguments,
    and dispatches to the appropriate functionality based on the selected subcommand.
    
    Subcommands:
        - LocalClusterExpansion: Generates files required for local cluster expansion.
        - GenerateEvents: Generates events and related files for simulations.
        - fitLCEmodel: Fits the local cluster expansion model using provided data.
        - KMCSimulation: Runs a kinetic Monte Carlo simulation using specified input files.
    
    The function handles argument parsing, input validation, and conversion of GUI-friendly
    arguments to internal representations. It also manages file paths, working directories,
    and invokes the relevant kmcpy modules for each subcommand.
    
    Raises:
        SystemExit: If argument parsing fails or required arguments are missing.
    """
    if not HAS_GOOEY:
        raise ImportError(
            "Gooey is not installed. To use the GUI, install kmcpy with GUI dependencies:\n"
            "pip install kmcpy[gui]"
        )
    
    settings_msg = "kmcpy version " + kmcpy._version.__version__
    parser = GooeyParser(description=settings_msg)

    subs = parser.add_subparsers(help="commands", dest="command")

    # local cluster expansion
    lce_parser = subs.add_parser(
        "LocalClusterExpansion",
        help="generate files required for local cluster expansion",
    )
    lce_parser.add_argument(
        "prim_cif_name",
        help="path to the cif file for generating local cluster expansion",
        type=str,
        widget="FileChooser",
        default="~/Documents/GitHub/kmcPy_dev/test/v3_nasicon_bulk/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",
    )
    lce_parser.add_argument(
        "convert_to_primitive_cell", choices=["yes", "no"], default="yes"
    )
    lce_parser.add_argument(
        "mobile_ion_identifier_type", choices=["label", "specie"], default="label"
    )
    lce_parser.add_argument("mobile_ion_specie_identifier", default="Na1")
    lce_parser.add_argument("mobile_ion_specie_2_identifier", default="Na2")
    lce_parser.add_argument("species_to_be_removed", default="Zr4+,O2-,O,Zr")
    lce_parser.add_argument("cutoff_region", default=4.0)
    lce_parser.add_argument("cutoff_for_point_cluster", default=10, type=int)
    lce_parser.add_argument("cutoff_for_pair_cluster", default=6, type=int)
    lce_parser.add_argument("cutoff_for_triplet_cluster", default=6, type=int)
    lce_parser.add_argument("cutoff_for_quadruplet_cluster", default=0, type=int)
    lce_parser.add_argument("--is_write_basis", action="store_true")
    lce_parser.add_argument(
        "local_cluster_expansion_json",
        default="~/Documents/GitHub/kmcPy_dev/test/v3_nasicon_bulk/lce.json",
    )

    # event generator + local cluster expansion
    event_parser = subs.add_parser("GenerateEvents", help="generate events")
    event_parser.add_argument(
        "prim_cif_name",
        help="path to the cif file for generating local cluster expansion",
        type=str,
        widget="FileChooser",
        default="~/Documents/GitHub/kmcPy_dev/dev/v3_nasicon_bulk",
    )
    event_parser.add_argument(
        "convert_to_primitive_cell", choices=["yes", "no"], default="yes"
    )
    event_parser.add_argument(
        "mobile_ion_identifier_type", choices=["label", "specie"], default="label"
    )
    event_parser.add_argument("mobile_ion_specie_identifier", default="Na1")
    event_parser.add_argument("mobile_ion_specie_2_identifier", default="Na2")
    event_parser.add_argument(
        "local_env_cutoff_dict", default="Na+,Na+,4.0;Na+,Si4+,4.0"
    )
    event_parser.add_argument("species_to_be_removed", default="Zr4+,O2-,O,Zr")
    event_parser.add_argument(
        "events_output_dir",
        default="~/Documents/GitHub/kmcPy_dev/dev/v3_nasicon_bulk",
        widget="DirChooser",
    )
    event_parser.add_argument("supercell_shape", default="2,1,1")
    # event_parser.add_argument("generate_events")

    # output
    event_parser.add_argument(
        "verbosity", choices=["INFO", "WARNING", "CRITICAL"], default="WARNING"
    )
    event_parser.add_argument("--export_local_env_structure", action="store_true")

    # things that shouldn't change at all
    event_parser.add_argument("event_fname", default="events.json")
    event_parser.add_argument("event_dependencies_fname", default="event_dependencies.csv")

    event_parser.add_argument("--distance_matrix_rtol", default=0.01, type=float)
    event_parser.add_argument("--distance_matrix_atol", default=0.01, type=float)
    event_parser.add_argument("--find_nearest_if_fail", default=False)

    fitting_parser = subs.add_parser(
        "fitLCEmodel", help="fit the local cluster expansion model."
    )

    fitting_parser.add_argument("alpha", default=1.5, type=float)
    fitting_parser.add_argument("max_iter", default=1000000, type=int)
    fitting_parser.add_argument("ekra_fname", default="ekra.txt", type=str)
    fitting_parser.add_argument("keci_fname", default="keci.txt", type=str)
    fitting_parser.add_argument("weight_fname", default="weight.txt", type=str)
    fitting_parser.add_argument(
        "corr_fname", default="correlation_matrix.txt", type=str
    )
    fitting_parser.add_argument(
        "fit_results_fname", default="fitting_results.json", type=str
    )
    fitting_parser.add_argument(
        "work_dir", default="~/Documents/GitHub/kmcPy_dev/dev/v3_nasicon_bulk"
    )

    # kmc task
    kmc_parser = subs.add_parser(
        "KMCSimulation", help="generate files required for local cluster expansion"
    )
    # directory and read files
    kmc_parser.add_argument(
        "work_dir", default="~/Documents/GitHub/kmcPy_dev/dev/v3_nasicon_bulk"
    )
    kmc_parser.add_argument(
        "fitting_results", default="./input/fitting_results.json", type=str
    )
    kmc_parser.add_argument(
        "fitting_results_site", default="./input/fitting_results_site.json", type=str
    )
    kmc_parser.add_argument("lce_fname", default="./input/lce.json", type=str)
    kmc_parser.add_argument("lce_site_fname", default="./input/lce_site.json", type=str)
    kmc_parser.add_argument(
        "prim_fname",
        default="./input/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",
        type=str,
    )
    kmc_parser.add_argument("event_fname", default="./input/events.json", type=str)
    kmc_parser.add_argument(
        "event_dependencies", default="./input/event_dependencies.csv", type=str
    )
    kmc_parser.add_argument(
        "initial_state", default="./input/initial_state.json", type=str
    )

    # cell
    kmc_parser.add_argument("immutable_sites", default="Zr4+,O2-,O,Zr", type=str)
    kmc_parser.add_argument("supercell_shape", default="2,1,1")
    kmc_parser.add_argument(
        "convert_to_primitive_cell", choices=["yes", "no"], default="yes"
    )

    # simulation condition
    kmc_parser.add_argument("attempt_frequency", default=5000000000000, type=float)
    kmc_parser.add_argument("kmc_pass", default=100, type=int)
    kmc_parser.add_argument("T", default=298, type=int)
    kmc_parser.add_argument("q", default=1.0, type=float)
    kmc_parser.add_argument("elem_hop_distance", default=3.47782, type=float)
    kmc_parser.add_argument("dimension", default=3, type=int)

    # random
    kmc_parser.add_argument("--random_seed", default=12345, type=int)

    # no need to change?
    kmc_parser.add_argument("--structure_idx", default=1, type=int)
    kmc_parser.add_argument("--comp", default=1, type=int)
    kmc_parser.add_argument("equilibriation_pass", default=1, type=int)

    args = parser.parse_args()

    # print(vars(args))
    if args.convert_to_primitive_cell == "yes":
        args.convert_to_primitive_cell = True
    else:
        args.convert_to_primitive_cell = False

    if args.command == "LocalClusterExpansion":
        args.species_to_be_removed = args.species_to_be_removed.split(",")
        cutoff_cluster = [
            int(args.cutoff_for_pair_cluster),
            int(args.cutoff_for_triplet_cluster),
            int(args.cutoff_for_quadruplet_cluster),
        ]
        print((vars(args)))
        a = LocalClusterExpansion()
        a.initialization(
            cutoff_cluster=cutoff_cluster,
            template_cif_fname=args.prim_cif_name,
            **vars(args)
        )
        a.to_json(args.local_cluster_expansion_json)

    if args.command == "GenerateEvents":
        np.set_printoptions(precision=2)
        args.species_to_be_removed = args.species_to_be_removed.split(",")
        tmp_local_env_cutoff_dict = {}

        for cutoff in args.local_env_cutoff_dict.split(";"):
            cutoff = cutoff.split(",")
            tmp_local_env_cutoff_dict[(str(cutoff[0]), str(cutoff[1]))] = float(
                cutoff[2]
            )

        args.local_env_cutoff_dict = tmp_local_env_cutoff_dict

        # supercell
        args.supercell_shape = [int(scale) for scale in args.supercell_shape.split(",")]

        # fanme_path
        args.event_fname = os.path.join(args.events_output_dir, args.event_fname)
        args.event_dependencies_fname = os.path.join(
            args.events_output_dir, args.event_dependencies_fname
        )

        if isinstance(args.find_nearest_if_fail, str):
            args.find_nearest_if_fail = args.find_nearest_if_fail.lower() in {
                "1",
                "true",
                "yes",
                "y",
            }

        print((vars(args)))

        generator = EventGenerator()
        generator.generate_events(
            structure_file=args.prim_cif_name,
            convert_to_primitive_cell=args.convert_to_primitive_cell,
            local_env_cutoff_dict=args.local_env_cutoff_dict,
            mobile_ion_identifier_type=args.mobile_ion_identifier_type,
            mobile_ion_identifiers=(
                args.mobile_ion_specie_identifier,
                args.mobile_ion_specie_2_identifier,
            ),
            species_to_be_removed=args.species_to_be_removed,
            distance_matrix_rtol=args.distance_matrix_rtol,
            distance_matrix_atol=args.distance_matrix_atol,
            find_nearest_if_fail=args.find_nearest_if_fail,
            export_local_env_structure=args.export_local_env_structure,
            supercell_shape=args.supercell_shape,
            event_file=args.event_fname,
            event_dependencies_file=args.event_dependencies_fname,
        )

    if args.command == "KMCSimulation":

        args.supercell_shape = [int(scale) for scale in args.supercell_shape.split(",")]
        args.immutable_sites = [scale for scale in args.immutable_sites.split(",")]
        os.chdir(args.work_dir)
        print(vars(args))

        # Convert GUI args to SimulationConfig using modern parameter names.
        legacy_mapping = {
            "fitting_results": "fitting_results_file",
            "fitting_results_site": "fitting_results_site_file",
            "lce_fname": "cluster_expansion_file",
            "lce_site_fname": "cluster_expansion_site_file",
            "prim_fname": "structure_file",
            "event_fname": "event_file",
            "initial_state": "initial_state_file",
            "kmc_pass": "kmc_passes",
            "T": "temperature",
            "q": "mobile_ion_charge",
            "elem_hop_distance": "elementary_hop_distance",
            "equilibriation_pass": "equilibration_passes",
        }
        valid_config_keys = {
            "structure_file",
            "supercell_shape",
            "dimension",
            "mobile_ion_specie",
            "mobile_ion_charge",
            "elementary_hop_distance",
            "model_type",
            "cluster_expansion_file",
            "cluster_expansion_site_file",
            "fitting_results_file",
            "fitting_results_site_file",
            "event_file",
            "event_dependencies",
            "immutable_sites",
            "convert_to_primitive_cell",
            "initial_state_file",
            "initial_occupations",
            "temperature",
            "attempt_frequency",
            "equilibration_passes",
            "kmc_passes",
            "random_seed",
            "name",
        }

        config_params = {}
        for key, value in vars(args).items():
            mapped_key = legacy_mapping.get(key, key)
            if mapped_key in valid_config_keys and value is not None:
                config_params[mapped_key] = value

        config = SimulationConfig(**config_params)
        kmc = KMC.from_config(config)

        # run kmc
        kmc.run(config)

    if args.command == "fitLCEmodel":
        from kmcpy.models.local_cluster_expansion import LocalClusterExpansion

        os.chdir(args.work_dir)
        _, y_pred, y_true = LocalClusterExpansion().fit(
            alpha=args.alpha,
            max_iter=args.max_iter,
            ekra_fname=args.ekra_fname,
            keci_fname=args.keci_fname,
            weight_fname=args.weight_fname,
            corr_fname=args.corr_fname,
            fit_results_fname=args.fit_results_fname,
            lce_params_fname=None,
            lce_params_history_fname=None,
        )
        print("fitting", y_pred, y_true)


if __name__ == "__main__":
    main()
