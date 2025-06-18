#!pythonw
"""
Example program to demonstrate Gooey's presentation of subparsers
"""
import os
import numpy as np
from gooey import Gooey, GooeyParser
from kmcpy.io import _load_occ
from kmcpy.kmc import KMC
from kmcpy.event_generator import generate_events
from kmcpy.model import LocalClusterExpansion
import kmcpy._version
from kmcpy.external.structure import StructureKMCpy

@Gooey(optional_cols=2, program_name="kMCpy GUI", default_size=(1024, 768))
def main():
    settings_msg = "kmcpy version " + kmcpy._version.__version__
    parser = GooeyParser(description=settings_msg)
    parser.add_argument(
        "--verbose",
        help="be verbose",
        dest="verbose",
        action="store_true",
        default=False,
    )

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
    lce_parser.add_argument("mobile_ion_specie_1_identifier", default="Na1")
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
    event_parser.add_argument("mobile_ion_specie_1_identifier", default="Na1")
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
    event_parser.add_argument("event_kernal_fname", default="event_kernal.csv")

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
        "event_kernel", default="./input/event_kernal.csv", type=str
    )
    kmc_parser.add_argument(
        "mc_results", default="./input/initial_state.json", type=str
    )

    # cell
    kmc_parser.add_argument("immutable_sites", default="Zr4+,O2-,O,Zr", type=str)
    kmc_parser.add_argument("supercell_shape", default="2,1,1")
    kmc_parser.add_argument(
        "convert_to_primitive_cell", choices=["yes", "no"], default="yes"
    )

    # simulation condition
    kmc_parser.add_argument("v", default=5000000000000, type=int)
    kmc_parser.add_argument("kmc_pass", default=100, type=int)
    kmc_parser.add_argument("T", default=298, type=int)
    kmc_parser.add_argument("q", default=1.0, type=float)
    kmc_parser.add_argument("elem_hop_distance", default=3.47782, type=float)
    kmc_parser.add_argument("dimension", default=3, type=int)

    # random
    kmc_parser.add_argument("--random_seed", default=12345, type=int)
    kmc_parser.add_argument("--use_numpy_random_kernel", action="store_true")

    # no need to change?
    kmc_parser.add_argument("--structure_idx", default=1, type=int)
    kmc_parser.add_argument("--comp", default=1, type=int)
    kmc_parser.add_argument("verbose", default=True, type=bool)
    kmc_parser.add_argument("equ_pass", default=1, type=int)

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
        args.event_kernal_fname = os.path.join(
            args.events_output_dir, args.event_kernal_fname
        )

        print((vars(args)))

        generate_events(**vars(args))

    if args.command == "KMCSimulation":

        args.supercell_shape = [int(scale) for scale in args.supercell_shape.split(",")]
        args.immutable_sites = [scale for scale in args.immutable_sites.split(",")]
        os.chdir(args.work_dir)
        print(vars(args))

        # workout the sites to be selected
        structure = StructureKMCpy.from_cif(args.prim_fname, primitive=False)

        immutable_sites = []
        for site in structure.sites:
            if site.species in args.immutable_sites:
                immutable_sites.append(site.index)

        occ = _load_occ(
            fname=args.mc_results,
            shape=args.supercell_shape,
            select_sites = immutable_sites,
        )
        kmc = KMC()
        events_initialized = kmc.initialization(occ=occ, **vars(args))
        kmc.load_site_event_list(args.event_kernel)

        # # step 3 run kmc
        kmc.run_from_database(events=events_initialized, **vars(args))

    if args.command == "fitLCEmodel":
        from kmcpy.fitting import Fitting

        os.chdir(args.work_dir)
        y_pred, y_true = Fitting.fit(**vars(args))
        print("fitting", y_pred, y_true)


if __name__ == "__main__":
    main()
