#!/usr/bin/env python3
"""Run a minimal end-to-end NASICON tutorial simulation using bundled example data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from kmcpy.event import EventGenerator
from kmcpy.simulator.config import Configuration
from kmcpy.simulator.kmc import KMC


NASICON_SITE_MAPPING = {"Na": ["Na", "X"], "Zr": "Zr", "Si": ["Si", "P"], "O": "O"}


def parse_supercell_shape(value: str) -> tuple[int, int, int]:
    """Parse a comma-separated supercell string (for example: 2,1,1)."""
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "supercell-shape must contain exactly 3 comma-separated integers"
        )
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("supercell-shape must contain integers") from exc

    if any(component <= 0 for component in shape):
        raise argparse.ArgumentTypeError("supercell-shape values must be positive")
    return shape


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a tutorial kMC simulation using files in tests/files"
    )
    parser.add_argument(
        "--output-dir",
        default="example/output/tutorial",
        help="Directory where generated event files and simulation outputs are written.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=298.0,
        help="Simulation temperature in K.",
    )
    parser.add_argument(
        "--attempt-frequency",
        type=float,
        default=5e12,
        help="Attempt frequency (prefactor) in Hz.",
    )
    parser.add_argument(
        "--equilibration-passes",
        type=int,
        default=1,
        help="Number of equilibration passes.",
    )
    parser.add_argument(
        "--kmc-passes",
        type=int,
        default=100,
        help="Number of production kMC passes.",
    )
    parser.add_argument(
        "--supercell-shape",
        type=parse_supercell_shape,
        default=(2, 1, 1),
        help="Supercell shape as comma-separated integers, for example: 2,1,1",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=12345,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--regenerate-events",
        action="store_true",
        help="Regenerate event and dependency files even if they already exist.",
    )
    return parser


def resolve_output_dir(repo_root: Path, output_dir_arg: str) -> Path:
    output_dir = Path(output_dir_arg)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    return output_dir.resolve()


def generate_events_if_needed(
    structure_file: Path,
    event_file: Path,
    supercell_shape: tuple[int, int, int],
    regenerate_events: bool,
) -> None:
    """Generate event library with embedded dependencies (single file)."""
    if not regenerate_events and event_file.exists():
        print("Reusing existing event file:")
        print(f"  {event_file}")
        return

    print("Generating event library from example structure...")
    generator = EventGenerator()
    generator.generate_events(
        structure_file=str(structure_file),
        convert_to_primitive_cell=False,
        local_env_cutoff_dict={("Na+", "Na+"): 4.0, ("Na+", "Si4+"): 4.0},
        mobile_ion_identifiers=("Na1", "Na2"),
        site_mapping=NASICON_SITE_MAPPING,
        distance_matrix_rtol=0.01,
        distance_matrix_atol=0.01,
        find_nearest_if_fail=False,
        export_local_env_structure=False,
        supercell_shape=list(supercell_shape),
        event_file=str(event_file),
        # Dependencies are embedded in events.json - no separate file needed
    )


def run_tutorial(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    files_dir = repo_root / "tests" / "files"

    output_dir = resolve_output_dir(repo_root=repo_root, output_dir_arg=args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    structure_file = files_dir / "EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif"
    event_file = output_dir / "events.json"

    generate_events_if_needed(
        structure_file=structure_file,
        event_file=event_file,
        supercell_shape=args.supercell_shape,
        regenerate_events=args.regenerate_events,
    )

    config = Configuration(
        structure_file=str(structure_file),
        model_file=str(files_dir / "input/model.json"),
        event_file=str(event_file),
        # No event_dependencies needed - embedded in events.json
        initial_state_file=str(files_dir / "input/initial_state.json"),
        mobile_ion_specie="Na",
        temperature=args.temperature,
        attempt_frequency=args.attempt_frequency,
        equilibration_passes=args.equilibration_passes,
        kmc_passes=args.kmc_passes,
        supercell_shape=args.supercell_shape,
        site_mapping=NASICON_SITE_MAPPING,
        immutable_sites=("Zr", "O", "Zr4+", "O2-"),
        convert_to_primitive_cell=False,
        mobile_ion_charge=1.0,
        elementary_hop_distance=3.47782,
        dimension=3,
        name="NASICON_Tutorial",
        random_seed=args.random_seed,
    )

    print("Initializing KMC from generated tutorial configuration...")
    kmc = KMC.from_config(config)

    # Example custom property callback: sample occupied-site fraction every 50 events.
    def calc_occupation(state, step, sim_time):
        _ = step, sim_time
        occupied = sum(1 for occ in state.occupations if occ < 0)
        return occupied / len(state.occupations)

    kmc.attach(calc_occupation, interval=50, name="calc_occupation")

    original_cwd = Path.cwd()
    try:
        os.chdir(output_dir)
        print(f"Running simulation in: {output_dir}")
        tracker = kmc.run()
    finally:
        os.chdir(original_cwd)

    print("Tutorial run complete.")
    print(
        "Final metrics (time, msd, jump_diffusivity, tracer_diffusivity, conductivity, havens_ratio, correlation_factor):"
    )
    print(f"  {tracker.return_current_info()}")
    print("Generated outputs in:")
    print(f"  {output_dir}")
    print("Look for files named like:")
    print("  results_NASICON_Tutorial.csv.gz")
    print("  custom_results_NASICON_Tutorial.json.gz")
    print("  displacement_NASICON_Tutorial_<pass>.csv.gz")
    print("  hop_counter_NASICON_Tutorial_<pass>.csv.gz")
    print("  current_occ_NASICON_Tutorial_<pass>.csv.gz")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_tutorial(args)


if __name__ == "__main__":
    main()
