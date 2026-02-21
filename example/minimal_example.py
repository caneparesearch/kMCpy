#!/usr/bin/env python3
"""Minimal kMCpy example using bundled NASICON data."""

from __future__ import annotations

import json
import os
from pathlib import Path

from kmcpy.event import EventGenerator
from kmcpy.simulator.config import SimulationConfig
from kmcpy.simulator.kmc import KMC


def load_initial_occupations(initial_state_file: Path) -> list[int]:
    """Load occupation and convert from 0/1 to Chebyshev basis (-1/1)."""
    raw = json.loads(initial_state_file.read_text())["occupation"]
    return [-1 if value == 0 else value for value in raw]


def generate_events_if_needed(
    structure_file: Path,
    event_file: Path,
    event_dependencies_file: Path,
    supercell_shape: tuple[int, int, int],
) -> None:
    """Generate an event library consistent with the current supercell setup."""
    if event_file.exists() and event_dependencies_file.exists():
        return

    generator = EventGenerator()
    generator.generate_events(
        structure_file=str(structure_file),
        convert_to_primitive_cell=False,
        local_env_cutoff_dict={("Na+", "Na+"): 4.0, ("Na+", "Si4+"): 4.0},
        mobile_ion_identifier_type="label",
        mobile_ion_identifiers=("Na1", "Na2"),
        species_to_be_removed=["O2-", "O", "Zr4+", "Zr"],
        distance_matrix_rtol=0.01,
        distance_matrix_atol=0.01,
        find_nearest_if_fail=False,
        export_local_env_structure=False,
        supercell_shape=list(supercell_shape),
        event_file=str(event_file),
        event_dependencies_file=str(event_dependencies_file),
    )


def build_config() -> SimulationConfig:
    """Create a small simulation config with bundled example files."""
    repo_root = Path(__file__).resolve().parent.parent
    files_dir = repo_root / "example" / "files"
    output_dir = repo_root / "example" / "output" / "minimal"
    output_dir.mkdir(parents=True, exist_ok=True)

    supercell_shape = (2, 1, 1)
    event_file = output_dir / "events.json"
    event_dependencies_file = output_dir / "event_dependencies.csv"

    generate_events_if_needed(
        structure_file=files_dir / "nasicon.cif",
        event_file=event_file,
        event_dependencies_file=event_dependencies_file,
        supercell_shape=supercell_shape,
    )

    return SimulationConfig.create(
        # System parameters: what to simulate.
        structure_file=str(files_dir / "nasicon.cif"),
        cluster_expansion_file=str(files_dir / "input" / "lce.json"),
        fitting_results_file=str(files_dir / "input" / "fitting_results.json"),
        cluster_expansion_site_file=str(files_dir / "input" / "lce_site.json"),
        fitting_results_site_file=str(files_dir / "input" / "fitting_results_site.json"),
        event_file=str(event_file),
        event_dependencies=str(event_dependencies_file),
        initial_occupations=load_initial_occupations(files_dir / "input" / "initial_state.json"),
        supercell_shape=supercell_shape,
        dimension=3,
        mobile_ion_specie="Na",
        mobile_ion_charge=1.0,
        elementary_hop_distance=3.47782,
        immutable_sites=("Zr", "O", "Zr4+", "O2-"),
        convert_to_primitive_cell=False,
        # Runtime parameters: how to simulate.
        temperature=298.0,
        attempt_frequency=5e12,
        equilibration_passes=1,
        kmc_passes=50,
        random_seed=12345,
        name="MinimalExample",
    )


def main() -> None:
    print("Available SimulationConfig keywords:")
    SimulationConfig.help_parameters()

    config = build_config()
    print(f"\nRunning: {config.summary()}")

    kmc = KMC.from_config(config)

    output_dir = Path(__file__).resolve().parent / "output" / "minimal"
    output_dir.mkdir(parents=True, exist_ok=True)

    original_cwd = Path.cwd()
    try:
        os.chdir(output_dir)
        tracker = kmc.run(config)
    finally:
        os.chdir(original_cwd)

    print("Run complete.")
    print(
        "Final metrics (time, msd, jump_diffusivity, tracer_diffusivity, conductivity, havens_ratio, correlation_factor):"
    )
    print(f"  {tracker.return_current_info()}")
    print(f"Outputs written under: {output_dir}")


if __name__ == "__main__":
    main()
