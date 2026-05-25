"""CLI utilities to scaffold a kMCpy Configuration YAML template."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent
from typing import Sequence


DEFAULT_TEMPLATE_FILENAME = "input_template.yaml"


def build_template() -> str:
    """Return a commented YAML template for ``Configuration``."""
    return dedent(
        """\
        # kMCpy input template (modern Configuration format)
        #
        # Usage:
        #   1) Fill the required file paths below.
        #   2) Adjust optional settings as needed.
        #   3) Run: run_kmc --input input_template.yaml
        #
        # API usage:
        #   from kmcpy.simulator.config import Configuration
        #   config = Configuration.from_file("input_template.yaml")

        kmc:
          type: default
          default:
            # ----- Required loader inputs -----
            # Path to crystal structure (CIF/structure file)
            structure_file: "path/to/structure.cif"
            # Path to migration event library JSON
            event_file: "path/to/event.json"
            # Path to serialized model JSON
            model_file: "path/to/model.json"

            # ----- Optional loader inputs -----
            # Optional serialized initial simulation state file
            initial_state_file: null
            # Optional direct initial occupations list (used if state file is null)
            initial_occupations: null

            # ----- System fields -----
            # Supercell replication factors [a, b, c]
            supercell_shape: [1, 1, 1]
            # Dimensionality of transport (1, 2, or 3)
            dimension: 3
            # Mobile ion specie label
            mobile_ion_specie: "Li"
            # Mobile ion charge in |e|
            mobile_ion_charge: 1.0
            # Characteristic hop distance (Angstrom)
            elementary_hop_distance: 1.0
            # Optional model selector for raw model payloads. Standard
            # kmcpy.model_file envelopes infer this from model_file.
            model_type: "composite_lce"
            # Site mapping. One allowed species means fixed; multiple means active.
            site_mapping:
              Li: [Li, X]
            # Convert structure to primitive cell before simulation
            convert_to_primitive_cell: false

            # ----- Runtime fields -----
            # Temperature in Kelvin
            temperature: 300.0
            # Attempt frequency (Hz)
            attempt_frequency: 10000000000000.0
            # Number of equilibration passes
            equilibration_passes: 1000
            # Number of production KMC passes
            kmc_passes: 10000
            # Optional RNG seed (null for nondeterministic)
            random_seed: null
            # Simulation label used in outputs
            name: "DefaultSimulation"

            # ----- Optional property sampling controls -----
            # Global property sampling event-step interval (null = default once per pass)
            property_sampling_interval: null
            # Global property sampling time interval (null = disabled)
            property_sampling_time_interval: null
            # Built-in property toggles (defaults to enabled for all fields)
            # Supported keys: msd, jump_diffusivity, tracer_diffusivity,
            #                 conductivity, havens_ratio, correlation_factor
            builtin_property_enabled: {}
            # Optional custom callback definitions resolved by import path.
            # Example:
            # property_callbacks:
            #   - callable: "myproject.kmc_props:calc_occupation"
            #     name: "occupied_fraction"
            #     interval: 100
            #     time_interval: null
            #     store: true
            #     max_records: null
            #     enabled: true
            property_callbacks: []
        """
    )


def write_template(output: str | Path, force: bool = False) -> Path:
    """Write the template to ``output`` and return the output path."""
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {output_path}. "
            "Use --force to overwrite."
        )

    output_path.write_text(build_template(), encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build parser for the standalone ``kmcpy-init`` style command."""
    parser = argparse.ArgumentParser(
        description="Generate a commented kMCpy YAML input template."
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_TEMPLATE_FILENAME,
        help=f"Output YAML path (default: {DEFAULT_TEMPLATE_FILENAME})",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for generating template files."""
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = write_template(args.output, force=args.force)
    print(f"Template written to: {output_path}")
    print(f"Next step: run_kmc --input {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
