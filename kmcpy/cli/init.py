"""CLI utilities to scaffold a kMCpy input-file template."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent
from typing import Sequence


DEFAULT_TEMPLATE_FILENAME = "input_template.yaml"


TEMPLATE_FORMATS = ("simulation", "configuration")


def build_template(template_format: str = "simulation") -> str:
    """Return a commented YAML input template.

    ``"simulation"`` (default) is the sectioned input read by
    ``Simulation.from_file`` and ``kmcpy run``; ``"configuration"`` is the flat
    ``Configuration`` format.
    """
    if template_format == "simulation":
        return build_simulation_template()
    if template_format == "configuration":
        return build_configuration_template()
    raise ValueError(f"Unknown template format {template_format!r}; use one of {TEMPLATE_FORMATS}")


def build_simulation_template() -> str:
    """Return a commented YAML template for ``Simulation.from_file``."""
    return dedent(
        """\
        # kMCpy simulation input
        #
        # Usage:
        #   1) Point lattice_structure.structure at your CIF; set what may vary.
        #   2) Choose events, model, and initial state (file paths or built-in types).
        #   3) Run: kmcpy run --input input_template.yaml
        # Relative paths are resolved from the folder of this file.
        #
        # Python: kmcpy.Simulation.from_file("input_template.yaml").run()

        lattice_structure:
          # CIF with every site that can be occupied. Partial occupancies
          # define what a site may hold (Li: 0.5 -> Li or vacancy).
          structure: path/to/structure.cif
          # What may vary, for species whose sites are fully occupied in the
          # CIF. "X" is the vacancy; unlisted species are fixed. Not needed if
          # the CIF has the partial occupancies.
          site_mapping:
            Li: [Li, X]
          # Repetitions of the cell that are simulated [a, b, c]
          supercell_shape: [1, 1, 1]
          # Reduce the CIF to its primitive cell first
          primitive: false

        # Events: an event file, or a generator, for example
        #   events: {type: hop, cutoff: 4.0}                       # Angstrom
        #   events: {type: hop, cutoffs: [[Li+, Li+, 4.0]], labels: [Li1, Li2]}
        events: path/to/events.json

        # Rate model: a model file of any registered type, or inline parameters
        #   model: {type: local_barrier, default_barrier: 300.0}   # meV
        model: path/to/model.json

        # Initial state: a state file, explicit occupations, or a builder
        #   state: {occupations: [0, 1, 0, 1]}
        #   state: {type: random, fractions: {Li: 0.5}, seed: 1}
        state: path/to/initial_state.json

        run:
          temperature: 300.0              # K
          kmc_passes: 10000
          random_seed: null               # set an integer for reproducible runs
          output_dir: results             # folder for result files
          # Optional (defaults shown):
          # attempt_frequency: 1.0e+13    # Hz
          # equilibration_passes: 1000
          # name: DefaultSimulation       # label in result file names
          # dimension: 3                  # dimensionality of diffusion
          # The mobile ion, its charge, and the hop length are taken from
          # site_mapping, the structure, and the events; set
          # mobile_ion_specie / mobile_ion_charge / elementary_hop_distance
          # here only to override them.
          #
          # Results: msd (Angstrom^2), jump_diffusivity and tracer_diffusivity
          # (cm^2/s), conductivity (mS/cm), havens_ratio, correlation_factor.
          # Sampling and custom properties (see the docs: Attach properties):
          # property_sampling_interval: null      # steps; null = once per pass
          # property_sampling_time_interval: null # s
          # builtin_property_enabled: {}
          # property_callbacks:
          #   - callable: "myproject.kmc_props:calc_occupation"
          #     name: occupied_fraction
          #     interval: 100
        """
    )


def build_configuration_template() -> str:
    """Return a commented YAML template for the flat ``Configuration`` format."""
    return dedent(
        """\
        # kMCpy input template (flat Configuration format)
        #
        # Usage:
        #   1) Fill the required file paths below.
        #   2) Adjust optional settings as needed.
        #   3) Run: kmcpy run --input input_template.yaml
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
            # Optional model selector for hand-written payloads. Files written
            # by model.to(...) infer this from model_file.
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
            # Global property sampling time interval in seconds (null = disabled)
            property_sampling_time_interval: null
            # Built-in property toggles (defaults to enabled for all fields)
            # Supported keys: msd, jump_diffusivity, tracer_diffusivity,
            #                 conductivity, havens_ratio, correlation_factor
            # Output units:
            #   time: s
            #   msd: Angstrom^2
            #   jump_diffusivity, tracer_diffusivity: cm^2/s
            #   conductivity: mS/cm
            #   havens_ratio, correlation_factor: dimensionless
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


def write_template(
    output: str | Path, force: bool = False, template_format: str = "simulation"
) -> Path:
    """Write the template to ``output`` and return the output path."""
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {output_path}. "
            "Use --force to overwrite."
        )

    output_path.write_text(build_template(template_format), encoding="utf-8")
    return output_path


INIT_DESCRIPTION = (
    "Generate a commented YAML template for a kMCpy input file. "
    "Edit the paths and runtime settings before running it."
)
INIT_EPILOG = (
    "Examples:\n"
    "  kmcpy init --output input_template.yaml\n"
    "  kmcpy init --output input.yaml --force\n"
    "  kmcpy init --format configuration --output config.yaml\n"
    "  kmcpy run --input input.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    """Build parser for the standalone ``kmcpy-init`` style command."""
    parser = argparse.ArgumentParser(
        description=INIT_DESCRIPTION,
        epilog=INIT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return configure_parser(parser)


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``init`` arguments to ``parser`` and return it."""
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
    parser.add_argument(
        "--format",
        choices=TEMPLATE_FORMATS,
        default="simulation",
        help="simulation (default): lattice_structure/events/model/state/run sections; "
        "configuration: flat Configuration fields with file paths.",
    )
    return parser


def run_init_command(args: argparse.Namespace) -> Path:
    """Write the template requested by parsed ``init`` arguments."""
    output_path = write_template(args.output, force=args.force, template_format=args.format)
    print(f"Template written to: {output_path}")
    print(f"Next step: kmcpy run --input {output_path}")
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for generating template files."""
    run_init_command(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
