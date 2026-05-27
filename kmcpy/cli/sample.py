"""CLI helpers for writing concrete starter input artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from kmcpy.models import LocalBarrierModel
from kmcpy.simulator.config import Configuration
from kmcpy.simulator.state import State


DEFAULT_SAMPLE_CONFIG_FILENAME = "input.yaml"
DEFAULT_SAMPLE_MODEL_FILENAME = "model.json"
DEFAULT_SAMPLE_STATE_FILENAME = "initial_state.json"


def _prepare_output_path(output: str | Path, force: bool = False) -> Path:
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {output_path}. "
            "Use --force to overwrite."
        )
    return output_path


def sample_configuration(
    model_file: str = DEFAULT_SAMPLE_MODEL_FILENAME,
    initial_state_file: str = DEFAULT_SAMPLE_STATE_FILENAME,
    event_file: str = "events.json",
    structure_file: str = "structure.cif",
) -> Configuration:
    """Return a small reloadable Configuration for a local-barrier run."""
    return Configuration(
        structure_file=structure_file,
        event_file=event_file,
        model_file=model_file,
        initial_state_file=initial_state_file,
        initial_occupations=None,
        supercell_shape=(1, 1, 1),
        dimension=3,
        mobile_ion_specie="Li",
        mobile_ion_charge=1.0,
        elementary_hop_distance=1.0,
        model_type="local_barrier",
        site_mapping={"Li": ["Li", "X"]},
        convert_to_primitive_cell=False,
        temperature=300.0,
        attempt_frequency=1e13,
        equilibration_passes=0,
        kmc_passes=100,
        random_seed=12345,
        name="SampleSimulation",
    )


def write_sample_config(
    output: str | Path,
    force: bool = False,
    model_file: str = DEFAULT_SAMPLE_MODEL_FILENAME,
    initial_state_file: str = DEFAULT_SAMPLE_STATE_FILENAME,
    event_file: str = "events.json",
    structure_file: str = "structure.cif",
) -> Path:
    """Write a sample Configuration YAML or JSON file."""
    output_path = _prepare_output_path(output, force=force)
    config = sample_configuration(
        model_file=model_file,
        initial_state_file=initial_state_file,
        event_file=event_file,
        structure_file=structure_file,
    )
    if output_path.suffix.lower() in {".yaml", ".yml"}:
        config.to(output_path, include_loader_paths=True, section="kmc")
    else:
        config.to(output_path, include_loader_paths=True)
    return output_path


def write_sample_model(
    output: str | Path,
    force: bool = False,
    barrier: float = 300.0,
) -> Path:
    """Write a constant-barrier LocalBarrierModel JSON file."""
    output_path = _prepare_output_path(output, force=force)
    model = LocalBarrierModel.constant_barrier(float(barrier))
    model.to(str(output_path))
    return output_path


def write_sample_state(
    output: str | Path,
    occupations: Sequence[int] = (0, 1),
    force: bool = False,
) -> Path:
    """Write an initial State JSON file with active-site occupations."""
    output_path = _prepare_output_path(output, force=force)
    State.from_occupations(occupations).save_checkpoint(str(output_path))
    return output_path


def write_sample_set(
    output_dir: str | Path,
    force: bool = False,
    barrier: float = 300.0,
    occupations: Sequence[int] = (0, 1),
) -> dict[str, Path]:
    """Write a small config/model/state starter set into ``output_dir``."""
    root = Path(output_dir).expanduser()
    paths = {
        "config": root / DEFAULT_SAMPLE_CONFIG_FILENAME,
        "model": root / DEFAULT_SAMPLE_MODEL_FILENAME,
        "state": root / DEFAULT_SAMPLE_STATE_FILENAME,
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing files: {names}. "
            "Use --force to overwrite."
        )

    root.mkdir(parents=True, exist_ok=True)
    write_sample_model(paths["model"], force=force, barrier=barrier)
    write_sample_state(paths["state"], occupations=occupations, force=force)
    write_sample_config(
        paths["config"],
        force=force,
        model_file=str(paths["model"]),
        initial_state_file=str(paths["state"]),
    )
    return paths


def _occupations_argument(values: Sequence[str]) -> list[int]:
    occupations: list[int] = []
    for value in values:
        parts = [part for part in value.split(",") if part]
        try:
            occupations.extend(int(part) for part in parts)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "occupations must be integers, separated by spaces or commas"
            ) from exc
    if not occupations:
        raise argparse.ArgumentTypeError("at least one occupation value is required")
    return occupations


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add sample subcommands to ``parser`` and return it."""
    subparsers = parser.add_subparsers(dest="sample_command", required=True)

    config_parser = subparsers.add_parser(
        "config",
        help="Write a sample Configuration YAML or JSON file.",
        description="Write a sample reloadable Configuration input file.",
        epilog=(
            "Examples:\n"
            "  kmcpy sample config --output input.yaml\n"
            "  kmcpy sample config --output input.json --model-file model.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    config_parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_SAMPLE_CONFIG_FILENAME,
        help=f"Output YAML or JSON path (default: {DEFAULT_SAMPLE_CONFIG_FILENAME}).",
    )
    config_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    config_parser.add_argument(
        "--model-file",
        default=DEFAULT_SAMPLE_MODEL_FILENAME,
        help=f"Model path written into the sample config (default: {DEFAULT_SAMPLE_MODEL_FILENAME}).",
    )
    config_parser.add_argument(
        "--initial-state-file",
        default=DEFAULT_SAMPLE_STATE_FILENAME,
        help=(
            "Initial state path written into the sample config "
            f"(default: {DEFAULT_SAMPLE_STATE_FILENAME})."
        ),
    )
    config_parser.add_argument(
        "--event-file",
        default="events.json",
        help="Event library path written into the sample config (default: events.json).",
    )
    config_parser.add_argument(
        "--structure-file",
        default="structure.cif",
        help="Structure path written into the sample config (default: structure.cif).",
    )

    model_parser = subparsers.add_parser(
        "model",
        help="Write a constant-barrier LocalBarrierModel JSON file.",
        description="Write a constant-barrier LocalBarrierModel JSON file.",
        epilog=(
            "Examples:\n"
            "  kmcpy sample model --output model.json\n"
            "  kmcpy sample model --output model.json --barrier 450"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    model_parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_SAMPLE_MODEL_FILENAME,
        help=f"Output model JSON path (default: {DEFAULT_SAMPLE_MODEL_FILENAME}).",
    )
    model_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    model_parser.add_argument(
        "--barrier",
        type=float,
        default=300.0,
        help="Constant migration barrier in meV (default: 300).",
    )

    state_parser = subparsers.add_parser(
        "state",
        help="Write an initial State JSON file.",
        description="Write an initial State JSON file with active-site occupations.",
        epilog=(
            "Examples:\n"
            "  kmcpy sample state --output initial_state.json\n"
            "  kmcpy sample state --output initial_state.json --occupations 0,1,0"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    state_parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_SAMPLE_STATE_FILENAME,
        help=f"Output state JSON path (default: {DEFAULT_SAMPLE_STATE_FILENAME}).",
    )
    state_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    state_parser.add_argument(
        "--occupations",
        nargs="+",
        default=["0", "1"],
        help="Active-site occupations, separated by spaces or commas.",
    )

    all_parser = subparsers.add_parser(
        "all",
        help="Write input.yaml, model.json, and initial_state.json.",
    )
    all_parser.description = (
        "Write a linked starter set containing input.yaml, model.json, "
        "and initial_state.json."
    )
    all_parser.epilog = (
        "Examples:\n"
        "  kmcpy sample all --output-dir kmcpy_sample\n"
        "  kmcpy sample all --output-dir demo --barrier 450 --occupations 0,1"
    )
    all_parser.formatter_class = argparse.RawDescriptionHelpFormatter
    all_parser.add_argument(
        "-o",
        "--output-dir",
        default="kmcpy_sample",
        help="Directory for the generated sample files (default: kmcpy_sample).",
    )
    all_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing sample files in the output directory.",
    )
    all_parser.add_argument(
        "--barrier",
        type=float,
        default=300.0,
        help="Constant migration barrier in meV for model.json (default: 300).",
    )
    all_parser.add_argument(
        "--occupations",
        nargs="+",
        default=["0", "1"],
        help="Active-site occupations, separated by spaces or commas.",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Build parser for the standalone sample command."""
    parser = argparse.ArgumentParser(
        description="Generate concrete sample kMCpy input artifacts.",
        epilog=(
            "Examples:\n"
            "  kmcpy sample all --output-dir kmcpy_sample\n"
            "  kmcpy sample model --output model.json --barrier 300"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return configure_parser(parser)


def run_sample_command(args: argparse.Namespace):
    """Execute a parsed sample subcommand."""
    if args.sample_command == "config":
        return write_sample_config(
            args.output,
            force=args.force,
            model_file=args.model_file,
            initial_state_file=args.initial_state_file,
            event_file=args.event_file,
            structure_file=args.structure_file,
        )
    if args.sample_command == "model":
        return write_sample_model(args.output, force=args.force, barrier=args.barrier)
    if args.sample_command == "state":
        return write_sample_state(
            args.output,
            occupations=_occupations_argument(args.occupations),
            force=args.force,
        )
    if args.sample_command == "all":
        return write_sample_set(
            args.output_dir,
            force=args.force,
            barrier=args.barrier,
            occupations=_occupations_argument(args.occupations),
        )
    raise ValueError(f"Unknown sample command: {args.sample_command}")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for generating sample files."""
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_sample_command(args)
    if isinstance(result, dict):
        for name, path in result.items():
            print(f"{name}: {path}")
    else:
        print(f"Sample written to: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
