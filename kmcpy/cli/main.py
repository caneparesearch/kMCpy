"""Top-level ``kmcpy`` CLI with subcommands."""

from __future__ import annotations

import argparse
from typing import Sequence

from kmcpy.cli.init import DEFAULT_TEMPLATE_FILENAME, write_template
from kmcpy.cli.run_kmc import RUN_HELP_EPILOG
from kmcpy.cli.run_kmc import configure_parser as configure_run_parser
from kmcpy.cli.run_kmc import run_kmc
from kmcpy.cli.sample import configure_parser as configure_sample_parser
from kmcpy.cli.sample import run_sample_command


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``kmcpy`` parser."""
    parser = argparse.ArgumentParser(
        prog="kmcpy",
        description="kMCpy command-line tools.",
        epilog=(
            "Common workflow:\n"
            "  kmcpy init --output input.yaml\n"
            "  # or: kmcpy sample all --output-dir kmcpy_sample\n"
            "  kmcpy run --input input.yaml\n\n"
            "The standalone `run_kmc --input input.yaml` command is also supported."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init",
        help="Generate a commented YAML template for a KMC simulation.",
        description=(
            "Generate a commented YAML template for a kMCpy Configuration. "
            "Edit the paths and runtime settings before running it."
        ),
        epilog=(
            "Examples:\n"
            "  kmcpy init --output input_template.yaml\n"
            "  kmcpy init --output input.yaml --force\n"
            "  run_kmc --input input.yaml"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    init_parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_TEMPLATE_FILENAME,
        help=f"Output YAML path (default: {DEFAULT_TEMPLATE_FILENAME})",
    )
    init_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run a kMC simulation from a Configuration input file.",
        description=(
            "Run a kMC simulation. The preferred interface is "
            "`kmcpy run --input input.yaml`."
        ),
        epilog=RUN_HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    configure_run_parser(run_parser)

    sample_parser = subparsers.add_parser(
        "sample",
        help="Generate concrete sample input/model/state files.",
        description="Generate concrete sample kMCpy input artifacts.",
    )
    configure_sample_parser(sample_parser)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``kmcpy`` command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        output_path = write_template(args.output, force=args.force)
        print(f"Template written to: {output_path}")
        print(f"Next step: run_kmc --input {output_path}")
        return 0

    if args.command == "run":
        run_kmc(args)
        return 0

    if args.command == "sample":
        result = run_sample_command(args)
        if isinstance(result, dict):
            for name, path in result.items():
                print(f"{name}: {path}")
        else:
            print(f"Sample written to: {result}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
