"""Top-level ``kmcpy`` CLI with subcommands."""

from __future__ import annotations

import argparse
from typing import Sequence

from kmcpy.cli.init import INIT_DESCRIPTION, INIT_EPILOG, run_init_command
from kmcpy.cli.init import configure_parser as configure_init_parser
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
        help="Generate a commented YAML input template.",
        description=INIT_DESCRIPTION,
        epilog=INIT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    configure_init_parser(init_parser)

    run_parser = subparsers.add_parser(
        "run",
        help="Run a kMC simulation from an input file.",
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
        run_init_command(args)
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
