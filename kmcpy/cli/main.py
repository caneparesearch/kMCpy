"""Top-level ``kmcpy`` CLI with subcommands."""

from __future__ import annotations

import argparse
from typing import Sequence

from kmcpy.cli.init import DEFAULT_TEMPLATE_FILENAME, write_template


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``kmcpy`` parser."""
    parser = argparse.ArgumentParser(
        prog="kmcpy",
        description="kMCpy command-line tools",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init",
        help="Generate a commented YAML template for a KMC simulation.",
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

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
