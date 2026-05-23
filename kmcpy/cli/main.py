"""Top-level ``kmcpy`` CLI with subcommands."""

from __future__ import annotations

import argparse
from typing import Sequence

from kmcpy.cli.init import DEFAULT_TEMPLATE_FILENAME, write_template
from kmcpy.cli.pack_tabulated_model import (
    DEFAULT_BUNDLE_FILENAME,
    write_tabulated_model_bundle,
)


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

    pack_tabulated_parser = subparsers.add_parser(
        "pack-tabulated-model",
        help="Build a tabulated model bundle JSON from tabulated entries.",
    )
    pack_tabulated_parser.add_argument(
        "--entries-file",
        required=True,
        help="Path to JSON entries file (list or object with key 'entries').",
    )
    pack_tabulated_parser.add_argument(
        "--name",
        help="Optional tabulated model name.",
    )
    pack_tabulated_parser.add_argument(
        "--default-property",
        help="Optional default property key for compute(...).",
    )
    pack_tabulated_parser.add_argument(
        "--probability-property",
        help="Optional barrier property key for Arrhenius probability.",
    )
    pack_tabulated_parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_BUNDLE_FILENAME,
        help=f"Output bundle JSON path (default: {DEFAULT_BUNDLE_FILENAME})",
    )
    pack_tabulated_parser.add_argument(
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

    if args.command == "pack-tabulated-model":
        output_path = write_tabulated_model_bundle(
            output=args.output,
            entries_file=args.entries_file,
            name=args.name,
            default_property=args.default_property,
            probability_property=args.probability_property,
            force=args.force,
        )
        print(f"Model bundle written to: {output_path}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
