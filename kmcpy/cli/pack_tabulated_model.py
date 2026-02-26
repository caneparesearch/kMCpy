"""CLI helper to build tabulated model bundles from JSON entries."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from kmcpy.io.config_io import ConfigIO


DEFAULT_BUNDLE_FILENAME = "model.json"


def write_tabulated_model_bundle(
    output: str | Path,
    entries_file: str,
    name: str | None = None,
    default_property: str | None = None,
    probability_property: str | None = None,
    force: bool = False,
) -> Path:
    """Build and write a tabulated model bundle from JSON input."""
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {output_path}. Use --force to overwrite."
        )

    bundle = ConfigIO.build_tabulated_model_bundle_from_file(
        entries_file=entries_file,
        name=name,
        default_property=default_property,
        probability_property=probability_property,
    )
    ConfigIO.save_model_bundle(bundle, str(output_path))
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build parser for standalone tabulated model packing command."""
    parser = argparse.ArgumentParser(
        description="Build a tabulated kMCpy model bundle from JSON entries.",
    )
    parser.add_argument(
        "--entries-file",
        required=True,
        help="Path to JSON file with tabulated entries (list or object with 'entries').",
    )
    parser.add_argument("--name", help="Optional tabulated model name.")
    parser.add_argument(
        "--default-property",
        help="Property key returned by compute(...) when property_name is omitted.",
    )
    parser.add_argument(
        "--probability-property",
        help="Property key used for Arrhenius probability computation.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_BUNDLE_FILENAME,
        help=f"Output bundle JSON path (default: {DEFAULT_BUNDLE_FILENAME})",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for creating a tabulated model bundle."""
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = write_tabulated_model_bundle(
        output=args.output,
        entries_file=args.entries_file,
        name=args.name,
        default_property=args.default_property,
        probability_property=args.probability_property,
        force=args.force,
    )
    print(f"Tabulated model bundle written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
