"""CLI helper to pack legacy model files into a single bundle JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from kmcpy.io.config_io import ConfigIO


DEFAULT_BUNDLE_FILENAME = "model.json"


def write_model_bundle(
    output: str | Path,
    kra_lce: str,
    kra_fit: str,
    site_lce: str | None = None,
    site_fit: str | None = None,
    force: bool = False,
) -> Path:
    """Create and write a bundled model JSON from legacy inputs."""
    if (site_lce is None) ^ (site_fit is None):
        raise ValueError("Provide both --site-lce and --site-fit, or neither.")

    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {output_path}. Use --force to overwrite."
        )

    bundle = ConfigIO.build_model_bundle_from_legacy_files(
        kra_lce=kra_lce,
        kra_fit=kra_fit,
        site_lce=site_lce,
        site_fit=site_fit,
    )
    ConfigIO.save_model_bundle(bundle, str(output_path))
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build parser for standalone pack-model command."""
    parser = argparse.ArgumentParser(
        description="Pack legacy kMCpy model files into a single model bundle JSON.",
    )
    parser.add_argument("--kra-lce", required=True, help="Path to KRA LCE JSON file.")
    parser.add_argument("--kra-fit", required=True, help="Path to KRA fitting-results JSON file.")
    parser.add_argument("--site-lce", help="Path to site LCE JSON file (optional, paired with --site-fit).")
    parser.add_argument("--site-fit", help="Path to site fitting-results JSON file (optional, paired with --site-lce).")
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
    """Entry point for packing a legacy model bundle."""
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = write_model_bundle(
        output=args.output,
        kra_lce=args.kra_lce,
        kra_fit=args.kra_fit,
        site_lce=args.site_lce,
        site_fit=args.site_fit,
        force=args.force,
    )
    print(f"Model bundle written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
