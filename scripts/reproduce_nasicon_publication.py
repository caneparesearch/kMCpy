#!/usr/bin/env python
"""Build and run a small NASICON publication-reproduction smoke workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

from kmcpy.event import Event, EventLib
from kmcpy.io.config_io import ConfigIO
from kmcpy.io.serialization import to_json_compatible
from kmcpy.simulator.config import Configuration
from kmcpy.simulator.kmc import KMC
from kmcpy.structure.local_site_ordering import LocalSiteOrderingConvention


DEFAULT_SOURCE_REPO = Path("/home/jerry/work/tmp/project_nasicon_kmc")
DEFAULT_OUTPUT_DIR = Path("/tmp/kmcpy_nasicon_repro")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=to_json_compatible)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _patch_legacy_pymatgen_species() -> None:
    """Allow old pymatgen pickles to load with modern pymatgen."""
    from pymatgen.core.periodic_table import Species

    if getattr(Species, "_kmcpy_legacy_pickle_patch", False):
        return

    original_str = Species.__str__

    def safe_str(self):
        if not hasattr(self, "_spin"):
            self._spin = None
        return original_str(self)

    Species.__str__ = safe_str
    Species._kmcpy_legacy_pickle_patch = True


def _load_legacy_pickle(source_repo: Path, relative_path: str) -> Any:
    _patch_legacy_pymatgen_species()
    legacy_lib = str(source_repo / "lib")
    if legacy_lib not in sys.path:
        sys.path.insert(0, legacy_lib)

    with (source_repo / relative_path).open("rb") as handle:
        return pickle.load(handle)


def _nested_int_lists(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_nested_int_lists(item) for item in value]
    return int(value)


def _legacy_lce_to_dict(legacy_lce: Any, name: str) -> dict[str, Any]:
    cluster_site_indices = getattr(legacy_lce, "sublattice_indices", None)
    if cluster_site_indices is None:
        raise ValueError("Legacy LCE object does not define sublattice_indices")

    return {
        "@module": "kmcpy.models.local_cluster_expansion",
        "@class": "LocalClusterExpansion",
        "name": name,
        "cluster_site_indices": _nested_int_lists(cluster_site_indices),
        "ordering_convention": LocalSiteOrderingConvention.from_name(
            "nasicon_publication_v1"
        ).as_dict(),
    }


def _latest_fit_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload.values() if isinstance(row, dict)]
    if not rows:
        raise ValueError(f"No fitting rows found in {path}")
    rows_with_ts = [row for row in rows if row.get("time_stamp") is not None]
    return max(rows_with_ts, key=lambda row: row["time_stamp"]) if rows_with_ts else rows[-1]


def validate_neb_sources(source_repo: Path) -> dict[str, Any]:
    lce_dir = source_repo / "local_cluster_expansion_new"
    data_csv = lce_dir / "data.csv"
    corr_file = lce_dir / "correlation_matrix.txt"
    env_dir = lce_dir / "different_envs"

    if not data_csv.exists():
        raise FileNotFoundError(f"Missing NEB-derived table: {data_csv}")
    if not corr_file.exists():
        raise FileNotFoundError(f"Missing correlation matrix: {corr_file}")
    if not env_dir.is_dir():
        raise FileNotFoundError(f"Missing CIF environment directory: {env_dir}")

    rows = _read_csv_rows(data_csv)
    if not rows:
        raise ValueError(f"No rows found in {data_csv}")

    required_columns = {
        "name",
        "ekra",
        "esite",
        "select",
        "weight_ekra",
        "weight_esite",
    }
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        raise ValueError(f"{data_csv} is missing required columns: {sorted(missing_columns)}")

    missing_cifs = sorted(
        {
            row["name"]
            for row in rows
            if row.get("name") and not (env_dir / Path(row["name"]).name).exists()
        }
    )
    if missing_cifs:
        raise FileNotFoundError(f"Missing CIF environments: {missing_cifs}")

    selected_rows = [row for row in rows if int(float(row["select"])) == 1]
    correlation_matrix = np.loadtxt(corr_file)
    if correlation_matrix.ndim == 1:
        correlation_matrix = correlation_matrix.reshape(1, -1)
    if correlation_matrix.shape[0] != len(rows):
        raise ValueError(
            f"Correlation matrix row count {correlation_matrix.shape[0]} "
            f"does not match NEB-derived rows {len(rows)}"
        )

    raw_neb_dirs = [
        source_repo / "local_cluster_expansion" / "data_house" / "neb_ci",
        source_repo / "manuscript" / "SI_figures" / "NEB_Figs" / "data",
    ]
    raw_neb_profile_count = sum(
        len(list(path.rglob("*.csv"))) for path in raw_neb_dirs if path.exists()
    )

    return {
        "data_csv": str(data_csv),
        "correlation_matrix": str(corr_file),
        "environment_dir": str(env_dir),
        "row_count": len(rows),
        "selected_row_count": len(selected_rows),
        "correlation_matrix_shape": list(correlation_matrix.shape),
        "raw_neb_profile_count": raw_neb_profile_count,
        "hashes": {
            "data_csv": _sha256(data_csv),
            "correlation_matrix": _sha256(corr_file),
        },
    }


def convert_legacy_lce_model(source_repo: Path, output_dir: Path) -> dict[str, Path]:
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    legacy_lce = _load_legacy_pickle(
        source_repo, "local_cluster_expansion_new/local_cluster_expansion.pkl"
    )
    lce_json = artifacts_dir / "legacy_lce.json"
    _json_dump(lce_json, _legacy_lce_to_dict(legacy_lce, "NASICONLegacyLCE"))

    model_json = artifacts_dir / "model.json"
    bundle = ConfigIO.build_model_bundle_from_legacy_files(
        kra_lce=str(lce_json),
        kra_fit=str(source_repo / "local_cluster_expansion_new" / "fitting_ekra.json"),
        site_lce=str(lce_json),
        site_fit=str(source_repo / "local_cluster_expansion_new" / "fitting_esite.json"),
    )
    ConfigIO.save_model_bundle(bundle, str(model_json))

    keci_length = len(bundle["kra"]["parameters"]["keci"])
    cluster_count = len(bundle["kra"]["lce"]["cluster_site_indices"])
    if keci_length != cluster_count:
        raise ValueError(
            f"Fitted KRA KECI length {keci_length} does not match "
            f"legacy LCE cluster count {cluster_count}"
        )
    site_keci_length = len(bundle["site"]["parameters"]["keci"])
    if site_keci_length != cluster_count:
        raise ValueError(
            f"Fitted site KECI length {site_keci_length} does not match "
            f"legacy LCE cluster count {cluster_count}"
        )

    return {"lce_json": lce_json, "model_json": model_json}


def convert_legacy_events(source_repo: Path, output_dir: Path) -> Path:
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    legacy_events = _load_legacy_pickle(source_repo, "inputs/events_222_double.pkl")
    event_lib = EventLib()
    for legacy_event in legacy_events:
        event_lib.add_event(
            Event(
                mobile_ion_indices=(
                    int(legacy_event.na1_idx),
                    int(legacy_event.na2_idx),
                ),
                local_env_indices=tuple(
                    int(index) for index in legacy_event.sorted_env_global_indices
                ),
            )
        )
    event_lib.generate_event_dependencies()

    event_json = artifacts_dir / "events_222_double.json"
    event_lib.to_json(str(event_json))
    return event_json


def load_initial_occupations(source_repo: Path) -> list[int]:
    config_csv = source_repo / "kmc" / "config.csv"
    if not config_csv.exists():
        raise FileNotFoundError(f"Missing 2x2x2 occupation file: {config_csv}")
    occupations = np.loadtxt(config_csv, delimiter=",")
    return [int(value) for value in occupations.tolist()]


def run_quick_kmc(
    source_repo: Path,
    output_dir: Path,
    model_json: Path,
    event_json: Path,
    initial_occupations: list[int],
    *,
    random_seed: int,
) -> dict[str, Any]:
    run_dir = output_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    structure_file = source_repo / "local_cluster_expansion_new" / "EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif"
    config = Configuration.create(
        structure_file=str(structure_file),
        model_file=str(model_json),
        event_file=str(event_json),
        initial_occupations=initial_occupations,
        mobile_ion_specie="Na",
        temperature=573.0,
        attempt_frequency=1e13,
        equilibration_passes=1,
        kmc_passes=20,
        supercell_shape=(2, 2, 2),
        immutable_sites=("Zr", "O", "Zr4+", "O2-"),
        convert_to_primitive_cell=True,
        mobile_ion_charge=1.0,
        elementary_hop_distance=3.47782,
        dimension=3,
        name="NASICON_Publication_Quick",
        random_seed=random_seed,
    )

    kmc = KMC.from_config(config)

    original_cwd = Path.cwd()
    try:
        os.chdir(run_dir)
        tracker = kmc.run(config)
    finally:
        os.chdir(original_cwd)

    metric_names = (
        "time",
        "msd",
        "jump_diffusivity",
        "tracer_diffusivity",
        "conductivity",
        "havens_ratio",
        "correlation_factor",
    )
    metrics = dict(zip(metric_names, tracker.return_current_info()))
    final_occupations = [int(value) for value in kmc.simulation_state.occupations]
    initial_array = np.array(initial_occupations)
    final_array = np.array(final_occupations)

    conservation = {
        "occupation_length": len(final_occupations),
        "initial_occupied_count": int(np.count_nonzero(initial_array == -1)),
        "final_occupied_count": int(np.count_nonzero(final_array == -1)),
        "initial_vacancy_count": int(np.count_nonzero(initial_array == 1)),
        "final_vacancy_count": int(np.count_nonzero(final_array == 1)),
        "occupied_count_conserved": bool(
            np.count_nonzero(initial_array == -1) == np.count_nonzero(final_array == -1)
        ),
    }

    return {
        "run_dir": str(run_dir),
        "metrics": metrics,
        "conservation": conservation,
        "output_files": sorted(path.name for path in run_dir.glob("*")),
    }


def run_reproduction(
    source_repo: Path = DEFAULT_SOURCE_REPO,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    mode: str = "quick",
    random_seed: int = 12345,
) -> dict[str, Any]:
    if mode != "quick":
        raise ValueError("Only --mode quick is supported by this reproduction harness")

    source_repo = source_repo.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    neb_summary = validate_neb_sources(source_repo)
    model_artifacts = convert_legacy_lce_model(source_repo, output_dir)
    event_json = convert_legacy_events(source_repo, output_dir)
    initial_occupations = load_initial_occupations(source_repo)
    _json_dump(output_dir / "artifacts" / "initial_occupations.json", initial_occupations)

    run_summary = run_quick_kmc(
        source_repo=source_repo,
        output_dir=output_dir,
        model_json=model_artifacts["model_json"],
        event_json=event_json,
        initial_occupations=initial_occupations,
        random_seed=random_seed,
    )

    artifact_hashes = {
        "lce_json": _sha256(model_artifacts["lce_json"]),
        "model_json": _sha256(model_artifacts["model_json"]),
        "event_json": _sha256(event_json),
    }

    fitting_sources = {
        "kra": str(source_repo / "local_cluster_expansion_new" / "fitting_ekra.json"),
        "site": str(source_repo / "local_cluster_expansion_new" / "fitting_esite.json"),
    }

    event_payload = json.loads(event_json.read_text(encoding="utf-8"))
    summary = {
        "mode": mode,
        "source_repo": str(source_repo),
        "output_dir": str(output_dir),
        "neb_sources": neb_summary,
        "artifacts": {
            "lce_json": str(model_artifacts["lce_json"]),
            "model_json": str(model_artifacts["model_json"]),
            "event_json": str(event_json),
            "initial_occupations_json": str(output_dir / "artifacts" / "initial_occupations.json"),
            "hashes": artifact_hashes,
        },
        "fit_metadata": {
            "kra": _latest_fit_record(Path(fitting_sources["kra"])),
            "site": _latest_fit_record(Path(fitting_sources["site"])),
        },
        "event_count": len(event_payload["events"]),
        "quick_run": run_summary,
    }

    _json_dump(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small modern-kMCpy reproduction harness for the NASICON publication workflow."
    )
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=DEFAULT_SOURCE_REPO,
        help=f"Path to the old NASICON repository (default: {DEFAULT_SOURCE_REPO})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--mode",
        choices=("quick",),
        default="quick",
        help="Reproduction mode. Only quick is currently supported.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=12345,
        help="Random seed for the deterministic quick KMC smoke run.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run_reproduction(
        source_repo=args.source_repo,
        output_dir=args.output_dir,
        mode=args.mode,
        random_seed=args.random_seed,
    )
    print(f"Summary written to: {Path(summary['output_dir']) / 'summary.json'}")
    print(json.dumps({"metrics": summary["quick_run"]["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
