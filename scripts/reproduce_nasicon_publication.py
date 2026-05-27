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
from monty.serialization import dumpfn
from kmcpy.models.composite_lce_model import CompositeLCEModel
from kmcpy.models.fitting.fitter import LCEFitter
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
from kmcpy.simulator.config import Configuration
from kmcpy.simulator.kmc import KMC
from kmcpy.structure import ActiveSiteOrder, LocalSiteOrder
from kmcpy.io.cif import load_labeled_structure_from_cif


DEFAULT_SOURCE_REPO = Path("/home/jerry/work/tmp/project_nasicon_kmc")
DEFAULT_OUTPUT_DIR = Path("/tmp/kmcpy_nasicon_repro")
NASICON_SITE_MAPPING = {"Na": ["Na", "X"], "Zr": "Zr", "Si": ["Si", "P"], "O": "O"}
NASICON_STRUCTURE = "local_cluster_expansion_new/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dumpfn(payload, path, indent=2)


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
        "local_site_order": LocalSiteOrder.from_name(
            "nasicon_nat_commun_2022"
        ).as_dict(),
    }


def _latest_fit_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload.values() if isinstance(row, dict)]
    if not rows:
        raise ValueError(f"No fitting rows found in {path}")
    rows_with_ts = [row for row in rows if row.get("time_stamp") is not None]
    return max(rows_with_ts, key=lambda row: row["time_stamp"]) if rows_with_ts else rows[-1]


def _load_2d_array(path: Path, **kwargs) -> np.ndarray:
    values = np.loadtxt(path, **kwargs)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values


def _correlation_matrix_from_occupation(
    occupation_matrix: np.ndarray, cluster_site_indices: Any
) -> np.ndarray:
    rows = []
    for occupation in occupation_matrix:
        corr_row = []
        for orbit in cluster_site_indices:
            orbit_sum = 0.0
            for cluster in orbit:
                indices = [int(index) for index in cluster]
                orbit_sum += float(np.prod(occupation[indices]))
            corr_row.append(orbit_sum)
        rows.append(corr_row)
    return np.array(rows, dtype=float)


def _value_check(actual: np.ndarray, expected: np.ndarray, *, atol: float = 1e-12) -> dict[str, Any]:
    if actual.shape != expected.shape:
        raise ValueError(
            f"Numeric check shape mismatch: calculated {actual.shape}, expected {expected.shape}"
        )
    abs_diff = np.abs(actual - expected)
    max_abs_diff = float(abs_diff.max()) if abs_diff.size else 0.0
    allclose = bool(np.allclose(actual, expected, atol=atol, rtol=0.0))
    if not allclose:
        raise ValueError(
            f"Numeric check failed: max absolute difference {max_abs_diff} exceeds {atol}"
        )
    return {
        "allclose_atol": atol,
        "max_abs_diff": max_abs_diff,
        "nonzero_diff_count": int(np.count_nonzero(abs_diff > atol)),
    }


def _selected_rows(rows: list[dict[str, str]]) -> np.ndarray:
    return np.array([int(float(row["select"])) == 1 for row in rows], dtype=bool)


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
    correlation_matrix = _load_2d_array(corr_file)
    if correlation_matrix.shape[0] != len(rows):
        raise ValueError(
            f"Correlation matrix row count {correlation_matrix.shape[0]} "
            f"does not match NEB-derived rows {len(rows)}"
        )

    occupation_file = lce_dir / "occupation.txt"
    if not occupation_file.exists():
        raise FileNotFoundError(f"Missing occupation matrix: {occupation_file}")
    occupation_matrix = _load_2d_array(occupation_file)
    legacy_lce = _load_legacy_pickle(
        source_repo, "local_cluster_expansion_new/local_cluster_expansion.pkl"
    )
    calculated_correlation_matrix = _correlation_matrix_from_occupation(
        occupation_matrix, legacy_lce.sublattice_indices
    )
    correlation_value_check = _value_check(
        calculated_correlation_matrix, correlation_matrix
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
        "occupation_matrix": str(occupation_file),
        "environment_dir": str(env_dir),
        "row_count": len(rows),
        "selected_row_count": len(selected_rows),
        "correlation_matrix_shape": list(correlation_matrix.shape),
        "occupation_matrix_shape": list(occupation_matrix.shape),
        "correlation_value_check": correlation_value_check,
        "raw_neb_profile_count": raw_neb_profile_count,
        "hashes": {
            "data_csv": _sha256(data_csv),
            "correlation_matrix": _sha256(corr_file),
            "occupation_matrix": _sha256(occupation_file),
        },
    }


def validate_fit_reproduction(source_repo: Path, output_dir: Path) -> dict[str, Any]:
    lce_dir = source_repo / "local_cluster_expansion_new"
    rows = _read_csv_rows(lce_dir / "data.csv")
    selected = _selected_rows(rows)
    correlation_matrix = _load_2d_array(lce_dir / "correlation_matrix.txt")[selected]
    artifacts_dir = output_dir / "artifacts" / "fit_reproduction"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    checks = {}
    targets = {
        "kra": {
            "column": "ekra",
            "weight_column": "weight_ekra",
            "alpha": 1.1,
            "fit_file": lce_dir / "fitting_ekra.json",
        },
        "site": {
            "column": "esite",
            "weight_column": "weight_esite",
            "alpha": 2.5,
            "fit_file": lce_dir / "fitting_esite.json",
        },
    }
    selected_rows = [row for row, is_selected in zip(rows, selected) if is_selected]
    for label, settings in targets.items():
        fit_dir = artifacts_dir / label
        fit_dir.mkdir(parents=True, exist_ok=True)
        y_true = np.array([float(row[settings["column"]]) for row in selected_rows])
        weight = np.array([float(row[settings["weight_column"]]) for row in selected_rows])
        corr_file = fit_dir / "correlation_matrix.txt"
        target_file = fit_dir / "target.txt"
        weight_file = fit_dir / "weight.txt"
        keci_file = fit_dir / "keci.txt"
        np.savetxt(corr_file, correlation_matrix, fmt="%.18e")
        np.savetxt(target_file, y_true, fmt="%.18e")
        np.savetxt(weight_file, weight, fmt="%.18e")

        fit_record = _latest_fit_record(settings["fit_file"])
        legacy_keci = np.array(fit_record["keci"], dtype=float)
        legacy_empty_cluster = float(fit_record["empty_cluster"])
        legacy_prediction = correlation_matrix @ legacy_keci + legacy_empty_cluster
        params, y_pred, _ = LocalClusterExpansion().fit(
            alpha=settings["alpha"],
            max_iter=1000000,
            ekra_fname=str(target_file),
            weight_fname=str(weight_file),
            corr_fname=str(corr_file),
            keci_fname=str(keci_file),
            lce_params_fname=None,
            lce_params_history_fname=None,
            fit_results_fname=None,
        )
        keci_check = _value_check(np.array(params.keci), legacy_keci, atol=1e-9)
        prediction_check = _value_check(np.array(y_pred), legacy_prediction, atol=1e-9)
        rmse_abs_diff = abs(float(params.rmse) - float(fit_record["rmse"]))
        loocv_abs_diff = abs(float(params.loocv) - float(fit_record["loocv"]))
        if rmse_abs_diff > 1e-9 or loocv_abs_diff > 1e-9:
            raise ValueError(
                f"{label} fit metrics do not reproduce legacy values: "
                f"rmse diff={rmse_abs_diff}, loocv diff={loocv_abs_diff}"
            )
        checks[label] = {
            "alpha": settings["alpha"],
            "keci": keci_check,
            "prediction": prediction_check,
            "empty_cluster_abs_diff": abs(
                float(params.empty_cluster) - legacy_empty_cluster
            ),
            "rmse_abs_diff": rmse_abs_diff,
            "loocv_abs_diff": loocv_abs_diff,
        }
    return checks


def convert_legacy_lce_model(source_repo: Path, output_dir: Path) -> dict[str, Path]:
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    legacy_lce = _load_legacy_pickle(
        source_repo, "local_cluster_expansion_new/local_cluster_expansion.pkl"
    )
    lce_json = artifacts_dir / "legacy_lce.json"
    _json_dump(lce_json, _legacy_lce_to_dict(legacy_lce, "NASICONLegacyLCE"))

    model_json = artifacts_dir / "model.json"
    kra_fit_file = source_repo / "local_cluster_expansion_new" / "fitting_ekra.json"
    site_fit_file = source_repo / "local_cluster_expansion_new" / "fitting_esite.json"

    kra_model = LocalClusterExpansion.from_file(str(lce_json))
    kra_model.set_parameters(LCEFitter.from_file(str(kra_fit_file)).model_parameters)
    site_model = LocalClusterExpansion.from_file(str(lce_json))
    site_model.set_parameters(LCEFitter.from_file(str(site_fit_file)).model_parameters)

    model = CompositeLCEModel(
        site_model=site_model,
        kra_model=kra_model,
        kra_fit_metadata=_latest_fit_record(kra_fit_file),
        site_fit_metadata=_latest_fit_record(site_fit_file),
    )
    model.to(str(model_json))
    model_data = model.as_dict()

    keci_length = len(model_data["kra"]["parameters"]["keci"])
    cluster_count = len(model_data["kra"]["lce"]["cluster_site_indices"])
    if keci_length != cluster_count:
        raise ValueError(
            f"Fitted KRA KECI length {keci_length} does not match "
            f"legacy LCE cluster count {cluster_count}"
        )
    site_keci_length = len(model_data["site"]["parameters"]["keci"])
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
    primitive_structure = load_labeled_structure_from_cif(
        str(source_repo / NASICON_STRUCTURE), primitive=True
    )
    active_site_order = ActiveSiteOrder.from_structure_and_mapping(
        primitive_structure,
        NASICON_SITE_MAPPING,
        supercell_shape=(2, 2, 2),
    )
    event_lib.set_index_metadata(active_site_order)

    event_json = artifacts_dir / "events_222_double.json"
    event_lib.to(str(event_json))
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

    structure_file = source_repo / NASICON_STRUCTURE
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
        site_mapping=NASICON_SITE_MAPPING,
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
        tracker = kmc.run()
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
    fit_value_checks = validate_fit_reproduction(source_repo, output_dir)
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
        "fit_value_checks": fit_value_checks,
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
