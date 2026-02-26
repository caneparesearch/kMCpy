from pathlib import Path

import pytest

from kmcpy.io.config_io import ConfigIO
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
from kmcpy.models.tabulated_model import TabulatedModel
from kmcpy.simulator.config import Configuration


@pytest.mark.unit
def test_build_and_load_model_bundle_from_legacy_files(tmp_path: Path):
    root = Path(__file__).parent / "files" / "input"
    bundle = ConfigIO.build_model_bundle_from_legacy_files(
        kra_lce=str(root / "lce.json"),
        kra_fit=str(root / "fitting_results.json"),
        site_lce=str(root / "lce_site.json"),
        site_fit=str(root / "fitting_results_site.json"),
    )

    assert bundle["format"] == "kmcpy.model_bundle.v1"
    assert bundle["model_type"] == "composite_lce"
    assert "kra" in bundle
    assert "site" in bundle

    output = tmp_path / "bundle.json"
    ConfigIO.save_model_bundle(bundle, str(output))
    loaded = ConfigIO.load_model_bundle(str(output))
    assert loaded["kra"]["parameters"]["keci"] == bundle["kra"]["parameters"]["keci"]


@pytest.mark.unit
def test_load_model_bundle_validation_error(tmp_path: Path):
    invalid = tmp_path / "invalid_bundle.json"
    invalid.write_text('{"format": "wrong", "model_type": "composite_lce"}', encoding="utf-8")

    with pytest.raises(ValueError, match="kmcpy.model_bundle.v1"):
        ConfigIO.load_model_bundle(str(invalid))


@pytest.mark.unit
def test_simulation_config_rejects_legacy_model_fields():
    with pytest.raises(ValueError, match="Legacy model fields are removed"):
        Configuration.from_dict(
            {
                "structure_file": "test.cif",
                "event_file": "event.json",
                "cluster_expansion_file": "lce.json",
                "fitting_results_file": "fit.json",
            }
        )


@pytest.mark.unit
def test_lce_model_type_uses_model_file_directly():
    root = Path(__file__).parent / "files" / "input"
    config = Configuration(
        structure_file="fake_structure.cif",
        event_file="fake_events.json",
        model_type="lce",
        model_file=str(root / "lce.json"),
    )

    model = ConfigIO._create_model_from_config(config)
    assert isinstance(model, LocalClusterExpansion)


@pytest.mark.unit
def test_build_and_load_tabulated_model_bundle_from_entries_file(tmp_path: Path):
    root = Path(__file__).parent / "files" / "input"
    bundle = ConfigIO.build_tabulated_model_bundle_from_file(
        entries_file=str(root / "tabulated_entries.json")
    )
    assert bundle["format"] == "kmcpy.model_bundle.v1"
    assert bundle["model_type"] == "tabulated"
    assert "tabulated" in bundle

    output = tmp_path / "tabulated_model.json"
    ConfigIO.save_model_bundle(bundle, str(output))
    loaded = ConfigIO.load_model_bundle(str(output))
    assert loaded["tabulated"]["default_property"] == "barrier"


@pytest.mark.unit
def test_load_tabulated_model_bundle_validation_error(tmp_path: Path):
    invalid = tmp_path / "invalid_tabulated_bundle.json"
    invalid.write_text(
        '{"format":"kmcpy.model_bundle.v1","model_type":"tabulated","tabulated":{"entries":[]}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-empty list key 'entries'"):
        ConfigIO.load_model_bundle(str(invalid))


@pytest.mark.unit
def test_tabulated_model_type_uses_model_file_directly():
    root = Path(__file__).parent / "files" / "input"
    config = Configuration(
        structure_file="fake_structure.cif",
        event_file="fake_events.json",
        model_type="tabulated",
        model_file=str(root / "tabulated_model_bundle.json"),
    )

    model = ConfigIO._create_model_from_config(config)
    assert isinstance(model, TabulatedModel)


@pytest.mark.unit
def test_build_tabulated_model_bundle_rejects_duplicate_entries():
    entries = [
        {
            "mobile_ion_indices": [0, 1],
            "local_env_indices": [1, 2, 3],
            "occupations": [1, -1, 1, -1],
            "properties": {"barrier": 100.0},
        },
        {
            "mobile_ion_indices": [0, 1],
            "local_env_indices": [1, 2, 3],
            "occupations": [1, -1, 1, -1],
            "properties": {"barrier": 110.0},
        },
    ]

    with pytest.raises(ValueError, match="Duplicate tabulated canonical key"):
        ConfigIO.build_tabulated_model_bundle(entries=entries)
