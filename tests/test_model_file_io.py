from pathlib import Path

import pytest

from kmcpy.io.model_file import (
    build_model_file_from_legacy_files,
    build_tabulated_model_file,
    build_tabulated_model_file_from_entries_file,
    load_model_file,
    save_model_file,
)
from kmcpy.simulator.components import create_model_from_config
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
from kmcpy.models.tabulated_model import TabulatedModel
from kmcpy.simulator.config import Configuration


@pytest.mark.unit
def test_load_model_file_validation_error(tmp_path: Path):
    invalid = tmp_path / "invalid_model_file.json"
    invalid.write_text('{"format": "wrong", "model_type": "composite_lce"}', encoding="utf-8")

    with pytest.raises(ValueError, match="kmcpy.model_file"):
        load_model_file(str(invalid))


@pytest.mark.unit
def test_build_model_file_from_legacy_files():
    root = Path(__file__).parent / "files" / "input"

    model_data = build_model_file_from_legacy_files(
        kra_lce=str(root / "lce.json"),
        kra_fit=str(root / "fitting_results.json"),
        site_lce=str(root / "lce_site.json"),
        site_fit=str(root / "fitting_results_site.json"),
    )

    assert model_data["format"] == "kmcpy.model_file"
    assert model_data["model_type"] == "composite_lce"
    assert model_data["kra"]["parameters"]["keci"]
    assert model_data["site"]["parameters"]["empty_cluster"] is not None


@pytest.mark.unit
def test_simulation_config_rejects_unknown_model_fields():
    with pytest.raises(ValueError, match="Unknown parameters"):
        Configuration.from_dict(
            {
                "structure_file": "test.cif",
                "event_file": "event.json",
                "cluster_expansion_file": "lce.json",
                "fitting_results_file": "fit.json",
            }
        )


@pytest.mark.unit
def test_simulation_config_from_dict_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown parameters"):
        Configuration.from_dict(
            {
                "structure_file": "test.cif",
                "event_file": "event.json",
                "model_file": "model.json",
                "temperaturee": 300.0,
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

    model = create_model_from_config(config)
    assert isinstance(model, LocalClusterExpansion)


@pytest.mark.unit
def test_build_and_load_tabulated_model_file_from_entries_file(tmp_path: Path):
    root = Path(__file__).parent / "files" / "input"
    model_data = build_tabulated_model_file_from_entries_file(
        entries_file=str(root / "tabulated_entries.json")
    )
    assert model_data["format"] == "kmcpy.model_file"
    assert model_data["model_type"] == "tabulated"
    assert "tabulated" in model_data

    output = tmp_path / "tabulated_model.json"
    save_model_file(model_data, str(output))
    loaded = load_model_file(str(output))
    assert loaded["tabulated"]["default_property"] == "barrier"


@pytest.mark.unit
def test_load_tabulated_model_file_validation_error(tmp_path: Path):
    invalid = tmp_path / "invalid_tabulated_model_file.json"
    invalid.write_text(
        '{"format":"kmcpy.model_file","model_type":"tabulated","tabulated":{"entries":[]}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-empty list key 'entries'"):
        load_model_file(str(invalid))


@pytest.mark.unit
def test_tabulated_model_type_uses_model_file_directly():
    root = Path(__file__).parent / "files" / "input"
    config = Configuration(
        structure_file="fake_structure.cif",
        event_file="fake_events.json",
        model_type="tabulated",
        model_file=str(root / "tabulated_model_file.json"),
    )

    model = create_model_from_config(config)
    assert isinstance(model, TabulatedModel)


@pytest.mark.unit
def test_build_tabulated_model_file_rejects_duplicate_entries():
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
        build_tabulated_model_file(entries=entries)
