from pathlib import Path

import pytest

from kmcpy.models.composite_lce_model import CompositeLCEModel
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
from kmcpy.models.local_env_catalog import LocalEnvCatalog
from kmcpy.simulator.components import create_model_from_config
from kmcpy.simulator.config import Configuration


@pytest.mark.unit
def test_composite_model_file_validation_error(tmp_path: Path):
    invalid = tmp_path / "invalid_model_file.json"
    invalid.write_text(
        '{"filetype": "wrong", "model_type": "composite_lce"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="kmcpy.model_file"):
        CompositeLCEModel.from_file(str(invalid))


@pytest.mark.unit
def test_composite_lce_model_from_legacy_files():
    root = Path(__file__).parent / "files" / "input"

    model = CompositeLCEModel.from_legacy_files(
        kra_lce=str(root / "lce.json"),
        kra_fit=str(root / "fitting_results.json"),
        site_lce=str(root / "lce_site.json"),
        site_fit=str(root / "fitting_results_site.json"),
    )
    model_data = model.to_model_file_dict()

    assert model_data["filetype"] == "kmcpy.model_file"
    assert model_data["model_type"] == "composite_lce"
    assert model_data["kra"]["parameters"]["keci"]
    assert model_data["site"]["parameters"]["empty_cluster"] is not None


@pytest.mark.unit
def test_simulation_config_rejects_unknown_model_fields():
    with pytest.raises(ValueError, match="Unknown configuration fields"):
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
    with pytest.raises(ValueError, match="Unknown configuration fields"):
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
def test_local_env_catalog_from_raw_entries_writes_model_file(tmp_path: Path):
    root = Path(__file__).parent / "files" / "input"
    model = LocalEnvCatalog.from_file(str(root / "local_env_catalog_entries.json"))

    output = tmp_path / "local_env_catalog.json"
    model.to(str(output))
    loaded = LocalEnvCatalog.from_file(str(output))

    assert loaded.default_property == "barrier"
    assert loaded.to_model_file_dict()["filetype"] == "kmcpy.model_file"


@pytest.mark.unit
def test_local_env_catalog_file_validation_error(tmp_path: Path):
    invalid = tmp_path / "invalid_local_env_catalog.json"
    invalid.write_text(
        '{"filetype":"kmcpy.model_file","model_type":"local_env_catalog","local_env_catalog":{"entries":[]}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-empty list"):
        LocalEnvCatalog.from_file(str(invalid))


@pytest.mark.unit
def test_model_type_is_inferred_from_model_file():
    root = Path(__file__).parent / "files" / "input"
    config = Configuration(
        structure_file="fake_structure.cif",
        event_file="fake_events.json",
        model_file=str(root / "local_env_catalog.json"),
    )

    model = create_model_from_config(config)
    assert isinstance(model, LocalEnvCatalog)


@pytest.mark.unit
def test_local_env_catalog_from_entries_rejects_duplicate_entries():
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

    with pytest.raises(
        ValueError, match="Duplicate local-environment catalog canonical key"
    ):
        LocalEnvCatalog.from_entries(entries=entries)
