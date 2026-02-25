from pathlib import Path

import pytest

from kmcpy.io.config_io import ConfigIO
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
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
