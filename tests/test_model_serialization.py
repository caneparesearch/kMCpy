import json
from pathlib import Path

import kmcpy.models as model_module
from kmcpy.models.composite_lce_model import CompositeLCEModel
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
from kmcpy.io.config_io import ConfigIO


def test_local_cluster_expansion_from_legacy_json_sets_name():
    root = Path(__file__).parent / "files" / "input"
    model = LocalClusterExpansion.from_json(str(root / "lce.json"))

    payload = model.as_dict()

    assert model.name == "LocalClusterExpansion"
    assert payload["name"] == "LocalClusterExpansion"


def test_local_cluster_expansion_from_file_matches_from_json():
    root = Path(__file__).parent / "files" / "input"
    from_file_model = LocalClusterExpansion.from_file(str(root / "lce.json"))
    from_json_model = LocalClusterExpansion.from_json(str(root / "lce.json"))

    assert from_file_model.as_dict() == from_json_model.as_dict()


def test_composite_lce_model_as_dict_with_legacy_json_inputs():
    root = Path(__file__).parent / "files" / "input"
    model = CompositeLCEModel.from_json(str(root / "model.json"))

    payload = model.as_dict()

    assert payload["site_model"]["name"] == "LocalClusterExpansion"
    assert payload["kra_model"]["name"] == "LocalClusterExpansion"


def test_composite_lce_model_from_file_matches_from_json():
    root = Path(__file__).parent / "files" / "input"
    from_file_model = CompositeLCEModel.from_file(str(root / "model.json"))
    from_json_model = CompositeLCEModel.from_json(str(root / "model.json"))

    assert from_file_model.kra_model is not None
    assert from_json_model.kra_model is not None
    assert from_file_model.site_model is not None
    assert from_json_model.site_model is not None
    assert from_file_model.kra_model.name == from_json_model.kra_model.name
    assert from_file_model.site_model.name == from_json_model.site_model.name


def test_composite_lce_model_from_bundle_without_site(tmp_path):
    root = Path(__file__).parent / "files" / "input"
    model_bundle = {
        "format": "kmcpy.model_bundle.v1",
        "model_type": "composite_lce",
        "kra": {
            "lce": json.loads((root / "lce.json").read_text(encoding="utf-8")),
            "parameters": {"keci": [1.0], "empty_cluster": 0.0},
            "fit_metadata": {"time_stamp": 1.0, "time": "now"},
        },
    }
    bundle_file = tmp_path / "model_bundle_no_site.json"
    bundle_file.write_text(json.dumps(model_bundle), encoding="utf-8")

    model = CompositeLCEModel.from_json(str(bundle_file))
    assert model.kra_model is not None
    assert model.site_model is None


def test_composite_lce_model_to_json_is_bundle_compatible(tmp_path):
    root = Path(__file__).parent / "files" / "input"
    model = CompositeLCEModel.from_json(str(root / "model.json"))

    output_bundle = tmp_path / "saved_bundle.json"
    model.to_json(str(output_bundle))

    loaded_bundle = ConfigIO.load_model_bundle(str(output_bundle))
    reloaded = CompositeLCEModel.from_json(str(output_bundle))

    assert loaded_bundle["format"] == "kmcpy.model_bundle.v1"
    assert reloaded.kra_model is not None


def test_exported_concrete_models_expose_pymatgen_style_constructors():
    concrete_models = [
        model_module.LocalClusterExpansion,
        model_module.CompositeLCEModel,
    ]
    for model_cls in concrete_models:
        assert callable(getattr(model_cls, "from_dict"))
        assert callable(getattr(model_cls, "from_file"))
        assert callable(getattr(model_cls, "from_json"))
