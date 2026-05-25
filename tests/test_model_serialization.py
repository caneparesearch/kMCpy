import json
from pathlib import Path

import pytest

import kmcpy.models as model_module
from kmcpy.models.composite_lce_model import CompositeLCEModel
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
from kmcpy.models.local_env_catalog import LocalEnvCatalog
from kmcpy.structure.local_lattice_structure import LocalLatticeStructure
from pymatgen.core import Lattice, Structure


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


def test_local_cluster_expansion_serializes_ordering_convention():
    structure = Structure(
        Lattice.cubic(20.0),
        ["Na", "Na", "Si"],
        [[5, 5, 5], [6, 5, 5], [7, 5, 5]],
        coords_are_cartesian=True,
    )
    local_lattice = LocalLatticeStructure(
        template_structure=structure,
        site_mapping={"Na": ["Na", "X"], "Si": ["Si", "P"]},
        center=0,
        cutoff=3.0,
        ordering_convention="nasicon_nat_commun_2022",
    )
    model = LocalClusterExpansion()
    model.build(local_lattice, cutoff_cluster=[3.0, 3.0, 0.0])

    payload = model.as_dict()
    reloaded = LocalClusterExpansion.from_dict(payload)

    assert payload["ordering_convention"]["name"] == "nasicon_nat_commun_2022"
    assert "local_environment_hash" in payload
    assert reloaded.ordering_convention.name == "nasicon_nat_commun_2022"


def test_lce_parameters_are_bound_to_orbit_fingerprints():
    structure = Structure(
        Lattice.cubic(20.0),
        ["Na", "Na", "Si"],
        [[5, 5, 5], [6, 5, 5], [7, 5, 5]],
        coords_are_cartesian=True,
    )
    local_lattice = LocalLatticeStructure(
        template_structure=structure,
        site_mapping={"Na": ["Na", "X"], "Si": ["Si", "P"]},
        center=0,
        cutoff=3.0,
    )
    model = LocalClusterExpansion()
    model.build(local_lattice, cutoff_cluster=[3.0, 3.0, 0.0])
    orbit_fingerprints = model.get_orbit_fingerprints()

    parameter_payload = {
        "keci": [0.0] * len(orbit_fingerprints),
        "empty_cluster": 0.0,
        "orbit_fingerprints": orbit_fingerprints,
        "local_environment_hash": model.local_environment_hash,
        "ordering_convention": model.ordering_convention.as_dict(),
    }
    model.set_parameters(parameter_payload)

    assert model.parameter_orbit_fingerprints == orbit_fingerprints
    assert model.parameter_local_environment_hash == model.local_environment_hash

    with pytest.warns(UserWarning, match="missing orbit_fingerprints"):
        model.set_parameters(
            {
                "keci": [0.0] * len(orbit_fingerprints),
                "empty_cluster": 0.0,
            }
        )

    short_keci = [0.0] * max(len(orbit_fingerprints) - 1, 0)
    with pytest.raises(ValueError, match="keci length"):
        model.set_parameters({"keci": short_keci, "empty_cluster": 0.0})

    wrong_fingerprints = list(orbit_fingerprints)
    wrong_fingerprints[0] = "wrong"
    with pytest.raises(ValueError, match="orbit_fingerprints"):
        bad_fingerprint_payload = dict(parameter_payload)
        bad_fingerprint_payload["orbit_fingerprints"] = wrong_fingerprints
        model.set_parameters(bad_fingerprint_payload)

    with pytest.raises(ValueError, match="local_environment_hash"):
        bad_hash_payload = dict(parameter_payload)
        bad_hash_payload["local_environment_hash"] = "wrong"
        model.set_parameters(bad_hash_payload)


def test_composite_lce_model_as_dict_with_legacy_json_inputs():
    root = Path(__file__).parent / "files" / "input"
    model = CompositeLCEModel.from_json(str(root / "model.json"))

    payload = model.as_dict()

    assert payload["site_model"]["name"] == "LocalClusterExpansion"
    assert payload["kra_model"]["name"] == "LocalClusterExpansion"



def test_composite_lce_model_from_dict_reconstructs_submodels():
    root = Path(__file__).parent / "files" / "input"
    model = CompositeLCEModel.from_file(str(root / "model.json"))

    reloaded = CompositeLCEModel.from_dict(model.as_dict())

    assert isinstance(reloaded.kra_model, LocalClusterExpansion)
    assert isinstance(reloaded.site_model, LocalClusterExpansion)
    assert reloaded.kra_model.name == "LocalClusterExpansion"
    assert reloaded.site_model.name == "LocalClusterExpansion"

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


def test_composite_lce_model_from_model_file_without_site(tmp_path):
    root = Path(__file__).parent / "files" / "input"
    lce_payload = json.loads((root / "lce.json").read_text(encoding="utf-8"))
    lce_model = LocalClusterExpansion.from_dict(lce_payload)
    orbit_fingerprints = lce_model.get_orbit_fingerprints()
    parameters = {
        "keci": [0.0] * len(orbit_fingerprints),
        "empty_cluster": 0.0,
        "orbit_fingerprints": orbit_fingerprints,
        "local_environment_hash": lce_model.local_environment_hash,
    }
    if getattr(lce_model, "ordering_convention", None) is not None:
        parameters["ordering_convention"] = lce_model.ordering_convention.as_dict()
    model_data = {
        "filetype": "kmcpy.model_file",
        "model_type": "composite_lce",
        "kra": {
            "lce": lce_payload,
            "parameters": parameters,
            "fit_metadata": {"time_stamp": 1.0, "time": "now"},
        },
    }
    model_data_file = tmp_path / "model_file_no_site.json"
    model_data_file.write_text(json.dumps(model_data), encoding="utf-8")

    model = CompositeLCEModel.from_json(str(model_data_file))
    assert model.kra_model is not None
    assert model.site_model is None


def test_composite_lce_model_to_json_is_model_file_compatible(tmp_path):
    root = Path(__file__).parent / "files" / "input"
    model = CompositeLCEModel.from_json(str(root / "model.json"))

    output_model_file = tmp_path / "saved_model_file.json"
    model.to_json(str(output_model_file))

    loaded_model_data = json.loads(output_model_file.read_text(encoding="utf-8"))
    reloaded = CompositeLCEModel.from_json(str(output_model_file))

    assert loaded_model_data["filetype"] == "kmcpy.model_file"
    assert "orbit_fingerprints" in loaded_model_data["kra"]["parameters"]
    assert "local_environment_hash" in loaded_model_data["kra"]["parameters"]
    assert "ordering_convention" in loaded_model_data["kra"]["parameters"]
    assert reloaded.kra_model is not None
    assert (
        loaded_model_data["kra"]["parameters"]["orbit_fingerprints"]
        == reloaded.kra_model.get_orbit_fingerprints()
    )
    assert (
        loaded_model_data["kra"]["parameters"]["local_environment_hash"]
        == reloaded.kra_model.local_environment_hash
    )


def test_local_env_catalog_from_file_matches_from_json():
    root = Path(__file__).parent / "files" / "input"
    from_file_model = LocalEnvCatalog.from_file(str(root / "local_env_catalog.json"))
    from_json_model = LocalEnvCatalog.from_json(str(root / "local_env_catalog.json"))

    assert from_file_model.as_dict() == from_json_model.as_dict()
    assert from_file_model.name == "LocalEnvCatalog"


def test_exported_concrete_models_expose_pymatgen_style_constructors():
    concrete_models = [
        model_module.LocalClusterExpansion,
        model_module.CompositeLCEModel,
        model_module.LocalEnvCatalog,
    ]
    for model_cls in concrete_models:
        assert callable(getattr(model_cls, "from_dict"))
        assert callable(getattr(model_cls, "from_file"))
        assert callable(getattr(model_cls, "from_json"))
        assert callable(getattr(model_cls, "to"))
