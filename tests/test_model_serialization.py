import json
from pathlib import Path

import numpy as np
import pytest

import kmcpy.models as model_module
from kmcpy.models.composite_lce_model import CompositeLCEModel
from kmcpy.models.local_barrier_model import LocalBarrierModel
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
from kmcpy.models.site_energy import (
    ExternalSiteEnergyModel,
)
from kmcpy.structure.local_lattice_structure import LocalLatticeStructure
from pymatgen.core import Lattice, Structure


def test_local_cluster_expansion_from_file_sets_name():
    root = Path(__file__).parent / "files" / "input"
    model = LocalClusterExpansion.from_file(str(root / "lce.json"))

    payload = model.as_dict()

    assert model.name == "LocalClusterExpansion"
    assert payload["name"] == "LocalClusterExpansion"


def test_local_cluster_expansion_from_file_is_repeatable():
    root = Path(__file__).parent / "files" / "input"
    first_model = LocalClusterExpansion.from_file(str(root / "lce.json"))
    second_model = LocalClusterExpansion.from_file(str(root / "lce.json"))

    assert first_model.as_dict() == second_model.as_dict()


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


def test_lce_fit_writes_local_environment_hash_not_ordering_convention(tmp_path):
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
    orbit_fingerprints = model.get_orbit_fingerprints()
    assert orbit_fingerprints

    n_features = len(orbit_fingerprints)
    n_samples = max(4, n_features + 2)
    correlation_matrix = np.array(
        [
            [
                ((sample + 1) * (feature + 2)) % 7 + 0.1 * feature
                for feature in range(n_features)
            ]
            for sample in range(n_samples)
        ],
        dtype=float,
    )
    coefficients = np.linspace(0.1, 0.2, n_features)
    e_kra = correlation_matrix @ coefficients + 0.3
    weight = np.ones(n_samples)

    corr_file = tmp_path / "correlation_matrix.txt"
    ekra_file = tmp_path / "e_kra.txt"
    weight_file = tmp_path / "weight.txt"
    keci_file = tmp_path / "keci.txt"
    fit_results_file = tmp_path / "fitting_results.json"
    params_file = tmp_path / "lce_params.json"

    np.savetxt(corr_file, correlation_matrix)
    np.savetxt(ekra_file, e_kra)
    np.savetxt(weight_file, weight)

    params, _, _ = model.fit(
        alpha=1e-8,
        max_iter=10000,
        ekra_fname=str(ekra_file),
        keci_fname=str(keci_file),
        weight_fname=str(weight_file),
        corr_fname=str(corr_file),
        fit_results_fname=str(fit_results_file),
        lce_params_fname=str(params_file),
        lce_params_history_fname=None,
        normalize=False,
    )

    assert params.local_environment_hash == model.local_environment_hash
    assert params.orbit_fingerprints == orbit_fingerprints
    assert params.ordering_convention is None

    params_payload = json.loads(params_file.read_text(encoding="utf-8"))
    assert params_payload["local_environment_hash"] == model.local_environment_hash
    assert params_payload["orbit_fingerprints"] == orbit_fingerprints
    assert "ordering_convention" not in params_payload

    fit_results_payload = json.loads(fit_results_file.read_text(encoding="utf-8"))
    latest_row = fit_results_payload[max(fit_results_payload, key=int)]
    assert latest_row["local_environment_hash"] == model.local_environment_hash
    assert latest_row["orbit_fingerprints"] == orbit_fingerprints
    assert "ordering_convention" not in latest_row


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


def test_composite_lce_model_as_dict_uses_model_file_payload():
    root = Path(__file__).parent / "files" / "input"
    model = CompositeLCEModel.from_file(str(root / "model.json"))

    payload = model.as_dict()

    assert payload["filetype"] == "kmcpy.model_file"
    assert payload["model_type"] == "composite_lce"
    assert payload["site"]["lce"]["name"] == "LocalClusterExpansion"
    assert payload["kra"]["lce"]["name"] == "LocalClusterExpansion"


def test_composite_lce_model_from_dict_reconstructs_submodels():
    root = Path(__file__).parent / "files" / "input"
    model = CompositeLCEModel.from_file(str(root / "model.json"))

    reloaded = CompositeLCEModel.from_dict(model.as_dict())

    assert isinstance(reloaded.kra_model, LocalClusterExpansion)
    assert isinstance(reloaded.site_model, LocalClusterExpansion)
    assert reloaded.kra_model.name == "LocalClusterExpansion"
    assert reloaded.site_model.name == "LocalClusterExpansion"


def test_composite_lce_model_from_file_is_repeatable():
    root = Path(__file__).parent / "files" / "input"
    first_model = CompositeLCEModel.from_file(str(root / "model.json"))
    second_model = CompositeLCEModel.from_file(str(root / "model.json"))

    assert first_model.kra_model is not None
    assert second_model.kra_model is not None
    assert first_model.site_model is not None
    assert second_model.site_model is not None
    assert first_model.kra_model.name == second_model.kra_model.name
    assert first_model.site_model.name == second_model.site_model.name


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

    model = CompositeLCEModel.from_file(str(model_data_file))
    assert model.kra_model is not None
    assert model.site_model is None


def test_composite_lce_model_to_is_model_file_compatible(tmp_path):
    root = Path(__file__).parent / "files" / "input"
    model = CompositeLCEModel.from_file(str(root / "model.json"))

    output_model_file = tmp_path / "saved_model_file.json"
    model.to(str(output_model_file))

    loaded_model_data = json.loads(output_model_file.read_text(encoding="utf-8"))
    reloaded = CompositeLCEModel.from_file(str(output_model_file))

    assert loaded_model_data["filetype"] == "kmcpy.model_file"
    assert "orbit_fingerprints" in loaded_model_data["kra"]["parameters"]
    assert "local_environment_hash" in loaded_model_data["kra"]["parameters"]
    assert "ordering_convention" not in loaded_model_data["kra"]["parameters"]
    assert "ordering_convention" in loaded_model_data["kra"]["lce"]
    assert reloaded.kra_model is not None
    assert (
        loaded_model_data["kra"]["parameters"]["orbit_fingerprints"]
        == reloaded.kra_model.get_orbit_fingerprints()
    )
    assert (
        loaded_model_data["kra"]["parameters"]["local_environment_hash"]
        == reloaded.kra_model.local_environment_hash
    )


def test_composite_lce_model_serializes_external_site_energy_model():
    root = Path(__file__).parent / "files" / "input"
    model = CompositeLCEModel.from_file(str(root / "model.json"))
    model.site_model = ExternalSiteEnergyModel(
        callable_ref="kmcpy.models.site_energy:constant_site_energy_difference",
        units="eV",
        kwargs={"value": 0.04},
    )

    payload = model.as_dict()
    reloaded = CompositeLCEModel.from_dict(payload)

    assert payload["site"]["model_type"] == "external_site_energy"
    assert payload["site"]["delta_convention"] == "after_minus_before"
    assert isinstance(reloaded.site_model, ExternalSiteEnergyModel)
    assert reloaded.site_model.units == "eV"
    assert reloaded.site_model.kwargs == {"value": 0.04}


def test_local_barrier_model_from_file_is_repeatable(tmp_path):
    model_file = tmp_path / "local_barrier.json"
    LocalBarrierModel.constant_barrier(300.0).to(str(model_file))
    first_model = LocalBarrierModel.from_file(str(model_file))
    second_model = LocalBarrierModel.from_file(str(model_file))

    assert first_model.as_dict() == second_model.as_dict()
    assert first_model.name == "ConstantBarrierModel"


def test_exported_concrete_models_expose_pymatgen_style_constructors():
    concrete_models = [
        model_module.LocalClusterExpansion,
        model_module.CompositeLCEModel,
        model_module.LocalBarrierModel,
        model_module.ExternalSiteEnergyModel,
        model_module.MappedSiteEnergyModel,
    ]
    for model_cls in concrete_models:
        assert callable(getattr(model_cls, "from_dict"))
        assert callable(getattr(model_cls, "from_file"))
        assert callable(getattr(model_cls, "to"))
