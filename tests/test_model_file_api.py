import json
from pathlib import Path

import pytest

from kmcpy.models.composite_lce_model import CompositeLCEModel
from kmcpy.models.fitting.fitter import LCEFitter
from kmcpy.models.local_barrier_model import LocalBarrierModel
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
from kmcpy.models.site_energy import (
    CallableSiteEnergyModel,
    MappedSiteEnergyModel,
)
from kmcpy.models.base import BaseModel
from kmcpy.simulator.config import Configuration


def _load_lce_with_latest_fit(lce_file: Path, fit_file: Path) -> LocalClusterExpansion:
    model = LocalClusterExpansion.from_file(str(lce_file))
    model.set_parameters(LCEFitter.from_file(str(fit_file)).model_parameters)
    return model


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
def test_composite_lce_model_joins_prepared_lce_models():
    root = Path(__file__).parent / "files" / "input"

    kra_model = _load_lce_with_latest_fit(
        root / "lce.json",
        root / "fitting_results.json",
    )
    site_model = _load_lce_with_latest_fit(
        root / "lce_site.json",
        root / "fitting_results_site.json",
    )
    model = CompositeLCEModel(site_model=site_model, kra_model=kra_model)
    model_data = model.as_dict()

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

    model = BaseModel.from_config(config)
    assert isinstance(model, LocalClusterExpansion)


@pytest.mark.unit
def test_local_barrier_model_type_is_inferred_from_model_file(tmp_path: Path):
    model_file = tmp_path / "local_barrier.json"
    LocalBarrierModel.constant_barrier(300.0).to(str(model_file))
    payload = json.loads(model_file.read_text(encoding="utf-8"))
    assert payload["@class"] == "LocalBarrierModel"
    assert "filetype" not in payload
    config = Configuration(
        structure_file="fake_structure.cif",
        event_file="fake_events.json",
        model_file=str(model_file),
    )

    model = BaseModel.from_config(config)
    assert isinstance(model, LocalBarrierModel)


@pytest.mark.unit
def test_callable_site_energy_model_type_is_inferred_from_model_file(tmp_path: Path):
    model_file = tmp_path / "callable_site_energy.json"
    CallableSiteEnergyModel(
        callable_ref="kmcpy.models.site_energy:constant_site_energy_difference",
        kwargs={"value": 10.0},
    ).to(str(model_file))
    payload = json.loads(model_file.read_text(encoding="utf-8"))
    assert payload["@class"] == "CallableSiteEnergyModel"
    assert "filetype" not in payload
    config = Configuration(
        structure_file="fake_structure.cif",
        event_file="fake_events.json",
        model_file=str(model_file),
    )

    model = BaseModel.from_config(config)
    assert isinstance(model, CallableSiteEnergyModel)


@pytest.mark.unit
def test_mapped_site_energy_model_type_is_inferred_from_model_file(tmp_path: Path):
    model_file = tmp_path / "mapped_site_energy.json"
    MappedSiteEnergyModel(
        delta_ref="tests.test_site_energy_models:smol_runtime_delta",
        delta_kwargs={"coefficients": [1.0]},
        state_mapping={0: 0, 1: 1},
    ).to(str(model_file))
    payload = json.loads(model_file.read_text(encoding="utf-8"))
    assert payload["@class"] == "MappedSiteEnergyModel"
    assert "filetype" not in payload
    config = Configuration(
        structure_file="fake_structure.cif",
        event_file="fake_events.json",
        model_file=str(model_file),
    )

    model = BaseModel.from_config(config)
    assert isinstance(model, MappedSiteEnergyModel)


@pytest.mark.unit
def test_local_barrier_exact_rules_reject_duplicate_entries():
    entries = [
        {
            "mobile_ion_indices": [0, 1],
            "local_env_indices": [1, 2, 3],
            "occupations": [1, 0, 1, 0],
            "properties": {"barrier": 100.0},
        },
        {
            "mobile_ion_indices": [0, 1],
            "local_env_indices": [1, 2, 3],
            "occupations": [1, 0, 1, 0],
            "properties": {"barrier": 110.0},
        },
    ]

    with pytest.raises(
        ValueError, match="Duplicate exact local-barrier rule"
    ):
        LocalBarrierModel.from_exact_entries(entries=entries)
