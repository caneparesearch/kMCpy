"""Model file loading and dispatch through BaseModel."""

from pathlib import Path

import pytest
from monty.serialization import dumpfn, loadfn

from kmcpy.models import BaseModel, LocalBarrierModel, SiteEnergyModel
from kmcpy.models.base import MODEL_FILETYPE
from kmcpy.simulator.config import Configuration


def _config(model_file: Path, model_type: str = "composite_lce") -> Configuration:
    return Configuration(
        structure_file="test.cif",
        model_file=str(model_file),
        model_type=model_type,
        mobile_ion_specie="Na",
        temperature=300.0,
        attempt_frequency=1e13,
    )


def _barrier_model() -> LocalBarrierModel:
    model = LocalBarrierModel(default_barrier=300.0, name="barrier")
    model.add_state_count_rule("vacant", barrier=250.0, min_count=1)
    return model


def _site_energy_model() -> SiteEnergyModel:
    return SiteEnergyModel(
        compute_ref="kmcpy.models.site_energy:constant_site_energy_difference",
        compute_kwargs={"value": 0.02},
        name="site",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "model,model_type,payload_key",
    [
        (_barrier_model(), "local_barrier", "local_barrier"),
        (_site_energy_model(), "site_energy", "site_energy"),
    ],
)
def test_envelope_model_files_load_by_class_and_by_dispatch(
    tmp_path, model, model_type, payload_key
):
    envelope = {
        "filetype": MODEL_FILETYPE,
        "model_type": model_type,
        payload_key: model.as_dict(),
    }
    model_file = tmp_path / "model.json"
    dumpfn(envelope, model_file)

    by_class = type(model).from_file(str(model_file))
    by_dispatch = BaseModel.from_config(_config(model_file))
    assert type(by_dispatch) is type(model)
    assert by_class.as_dict() == model.as_dict()
    assert by_dispatch.as_dict() == model.as_dict()
    # Envelopes are also accepted by from_dict.
    assert type(model).from_dict(envelope).as_dict() == model.as_dict()


@pytest.mark.unit
def test_envelope_with_wrong_model_type_is_rejected(tmp_path):
    model_file = tmp_path / "model.json"
    dumpfn(
        {"filetype": MODEL_FILETYPE, "model_type": "site_energy", "site_energy": {}},
        model_file,
    )
    with pytest.raises(ValueError, match="Expected model_type 'local_barrier', got 'site_energy'"):
        LocalBarrierModel.from_file(str(model_file))


@pytest.mark.unit
def test_to_and_from_file_round_trip(tmp_path):
    for model in (_barrier_model(), _site_energy_model()):
        model_file = tmp_path / f"{model.name}.json"
        model.to(str(model_file))
        assert type(model).from_file(str(model_file)).as_dict() == model.as_dict()
        assert BaseModel.from_config(_config(model_file)).as_dict() == model.as_dict()


@pytest.mark.unit
def test_moved_model_module_falls_back_to_registered_class(tmp_path):
    payload = _barrier_model().as_dict()
    payload["@module"] = "kmcpy.old_location.local_barrier_model"
    model_file = tmp_path / "model.json"
    dumpfn(payload, model_file)

    model = BaseModel.from_config(_config(model_file))
    assert isinstance(model, LocalBarrierModel)


@pytest.mark.unit
def test_unknown_model_class_or_type_is_rejected(tmp_path):
    payload = _barrier_model().as_dict()
    payload["@module"] = "kmcpy.nowhere"
    payload["@class"] = "NoSuchModel"
    model_file = tmp_path / "model.json"
    dumpfn(payload, model_file)
    with pytest.raises(ValueError, match="Cannot import model class 'kmcpy.nowhere.NoSuchModel'"):
        BaseModel.from_config(_config(model_file))

    envelope_file = tmp_path / "envelope.json"
    dumpfn({"filetype": MODEL_FILETYPE, "model_type": "mystery"}, envelope_file)
    with pytest.raises(ValueError, match="Unknown model type 'mystery'"):
        BaseModel.from_config(_config(envelope_file))

    with pytest.raises(ValueError, match="non-empty 'model_type'"):
        dumpfn({"filetype": MODEL_FILETYPE}, envelope_file)
        BaseModel.from_config(_config(envelope_file))


@pytest.mark.unit
def test_subclass_from_config_loads_its_own_class(tmp_path):
    model_file = tmp_path / "model.json"
    _barrier_model().to(str(model_file))
    assert isinstance(LocalBarrierModel.from_config(_config(model_file)), LocalBarrierModel)
    assert loadfn(model_file, cls=None)["@class"] == "LocalBarrierModel"
