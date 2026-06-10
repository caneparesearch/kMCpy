from pathlib import Path

import pytest

from kmcpy.cli.main import main as kmcpy_main
from kmcpy.cli.sample import (
    write_sample_config,
    write_sample_model,
    write_sample_set,
    write_sample_state,
)
from kmcpy.models import BaseModel, LocalBarrierModel
from kmcpy.simulator.config import Configuration
from kmcpy.simulator.state import State


@pytest.mark.unit
def test_write_sample_config_yaml_and_json_are_parseable(tmp_path: Path):
    yaml_file = tmp_path / "input.yaml"
    json_file = tmp_path / "input.json"

    write_sample_config(
        yaml_file,
        model_file="sample_model.json",
        initial_state_file="sample_state.json",
    )
    write_sample_config(json_file)

    yaml_config = Configuration.from_file(yaml_file)
    json_config = Configuration.from_file(json_file)

    assert yaml_config.model_type == "local_barrier"
    assert yaml_config.model_file == "sample_model.json"
    assert yaml_config.initial_state_file == "sample_state.json"
    assert json_config.model_type == "local_barrier"
    assert json_config.temperature == 300.0


@pytest.mark.unit
def test_write_sample_model_creates_loadable_local_barrier_model(tmp_path: Path):
    model_file = tmp_path / "model.json"

    write_sample_model(model_file, barrier=275.0)
    model = LocalBarrierModel.from_file(str(model_file))

    assert model.default_properties == {"barrier": 275.0}


@pytest.mark.unit
def test_write_sample_state_creates_loadable_state(tmp_path: Path):
    state_file = tmp_path / "initial_state.json"

    write_sample_state(state_file, occupations=[0, 1, 0])
    state = State.from_file(str(state_file))

    assert state.occupations == [0, 1, 0]
    assert state.time == 0.0
    assert state.step == 0


@pytest.mark.unit
def test_write_sample_set_creates_linked_artifacts(tmp_path: Path):
    paths = write_sample_set(tmp_path / "sample", barrier=250.0, occupations=[0, 1])

    assert set(paths) == {"config", "model", "state"}
    assert all(path.exists() for path in paths.values())

    config = Configuration.from_file(paths["config"])
    model = BaseModel.from_config(config)
    state = State.from_file(config.initial_state_file)

    assert isinstance(model, LocalBarrierModel)
    assert model.default_properties == {"barrier": 250.0}
    assert state.occupations == [0, 1]


@pytest.mark.unit
def test_sample_commands_from_top_level_cli(tmp_path: Path):
    model_file = tmp_path / "model.json"
    state_file = tmp_path / "initial_state.json"
    sample_dir = tmp_path / "sample"

    assert kmcpy_main(["sample", "model", "--output", str(model_file)]) == 0
    assert kmcpy_main(
        [
            "sample",
            "state",
            "--output",
            str(state_file),
            "--occupations",
            "0,1,0",
        ]
    ) == 0
    assert kmcpy_main(["sample", "all", "--output-dir", str(sample_dir)]) == 0

    assert model_file.exists()
    assert State.from_file(str(state_file)).occupations == [0, 1, 0]
    assert (sample_dir / "input.yaml").exists()
    assert (sample_dir / "model.json").exists()
    assert (sample_dir / "initial_state.json").exists()


@pytest.mark.unit
def test_sample_all_help_includes_units_and_overwrite_text(capsys):
    with pytest.raises(SystemExit) as exc_info:
        kmcpy_main(["sample", "all", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "input.yaml, model.json, and initial_state.json" in output
    assert "Constant migration barrier in meV" in output
    assert "Overwrite existing sample files" in output
    assert "kmcpy sample all --output-dir kmcpy_sample" in output


@pytest.mark.unit
def test_sample_writers_do_not_overwrite_without_force(tmp_path: Path):
    model_file = tmp_path / "model.json"
    write_sample_model(model_file)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_sample_model(model_file)

    write_sample_model(model_file, force=True)
