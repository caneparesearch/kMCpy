from pathlib import Path

import numpy as np
import pytest

from kmcpy.event import Event
from kmcpy.models.base import BaseModel
from kmcpy.models.local_barrier_model import LocalBarrierModel
from kmcpy.simulator.config import Configuration, RuntimeConfig
from kmcpy.simulator.state import State


@pytest.fixture
def event():
    return Event(mobile_ion_indices=(0, 1), local_env_indices=(1, 2, 3))


@pytest.mark.unit
def test_local_barrier_constant_barrier(event):
    model = LocalBarrierModel.constant_barrier(300.0)
    state = State(occupations=[-1, 1, 1, -1])

    assert model.compute(simulation_state=state, event=event) == 300.0

    runtime_config = RuntimeConfig(temperature=300.0, attempt_frequency=1e13)
    probability = model.compute_probability(
        event=event,
        runtime_config=runtime_config,
        simulation_state=state,
    )
    expected = 1e13 * np.exp(-300.0 / (8.617333262145e-2 * 300.0))
    assert np.isclose(probability, expected)


@pytest.mark.unit
def test_local_barrier_state_count_rule_matches_local_environment():
    model = LocalBarrierModel(default_barrier=300.0)
    model.add_state_count_rule(
        name="crowded",
        sites="local_env",
        state="occupied",
        min_count=3,
        barrier=450.0,
    )
    state = State(occupations=[-1, 1, -1, -1, -1])
    event = Event(mobile_ion_indices=(0, 1), local_env_indices=(2, 3, 4))

    assert model.compute(simulation_state=state, event=event) == 450.0


@pytest.mark.unit
def test_local_barrier_species_count_rule_matches_species_threshold():
    model = LocalBarrierModel(
        default_barrier=300.0,
        site_species={
            1: {-1: "P", 1: "Si"},
            2: {-1: "Si", 1: "P"},
            3: {-1: "Si", 1: "P"},
            4: {-1: "Al", 1: "Si"},
        },
    )
    model.add_species_count_rule(
        name="si_rich",
        sites="local_env",
        species="Si",
        min_count=4,
        barrier=420.0,
    )
    state = State(occupations=[-1, 1, -1, -1, 1])
    event = Event(mobile_ion_indices=(0, 1), local_env_indices=(1, 2, 3, 4))

    assert model.compute(simulation_state=state, event=event) == 420.0


@pytest.mark.unit
def test_local_barrier_pattern_rule_supports_wildcards(event):
    model = LocalBarrierModel(default_barrier=300.0)
    model.add_pattern_rule(
        name="pattern",
        sites="canonical",
        pattern=["occupied", "vacant", "*", "occupied"],
        barrier=250.0,
    )
    state = State(occupations=[-1, 1, 1, -1])

    assert model.compute(simulation_state=state, event=event) == 250.0


@pytest.mark.unit
def test_local_barrier_exact_rule_from_event_state(event):
    state = State(occupations=[-1, 1, 1, -1])
    exact_rule = LocalBarrierModel.entry_from_event_state(
        event=event,
        state=state,
        properties={"barrier": 260.0},
    )
    model = LocalBarrierModel(rules=[exact_rule])

    assert model.compute(simulation_state=state, event=event) == 260.0

    with pytest.raises(KeyError, match="No local barrier rule matched"):
        model.compute(
            simulation_state=State(occupations=[1, -1, 1, -1]),
            event=event,
        )


@pytest.mark.unit
def test_local_barrier_from_exact_entries_matches_payload():
    entries = [
        {
            "mobile_ion_indices": [0, 1],
            "local_env_indices": [1, 2, 3],
            "occupations": [1, -1, 1, -1],
            "properties": {"barrier": 250.0},
        }
    ]
    model = LocalBarrierModel.from_exact_entries(entries)
    state = State(occupations=[1, -1, 1, -1])
    event = Event(mobile_ion_indices=(0, 1), local_env_indices=(1, 2, 3))

    assert model.compute(simulation_state=state, event=event) == 250.0


@pytest.mark.unit
def test_local_barrier_model_file_roundtrip_and_config_dispatch(tmp_path: Path):
    model = LocalBarrierModel.constant_barrier(275.0)
    model_file = tmp_path / "local_barrier.json"
    model.to(str(model_file))

    loaded = LocalBarrierModel.from_file(str(model_file))
    assert loaded.default_properties["barrier"] == 275.0

    config = Configuration(
        structure_file="fake_structure.cif",
        event_file="fake_events.json",
        model_file=str(model_file),
    )
    dispatched = BaseModel.from_config(config)
    assert isinstance(dispatched, LocalBarrierModel)
