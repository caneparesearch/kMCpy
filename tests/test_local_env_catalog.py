from pathlib import Path

import numpy as np
import pytest

from kmcpy.event import Event
from kmcpy.models.local_env_catalog import LocalEnvCatalogEntry, LocalEnvCatalog
from kmcpy.simulator.config import RuntimeConfig
from kmcpy.simulator.state import State


@pytest.fixture
def local_env_catalog_entries():
    return [
        {
            "mobile_ion_indices": [0, 1],
            "local_env_indices": [1, 2, 3],
            "occupations": [1, -1, 1, -1],
            "properties": {"barrier": 250.0, "energy": 1.2},
        },
        {
            "mobile_ion_indices": [0, 1],
            "local_env_indices": [1, 2, 3],
            "occupations": [-1, 1, 1, -1],
            "properties": {"barrier": 300.0, "energy": 2.5},
        },
    ]


@pytest.mark.unit
def test_local_env_catalog_lookup_hit_and_property_selection(local_env_catalog_entries):
    model = LocalEnvCatalog(entries=local_env_catalog_entries, default_property="barrier")
    state = State(occupations=[1, -1, 1, -1])
    event = Event(mobile_ion_indices=(0, 1), local_env_indices=(1, 2, 3))

    barrier = model.compute(simulation_state=state, event=event)
    energy = model.compute(simulation_state=state, event=event, property_name="energy")

    assert barrier == 250.0
    assert energy == 1.2


@pytest.mark.unit
def test_local_env_catalog_lookup_miss_raises_key_error(local_env_catalog_entries):
    model = LocalEnvCatalog(entries=local_env_catalog_entries)
    state = State(occupations=[1, 1, -1, -1])
    event = Event(mobile_ion_indices=(0, 1), local_env_indices=(1, 2, 3))

    with pytest.raises(KeyError, match="No local-environment catalog entry found"):
        model.compute(simulation_state=state, event=event)


@pytest.mark.unit
def test_local_env_catalog_probability_uses_arrhenius(local_env_catalog_entries):
    model = LocalEnvCatalog(entries=local_env_catalog_entries, probability_property="barrier")
    state = State(occupations=[1, -1, 1, -1])
    event = Event(mobile_ion_indices=(0, 1), local_env_indices=(1, 2, 3))
    runtime_config = RuntimeConfig(temperature=300.0, attempt_frequency=1e13)

    probability = model.compute_probability(
        event=event,
        runtime_config=runtime_config,
        simulation_state=state,
    )
    expected = 1.0 * 1e13 * np.exp(-250.0 / (8.617333262145e-2 * 300.0))
    assert np.isclose(probability, expected)


@pytest.mark.unit
def test_local_env_catalog_rejects_duplicate_canonical_keys():
    entries = [
        {
            "mobile_ion_indices": [0, 1],
            "local_env_indices": [1, 2, 3],
            "occupations": [1, -1, 1, -1],
            "properties": {"barrier": 250.0},
        },
        {
            "mobile_ion_indices": [0, 1],
            "local_env_indices": [1, 2, 3],
            "occupations": [1, -1, 1, -1],
            "properties": {"barrier": 300.0},
        },
    ]

    with pytest.raises(ValueError, match="Duplicate local-environment catalog canonical key"):
        LocalEnvCatalog(entries=entries)


@pytest.mark.unit
def test_local_env_catalog_roundtrip_direct_json(tmp_path: Path, local_env_catalog_entries):
    model = LocalEnvCatalog(entries=local_env_catalog_entries)
    output = tmp_path / "local_env_catalog.json"
    model.to_json(str(output))

    reloaded = LocalEnvCatalog.from_file(str(output))
    state = State(occupations=[1, -1, 1, -1])
    event = Event(mobile_ion_indices=(0, 1), local_env_indices=(1, 2, 3))

    assert reloaded.compute(simulation_state=state, event=event) == 250.0


@pytest.mark.unit
def test_local_env_catalog_fit_is_not_supported(local_env_catalog_entries):
    model = LocalEnvCatalog(entries=local_env_catalog_entries)
    with pytest.raises(NotImplementedError, match="does not support fit"):
        model.fit()


@pytest.mark.unit
def test_local_env_catalog_from_entries_constructor(local_env_catalog_entries):
    model = LocalEnvCatalog.from_entries(entries=local_env_catalog_entries)
    state = State(occupations=[1, -1, 1, -1])
    event = Event(mobile_ion_indices=(0, 1), local_env_indices=(1, 2, 3))
    assert model.compute(simulation_state=state, event=event) == 250.0


@pytest.mark.unit
def test_local_env_catalog_add_entry_accepts_local_env_catalog_entry():
    model = LocalEnvCatalog.from_entries(
        entries=[
            {
                "mobile_ion_indices": [0, 1],
                "local_env_indices": [1, 2, 3],
                "occupations": [1, -1, 1, -1],
                "properties": {"barrier": 250.0},
            }
        ]
    )
    model.add_entry(
        LocalEnvCatalogEntry.from_dict(
            {
                "mobile_ion_indices": [0, 1],
                "local_env_indices": [1, 2, 3],
                "occupations": [-1, 1, 1, -1],
                "properties": {"barrier": 300.0},
            }
        )
    )

    state = State(occupations=[-1, 1, 1, -1])
    event = Event(mobile_ion_indices=(0, 1), local_env_indices=(1, 2, 3))
    assert model.compute(simulation_state=state, event=event) == 300.0
