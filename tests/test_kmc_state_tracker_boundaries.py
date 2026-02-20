import types
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pytest

from kmcpy.simulator.kmc import KMC, CallbackExecutionError
from kmcpy.simulator.state import SimulationState
from kmcpy.simulator.tracker import Tracker


class _DummyLattice:
    def __init__(self):
        self.matrix = np.eye(3)

    def get_cartesian_coords(self, frac_coords):
        return np.array(frac_coords, dtype=float)


class _DummyStructure:
    def __init__(self, species_symbols):
        self.species = [types.SimpleNamespace(symbol=s) for s in species_symbols]
        self.frac_coords = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.75, 0.0, 0.0]],
            dtype=float,
        )
        self.lattice = _DummyLattice()
        self.volume = 1.0


def _make_tracker():
    config = types.SimpleNamespace(
        mobile_ion_specie="Na",
        dimension=3,
        mobile_ion_charge=1.0,
        elementary_hop_distance=1.0,
        temperature=300.0,
        attempt_frequency=1e13,
    )
    structure = _DummyStructure(["Na", "Na", "O"])
    state = SimulationState(occupations=[-1, 1, 1], time=0.0, step=0)
    return Tracker(config=config, structure=structure, initial_state=state), state


@pytest.mark.unit
def test_tracker_update_does_not_advance_state_time():
    config = types.SimpleNamespace(
        mobile_ion_specie="Na",
        dimension=3,
        mobile_ion_charge=1.0,
        elementary_hop_distance=1.0,
        temperature=300.0,
        attempt_frequency=1e13,
    )
    structure = _DummyStructure(["Na", "Na", "O"])
    state = SimulationState(occupations=[-1, 1, 1], time=0.0, step=0)
    tracker = Tracker(config=config, structure=structure, initial_state=state)
    event = types.SimpleNamespace(mobile_ion_indices=(0, 1), probability=1.0)

    tracker.update(event=event, current_occ=state.occupations, dt=0.25)

    assert state.time == 0.0
    assert int(np.sum(tracker.hop_counter)) == 1


@pytest.mark.unit
def test_kmc_run_routes_dt_to_kmc_update(monkeypatch):
    import kmcpy.simulator.kmc as kmc_module

    class FakeTracker:
        last_instance = None

        def __init__(self, config, structure, initial_state, **kwargs):
            self.config = config
            self.structure = structure
            self.state = initial_state
            self.observed_times = []
            self.received_dts = []
            FakeTracker.last_instance = self

        def update(self, event, current_occ, dt):
            self.observed_times.append(self.state.time)
            self.received_dts.append(dt)

        def update_current_pass(self, current_pass):
            self.current_pass = current_pass

        def compute_properties(self):
            return None

        def show_current_info(self):
            return None

        def write_results(self, final_occupations, label=None):
            self.final_occupations = list(final_occupations)
            self.label = label

    monkeypatch.setattr(kmc_module, "Tracker", FakeTracker)

    kmc = KMC.__new__(KMC)
    kmc.structure = _DummyStructure(["Li", "Li", "O"])
    kmc.event_lib = types.SimpleNamespace(events=[types.SimpleNamespace(mobile_ion_indices=(0, 1))])
    kmc.simulation_state = SimulationState(occupations=[-1, 1, 1], time=0.0, step=0)

    config = types.SimpleNamespace(
        name="unit-test",
        random_seed=1,
        attempt_frequency=1e13,
        temperature=300.0,
        mobile_ion_specie="Li",
        equilibration_passes=1,
        kmc_passes=2,
    )

    proposed_dts = iter([0.01, 0.02, 0.11, 0.12, 0.21, 0.22])

    def fake_propose(events):
        return events[0], next(proposed_dts)

    update_dt_calls = []

    def fake_update(event, dt=0.0):
        update_dt_calls.append(dt)
        kmc.simulation_state.time += dt
        kmc.simulation_state.step += 1

    kmc.propose = fake_propose
    kmc.update = fake_update

    tracker = kmc.run(config=config, label="unit")

    assert tracker is FakeTracker.last_instance
    assert update_dt_calls == [0.0, 0.0, 0.11, 0.12, 0.21, 0.22]
    assert tracker.received_dts == [0.11, 0.12, 0.21, 0.22]
    assert np.allclose(tracker.observed_times, [0.0, 0.11, 0.23, 0.44])
    assert np.isclose(kmc.simulation_state.time, 0.66)


@pytest.mark.unit
def test_tracker_custom_property_step_interval():
    tracker, state = _make_tracker()
    tracker.set_global_property_frequency(interval=100, time_interval=None)

    def custom_prop(sim_state, step, sim_time):
        return {"step": step, "time": sim_time, "sites": len(sim_state.occupations)}

    tracker.register_property(custom_prop, name="custom", interval=2)
    for step in range(1, 6):
        state.step = step
        state.time = step * 0.1
        tracker.maybe_compute_properties(step=step, sim_time=state.time)

    records = tracker.get_custom_results("custom")
    assert [record["step"] for record in records] == [2, 4]
    assert records[-1]["value"]["sites"] == 3


@pytest.mark.unit
def test_tracker_custom_property_time_interval():
    tracker, state = _make_tracker()
    tracker.set_global_property_frequency(interval=100, time_interval=None)

    tracker.register_property(
        lambda sim_state, step, sim_time: (step, sim_time, sim_state.step),
        name="time_prop",
        time_interval=1.0,
    )
    for step in range(1, 7):
        state.step = step
        state.time = step * 0.4
        tracker.maybe_compute_properties(step=step, sim_time=state.time)

    records = tracker.get_custom_results("time_prop")
    assert [record["step"] for record in records] == [3, 6]


@pytest.mark.unit
def test_tracker_builtin_disable_emits_nan():
    tracker, state = _make_tracker()
    tracker.disable_builtin_property("msd")
    tracker.set_global_property_frequency(interval=1, time_interval=None)

    state.step = 1
    state.time = 1.0
    tracker.maybe_compute_properties(step=1, sim_time=1.0)

    assert np.isnan(tracker.results["msd"][-1])
    assert "msd" in tracker.results


@pytest.mark.unit
def test_tracker_callback_error_handling():
    tracker, state = _make_tracker()
    tracker.set_global_property_frequency(interval=1, time_interval=None)

    def bad_callback(sim_state, step, sim_time):
        raise RuntimeError("boom")

    called = {"count": 0}

    def on_error(exc, sim_state, step, sim_time):
        called["count"] += 1
        return True

    tracker.register_property(
        bad_callback,
        name="recoverable",
        interval=1,
        on_error=on_error,
    )
    state.step = 1
    state.time = 1.0
    tracker.maybe_compute_properties(step=1, sim_time=1.0)
    assert called["count"] == 1

    tracker.register_property(bad_callback, name="fatal", interval=1)
    state.step = 2
    state.time = 2.0
    with pytest.raises(CallbackExecutionError):
        tracker.maybe_compute_properties(step=2, sim_time=2.0)


@pytest.mark.unit
def test_tracker_write_results_includes_custom_records(tmp_path):
    tracker, state = _make_tracker()
    tracker.set_global_property_frequency(interval=1, time_interval=None)

    tracker.register_property(lambda *_: {"a": 1, "b": [1, 2]}, name="custom", interval=1)
    state.step = 1
    state.time = 1.0
    tracker.maybe_compute_properties(step=1, sim_time=1.0)

    old_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        tracker.write_results(current_occupation=state.occupations, label="unit")
    finally:
        os.chdir(old_cwd)

    output_file = tmp_path / "custom_results_unit.json.gz"
    assert output_file.exists()
    with gzip.open(output_file, "rt", encoding="utf-8") as fhandle:
        payload = json.load(fhandle)
    assert payload[0]["name"] == "custom"
    assert payload[0]["step"] == 1


@pytest.mark.unit
def test_kmc_attachment_management():
    kmc = KMC.__new__(KMC)
    kmc._ensure_property_state()

    def custom_prop(state, step, sim_time):
        return step + sim_time

    name = kmc.attach(custom_prop, interval=3, name="p1")
    assert name == "p1"
    assert kmc.list_attachments() == ["p1"]

    kmc.set_property_frequency(interval=5, time_interval=None)
    assert kmc._property_frequency_interval == 5

    kmc.disable_property("msd")
    assert "msd" in kmc.list_builtin_properties()

    kmc.detach("p1")
    assert kmc.list_attachments() == []

    kmc.attach(custom_prop, interval=2, name="p2")
    kmc.clear_attachments()
    assert kmc.list_attachments() == []
