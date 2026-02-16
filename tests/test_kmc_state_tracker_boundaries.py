import types

import numpy as np
import pytest

from kmcpy.simulator.kmc import KMC
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
