import numpy as np
import pytest

from kmcpy.event import Event
from kmcpy.models.composite_lce_model import CompositeLCEModel
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
from kmcpy.models.site_energy import ExternalSiteEnergyModel, ZeroSiteEnergyModel
from kmcpy.simulator.kmc import KMC
from kmcpy.simulator.config import RuntimeConfig
from kmcpy.simulator.state import State
from kmcpy.units import BOLTZMANN_CONSTANT_MEV_PER_K


class FixedLCE(LocalClusterExpansion):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def compute(self, simulation_state, event):
        return self.value


class FixedDeltaSiteModel:
    MODEL_TYPE = "fixed_delta_site_model"

    def __init__(self, delta):
        self.delta = float(delta)

    def compute_delta(self, event, simulation_state):
        return self.delta


class StatefulDeltaSiteModel(FixedDeltaSiteModel):
    def __init__(self, delta):
        super().__init__(delta)
        self.initialized_occupations = None
        self.applied_occupations = []

    def initialize_state(self, *, simulation_state, event_lib=None, structure=None, config=None):
        self.initialized_occupations = list(simulation_state.occupations)

    def apply_event(self, *, event, simulation_state):
        self.applied_occupations.append(list(simulation_state.occupations))


class HookedProbabilityModel:
    def __init__(self):
        self.applied_occupations = []

    def compute_probability(self, event, runtime_config, simulation_state):
        return 1.0

    def apply_event(self, *, event, simulation_state):
        self.applied_occupations.append(list(simulation_state.occupations))


class MinimalEventLib:
    def __init__(self, event):
        self.events = [event]

    def get_dependent_events(self, event_index):
        return [0]


@pytest.fixture
def hop_event():
    return Event(mobile_ion_indices=(0, 1), local_env_indices=())


@pytest.mark.unit
def test_composite_lce_accepts_external_site_delta_model(hop_event):
    model = CompositeLCEModel(
        kra_model=FixedLCE(200.0),
        site_model=FixedDeltaSiteModel(40.0),
    )
    runtime_config = RuntimeConfig(temperature=300.0, attempt_frequency=1e13)
    state = State(occupations=[0, 1])

    probability = model.compute_probability(
        event=hop_event,
        runtime_config=runtime_config,
        simulation_state=state,
    )

    expected_barrier = 200.0 + 40.0 / 2.0
    expected = 1e13 * np.exp(
        -expected_barrier / (BOLTZMANN_CONSTANT_MEV_PER_K * 300.0)
    )
    assert np.isclose(probability, expected)


@pytest.mark.unit
def test_composite_lce_preserves_legacy_site_lce_direction_convention(hop_event):
    model = CompositeLCEModel(
        kra_model=FixedLCE(200.0),
        site_model=FixedLCE(40.0),
    )
    runtime_config = RuntimeConfig(temperature=300.0, attempt_frequency=1e13)

    forward = model.compute_probability(
        event=hop_event,
        runtime_config=runtime_config,
        simulation_state=State(occupations=[0, 1]),
    )
    backward = model.compute_probability(
        event=hop_event,
        runtime_config=runtime_config,
        simulation_state=State(occupations=[1, 0]),
    )
    inactive = model.compute_probability(
        event=hop_event,
        runtime_config=runtime_config,
        simulation_state=State(occupations=[0, 0]),
    )

    forward_barrier = 200.0 + 40.0 / 2.0
    backward_barrier = 200.0 - 40.0 / 2.0
    assert np.isclose(
        forward,
        1e13 * np.exp(-forward_barrier / (BOLTZMANN_CONSTANT_MEV_PER_K * 300.0)),
    )
    assert np.isclose(
        backward,
        1e13 * np.exp(-backward_barrier / (BOLTZMANN_CONSTANT_MEV_PER_K * 300.0)),
    )
    assert inactive == 0.0


@pytest.mark.unit
def test_external_site_energy_model_converts_ev_to_mev(hop_event):
    model = ExternalSiteEnergyModel(
        callable_ref="kmcpy.models.site_energy:constant_site_energy_delta",
        units="eV",
        kwargs={"value": 0.04},
    )

    delta = model.compute_delta(
        event=hop_event,
        simulation_state=State(occupations=[0, 1]),
    )

    assert np.isclose(delta, 40.0)


@pytest.mark.unit
def test_external_site_energy_model_roundtrip_keeps_callable(hop_event):
    model = ExternalSiteEnergyModel(
        callable_ref="kmcpy.models.site_energy:constant_site_energy_delta",
        units="meV",
        kwargs={"value": 35.0},
    )

    reloaded = ExternalSiteEnergyModel.from_dict(model.as_dict())

    assert reloaded.as_dict() == model.as_dict()
    assert reloaded.compute_delta(
        event=hop_event,
        simulation_state=State(occupations=[0, 1]),
    ) == 35.0


@pytest.mark.unit
def test_zero_site_energy_model_returns_zero(hop_event):
    model = ZeroSiteEnergyModel()

    assert model.compute_delta(
        event=hop_event,
        simulation_state=State(occupations=[0, 1]),
    ) == 0.0


@pytest.mark.unit
def test_composite_lce_delegates_stateful_site_model_hooks(hop_event):
    site_model = StatefulDeltaSiteModel(40.0)
    model = CompositeLCEModel(kra_model=FixedLCE(200.0), site_model=site_model)
    state = State(occupations=[0, 1])

    model.initialize_state(simulation_state=state)
    state.apply_event(hop_event, dt=0.0)
    model.apply_event(event=hop_event, simulation_state=state)

    assert site_model.initialized_occupations == [0, 1]
    assert site_model.applied_occupations == [[1, 0]]


@pytest.mark.unit
def test_kmc_update_commits_model_hook_after_state_update(hop_event):
    state = State(occupations=[0, 1])
    model = HookedProbabilityModel()
    kmc = KMC.__new__(KMC)
    kmc.simulation_state = state
    kmc.occ_global = state.occupations
    kmc.model = model
    kmc.event_lib = MinimalEventLib(hop_event)
    kmc.config = type("Config", (), {"runtime_config": RuntimeConfig()})()
    kmc.prob_list = np.array([1.0])
    kmc.prob_cum_list = np.array([1.0])

    KMC.update(kmc, hop_event, dt=0.1)

    assert state.occupations == [1, 0]
    assert model.applied_occupations == [[1, 0]]
