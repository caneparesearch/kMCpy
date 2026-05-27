import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from kmcpy.event import Event
from kmcpy.models.composite_lce_model import CompositeLCEModel
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
from kmcpy.models.site_energy import SiteEnergyModel
from kmcpy.simulator.kmc import KMC
from kmcpy.simulator.config import RuntimeConfig
from kmcpy.simulator.state import State
from kmcpy.structure.active_site_order import ActiveSiteOrder
from kmcpy.units import BOLTZMANN_CONSTANT_MEV_PER_K


class FixedLCE(LocalClusterExpansion):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def compute(self, simulation_state, event):
        return self.value


class FixedSiteDifferenceModel:
    MODEL_TYPE = "fixed_site_difference_model"

    def __init__(self, energy_difference):
        self.energy_difference = float(energy_difference)

    def compute(self, event, simulation_state):
        return self.energy_difference


class StatefulSiteDifferenceModel(FixedSiteDifferenceModel):
    def __init__(self, energy_difference):
        super().__init__(energy_difference)
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


class FakeSmolProcessor:
    def __init__(self):
        self.coefs = np.array([0.5])
        self.calls = []

    def compute_feature_vector_change(self, occupancy, flips):
        self.calls.append((occupancy.copy(), list(flips)))
        delta = sum(new_code - int(occupancy[index]) for index, new_code in flips)
        return np.array([float(delta)])


class FakeCleaseEvaluator:
    def __init__(self, energy=5.0, proposed_delta=0.025):
        self.energy = float(energy)
        self.proposed_delta = float(proposed_delta)
        self.proposed_changes = []
        self.applied_changes = []

    def get_energy(self, applied_changes=None):
        return self.energy

    def get_energy_given_change(self, system_changes):
        self.proposed_changes.append(list(system_changes))
        return self.energy + self.proposed_delta

    def apply_system_changes(self, system_changes, keep=False):
        self.applied_changes.append((list(system_changes), keep))
        if keep:
            self.energy += self.proposed_delta


def smol_runtime_delta(runtime, external_occupation, changes, coefficients, **kwargs):
    flips = [change.as_flip() for change in changes]
    delta_features = runtime.compute_feature_vector_change(external_occupation, flips)
    return float(np.dot(delta_features, coefficients))


def clease_runtime_delta(runtime, external_occupation, changes, **kwargs):
    system_changes = [change.as_system_change_tuple() for change in changes]
    return runtime.get_energy_given_change(system_changes) - runtime.get_energy()


def clease_runtime_apply(runtime, external_occupation, changes, **kwargs):
    system_changes = [change.as_system_change_tuple() for change in changes]
    runtime.apply_system_changes(system_changes, keep=True)


@pytest.fixture
def hop_event():
    return Event(mobile_ion_indices=(0, 1), local_env_indices=())


def _active_site_order():
    structure = Structure(
        Lattice.cubic(4.0),
        ["Na", "Na"],
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
    )
    return ActiveSiteOrder.from_structure_and_mapping(
        structure,
        {"Na": ["Na", "X"]},
    )


@pytest.mark.unit
def test_composite_lce_accepts_external_site_difference_model(hop_event):
    model = CompositeLCEModel(
        kra_model=FixedLCE(200.0),
        site_model=FixedSiteDifferenceModel(40.0),
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
def test_composite_lce_applies_event_direction_to_site_lce_difference(hop_event):
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
def test_site_energy_model_converts_ev_to_mev(hop_event):
    model = SiteEnergyModel(
        compute_ref="kmcpy.models.site_energy:constant_site_energy_difference",
        units="eV",
        compute_kwargs={"value": 0.04},
    )

    delta = model.compute(
        event=hop_event,
        simulation_state=State(occupations=[0, 1]),
    )

    assert np.isclose(delta, 40.0)


@pytest.mark.unit
def test_site_energy_model_roundtrip_keeps_callable(hop_event):
    model = SiteEnergyModel(
        compute_ref="kmcpy.models.site_energy:constant_site_energy_difference",
        units="meV",
        compute_kwargs={"value": 35.0},
    )

    reloaded = SiteEnergyModel.from_dict(model.as_dict())

    assert reloaded.as_dict() == model.as_dict()
    assert reloaded.compute(
        event=hop_event,
        simulation_state=State(occupations=[0, 1]),
    ) == 35.0


@pytest.mark.unit
def test_composite_lce_delegates_stateful_site_model_hooks(hop_event):
    site_model = StatefulSiteDifferenceModel(40.0)
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
    kmc.model = model
    kmc.event_lib = MinimalEventLib(hop_event)
    kmc.config = type("Config", (), {"runtime_config": RuntimeConfig()})()
    kmc.prob_list = np.array([1.0])
    kmc.prob_cum_list = np.array([1.0])

    KMC.update(kmc, hop_event, dt=0.1)

    assert state.occupations == [1, 0]
    assert kmc.simulation_state.occupations == [1, 0]
    assert model.applied_occupations == [[1, 0]]


@pytest.mark.unit
def test_site_energy_model_uses_cached_occupation_and_local_flips(hop_event):
    processor = FakeSmolProcessor()
    model = SiteEnergyModel(
        runtime=processor,
        compute_fn=smol_runtime_delta,
        compute_kwargs={"coefficients": np.array([0.5])},
        site_mapping={0: 2, 1: 4},
        state_mapping_by_site={
            0: {0: 10, 1: 20},
            1: {0: 100, 1: 200},
        },
        external_size=6,
        external_dtype="int32",
        units="eV",
    )
    state = State(occupations=[0, 1])

    model.initialize_state(simulation_state=state)
    assert np.array_equal(model._site_lookup, np.array([2, 4]))
    assert model._state_lookup_by_site is not None

    # Runtime evaluation should use cached lookup arrays, not mapping dicts.
    model.site_mapping = {}
    model.state_mapping_by_site = {}
    delta = model.compute(event=hop_event, simulation_state=state)

    assert np.array_equal(
        model.external_occupation,
        np.array([0, 0, 10, 0, 200, 0], dtype=np.int32),
    )
    assert len(processor.calls) == 1
    occupancy_seen, flips = processor.calls[0]
    assert np.array_equal(
        occupancy_seen,
        np.array([0, 0, 10, 0, 200, 0], dtype=np.int32),
    )
    assert flips == [(2, 20), (4, 100)]
    assert np.isclose(delta, -45000.0)

    state.apply_event(hop_event, dt=0.0)
    model.apply_event(event=hop_event, simulation_state=state)

    assert np.array_equal(
        model.external_occupation,
        np.array([0, 0, 20, 0, 100, 0], dtype=np.int32),
    )


@pytest.mark.unit
def test_site_energy_model_records_site_order_hashes(hop_event):
    index_map = _active_site_order()
    model = SiteEnergyModel(
        compute_fn=smol_runtime_delta,
        compute_kwargs={"coefficients": np.array([1.0])},
        site_mapping={0: 4, 1: 2},
        active_site_order=index_map,
        external_size=5,
    )

    payload = model.as_dict()
    reloaded = SiteEnergyModel.from_dict(payload)

    assert payload["active_site_order"]["fingerprint"] == index_map.fingerprint
    assert payload["active_site_order_hash"] == index_map.fingerprint
    assert payload["external_site_order_hash"] == model.external_site_order_hash
    assert reloaded.active_site_order_hash == index_map.fingerprint
    assert reloaded.external_site_order_hash == model.external_site_order_hash


@pytest.mark.unit
def test_site_energy_model_rejects_wrong_active_site_order(hop_event):
    index_map = _active_site_order()
    model = SiteEnergyModel(
        compute_fn=smol_runtime_delta,
        compute_kwargs={"coefficients": np.array([1.0])},
        site_mapping={0: 0, 1: 1},
        active_site_order=index_map,
    )

    with pytest.raises(ValueError, match="active-site order contains"):
        model.initialize_state(simulation_state=State(occupations=[0]))


@pytest.mark.unit
def test_composite_lce_forwards_active_site_order_to_site_energy_model(hop_event):
    index_map = _active_site_order()
    site_model = SiteEnergyModel(
        compute_fn=smol_runtime_delta,
        compute_kwargs={"coefficients": np.array([1.0])},
        site_mapping={0: 0, 1: 1},
        active_site_order=None,
    )
    model = CompositeLCEModel(kra_model=FixedLCE(200.0), site_model=site_model)

    model.initialize_state(
        simulation_state=State(occupations=[0, 1]),
        event_lib=MinimalEventLib(hop_event),
        active_site_order=index_map,
    )

    assert site_model.active_site_order_hash == index_map.fingerprint


@pytest.mark.unit
def test_site_energy_model_can_commit_runtime_object(hop_event):
    evaluator = FakeCleaseEvaluator(energy=5.0, proposed_delta=0.025)
    model = SiteEnergyModel(
        runtime=evaluator,
        compute_fn=clease_runtime_delta,
        apply_fn=clease_runtime_apply,
        site_mapping={0: 3, 1: 5},
        state_mapping={0: "Li", 1: "X"},
        external_size=6,
        units="eV",
    )
    state = State(occupations=[0, 1])

    model.initialize_state(simulation_state=state)
    delta = model.compute(event=hop_event, simulation_state=state)

    assert np.isclose(delta, 25.0)
    proposed = evaluator.proposed_changes[0]
    assert proposed == [
        (3, "Li", "X"),
        (5, "X", "Li"),
    ]

    state.apply_event(hop_event, dt=0.0)
    model.apply_event(event=hop_event, simulation_state=state)

    applied, keep = evaluator.applied_changes[0]
    assert keep is True
    assert applied == [
        (3, "Li", "X"),
        (5, "X", "Li"),
    ]
    assert np.array_equal(
        model.external_occupation,
        np.array([0, 0, 0, "X", 0, "Li"], dtype=object),
    )


@pytest.mark.unit
def test_site_energy_model_validates_site_mapping_coverage(hop_event):
    model = SiteEnergyModel(
        compute_fn=smol_runtime_delta,
        compute_kwargs={"coefficients": np.array([1.0])},
        site_mapping={0: 2},
        external_size=3,
    )

    with pytest.raises(ValueError, match="site_mapping does not cover"):
        model.initialize_state(simulation_state=State(occupations=[0, 1]))


@pytest.mark.unit
def test_site_energy_model_validates_event_state_mappings(hop_event):
    model = SiteEnergyModel(
        compute_fn=smol_runtime_delta,
        compute_kwargs={"coefficients": np.array([1.0])},
        state_mapping_by_site={
            0: {0: 10},
            1: {1: 20},
        },
    )

    with pytest.raises(ValueError, match="occupation mapping is incompatible"):
        model.initialize_state(
            simulation_state=State(occupations=[0, 1]),
            event_lib=MinimalEventLib(hop_event),
        )
