"""Tests for caches used on the per-step KMC rate-update path."""

import numpy as np
import pytest

from kmcpy.event import Event, EventLib
from kmcpy.models.local_cluster_expansion import (
    LocalClusterExpansion,
    _calc_corr,
    _calc_corr_decorated,
    _flatten_cluster_indices,
    _to_numba_cluster_site_indices,
)
from kmcpy.models.site_energy import SiteEnergyModel
from kmcpy.simulator.state import State


# Ragged orbits, including an empty orbit, to exercise the flat offsets. Plain
# lists are used because numba typed lists cannot represent an empty orbit.
CLUSTER_SITE_INDICES = [
    [[0]],
    [[1], [2], [3]],
    [],
    [[0, 1], [2, 3]],
    [[0, 1, 2]],
]
CLUSTER_BASIS_INDICES = [
    [[0]],
    [[1], [0], [1]],
    [],
    [[0, 1], [1, 0]],
    [[1, 1, 0]],
]


def _reference_corr(occupation, cluster_site_indices):
    return np.array(
        [
            sum(np.prod([occupation[site] for site in cluster]) for cluster in orbit)
            for orbit in cluster_site_indices
        ],
        dtype=float,
    )


def _reference_decorated_corr(occupation, cluster_site_indices, basis_indices, values):
    corr = []
    for orbit, basis_orbit in zip(cluster_site_indices, basis_indices):
        total = 0.0
        for cluster, basis_cluster in zip(orbit, basis_orbit):
            product = 1.0
            for site, basis_index in zip(cluster, basis_cluster):
                product *= values[site, int(occupation[site]), basis_index]
            total += product
        corr.append(total)
    return np.array(corr)


@pytest.mark.unit
def test_flat_correlation_kernel_matches_nested_reference():
    occupation = np.array([1, -1, -1, 1], dtype=np.int64)
    orbit_offsets, cluster_offsets, sites, _ = _flatten_cluster_indices(
        CLUSTER_SITE_INDICES
    )
    corr = np.empty(len(CLUSTER_SITE_INDICES))

    _calc_corr(corr, occupation, orbit_offsets, cluster_offsets, sites)

    np.testing.assert_array_equal(
        corr, _reference_corr(occupation, CLUSTER_SITE_INDICES)
    )


@pytest.mark.unit
def test_flat_decorated_kernel_matches_nested_reference():
    occupation = np.array([0, 2, 1, 2], dtype=np.int64)
    rng = np.random.default_rng(0)
    site_basis_values = rng.normal(size=(4, 3, 2))
    orbit_offsets, cluster_offsets, sites, basis = _flatten_cluster_indices(
        CLUSTER_SITE_INDICES,
        CLUSTER_BASIS_INDICES,
    )
    corr = np.empty(len(CLUSTER_SITE_INDICES))

    _calc_corr_decorated(
        corr,
        occupation,
        orbit_offsets,
        cluster_offsets,
        sites,
        basis,
        site_basis_values,
    )

    np.testing.assert_array_equal(
        corr,
        _reference_decorated_corr(
            occupation,
            CLUSTER_SITE_INDICES,
            CLUSTER_BASIS_INDICES,
            site_basis_values,
        ),
    )


def _plain_lce():
    model = LocalClusterExpansion()
    model.cluster_site_indices = _to_numba_cluster_site_indices([[[0]], [[1]]])
    model.keci = [1.0, 10.0]
    model.empty_cluster = 0.5
    return model


@pytest.mark.unit
def test_lce_compute_uses_reassigned_parameters_and_clusters():
    model = _plain_lce()
    state = State(occupations=[-1, 1, 1])
    event = Event(mobile_ion_indices=(0, 1), local_env_indices=(1, 2))

    # Local occupations are [1, 1] (active sites 1 and 2).
    assert model.compute(simulation_state=state, event=event) == pytest.approx(11.5)

    model.keci = [2.0, 3.0]
    assert model.compute(simulation_state=state, event=event) == pytest.approx(5.5)

    model.keci[1] = 4.0
    assert model.compute(simulation_state=state, event=event) == pytest.approx(6.5)

    model.cluster_site_indices = _to_numba_cluster_site_indices([[[0, 1]], [[1]]])
    state.occupations[2] = -1
    # Local occupations are [1, -1]: corr = [1 * -1, -1], keci = [2, 4].
    assert model.compute(simulation_state=state, event=event) == pytest.approx(-5.5)


@pytest.mark.unit
def test_lce_compute_revalidates_reassigned_keci_length():
    model = _plain_lce()
    model.get_orbit_fingerprints = lambda: ["a", "b"]
    model.parameter_orbit_fingerprints = ["a", "b"]
    state = State(occupations=[1, 1])
    event = Event(mobile_ion_indices=(0, 1), local_env_indices=(0, 1))
    model.compute(simulation_state=state, event=event)

    model.keci = [1.0]
    with pytest.raises(ValueError, match="keci length"):
        model.compute(simulation_state=state, event=event)


@pytest.mark.unit
def test_event_lib_dependency_cache_follows_regenerated_matrix():
    event_lib = EventLib()
    event_lib.add_event(Event(mobile_ion_indices=(0, 1), local_env_indices=(2,)))
    event_lib.add_event(Event(mobile_ion_indices=(2, 3), local_env_indices=()))
    event_lib.add_event(Event(mobile_ion_indices=(4, 5), local_env_indices=()))
    event_lib.generate_event_dependencies()

    assert sorted(event_lib.get_dependent_events(0)) == [0, 1]
    assert event_lib.get_dependent_events(2) == [2]

    event_lib.add_event(Event(mobile_ion_indices=(5, 6), local_env_indices=()))
    event_lib.generate_event_dependencies()

    assert sorted(event_lib.get_dependent_events(2)) == [2, 3]
    assert event_lib.get_dependent_events(99) == []


@pytest.mark.unit
def test_site_energy_rejects_unused_compute_kwargs_on_every_call():
    def compute_fn(changes):
        return 0.0

    model = SiteEnergyModel(
        compute_fn=compute_fn,
        compute_kwargs={"not_a_parameter": 1.0},
        site_mapping=[0, 1],
    )
    state = State(occupations=[0, 1])
    event = Event(mobile_ion_indices=(0, 1), local_env_indices=())

    for _ in range(2):
        with pytest.raises(TypeError, match="not_a_parameter"):
            model.compute(event=event, simulation_state=state)


# --- Batched composite rate evaluation -------------------------------------

from kmcpy.models.composite_lce_model import CompositeLCEModel
from kmcpy.simulator.config import RuntimeConfig
from kmcpy.simulator.kmc import _propose

RUNTIME_CONFIG = RuntimeConfig(temperature=300.0, attempt_frequency=1e13)
BATCH_EVENTS = [
    Event(mobile_ion_indices=(0, 1), local_env_indices=(2, 3)),
    Event(mobile_ion_indices=(1, 2), local_env_indices=(0, 3)),
    Event(mobile_ion_indices=(2, 3), local_env_indices=(0, 1)),
    Event(mobile_ion_indices=(3, 0), local_env_indices=(1, 2)),
    Event(mobile_ion_indices=(0, 2), local_env_indices=(1, 3)),
]


def _lce(keci, empty_cluster):
    model = LocalClusterExpansion()
    model.cluster_site_indices = _to_numba_cluster_site_indices(
        [[[0]], [[1]], [[0, 1]]]
    )
    model.keci = list(keci)
    model.empty_cluster = empty_cluster
    return model


def _scalar_rates(model, events, state):
    return np.array(
        [
            model.compute_probability(
                event=event,
                runtime_config=RUNTIME_CONFIG,
                simulation_state=state,
            )
            for event in events
        ]
    )


def _batched_rates(model, events, state):
    return model.compute_probabilities(
        events=events,
        event_indices=list(range(len(events))),
        runtime_config=RUNTIME_CONFIG,
        simulation_state=state,
    )


def _initialized_composite(site_model, state, events=BATCH_EVENTS):
    model = CompositeLCEModel(
        kra_model=_lce([12.0, -7.5, 3.25], 250.0),
        site_model=site_model,
    )
    model.initialize_state(simulation_state=state, event_lib=events)
    return model


@pytest.mark.unit
@pytest.mark.parametrize("with_site_model", [True, False])
def test_batched_composite_rates_match_scalar_rates_exactly(with_site_model):
    state = State(occupations=[0, 1, 0, 1])
    site_model = _lce([4.0, 9.0, -2.0], 1.5) if with_site_model else None
    model = _initialized_composite(site_model, state)

    assert model._batch_evaluator is not None
    batched = _batched_rates(model, BATCH_EVENTS, state)
    scalar = _scalar_rates(model, BATCH_EVENTS, state)

    # Forward, backward, and inactive events all occur in this state.
    assert (scalar > 0).any() and (scalar == 0).any()
    np.testing.assert_array_equal(batched, scalar)


@pytest.mark.unit
def test_batched_composite_rates_follow_state_changes():
    state = State(occupations=[0, 1, 0, 1])
    model = _initialized_composite(_lce([4.0, 9.0, -2.0], 1.5), state)

    # Accepted event committed through the model hook.
    state.apply_event(BATCH_EVENTS[0], dt=0.0)
    model.apply_event(event=BATCH_EVENTS[0], simulation_state=state)
    np.testing.assert_array_equal(
        _batched_rates(model, BATCH_EVENTS, state),
        _scalar_rates(model, BATCH_EVENTS, state),
    )

    # Occupations replaced outside the model; the step change forces a resync.
    state.occupations = [1, 0, 1, 0]
    state.step += 1
    np.testing.assert_array_equal(
        _batched_rates(model, BATCH_EVENTS, state),
        _scalar_rates(model, BATCH_EVENTS, state),
    )

    # A different State object also forces a resync.
    other_state = State(occupations=[1, 1, 0, 0])
    np.testing.assert_array_equal(
        _batched_rates(model, BATCH_EVENTS, other_state),
        _scalar_rates(model, BATCH_EVENTS, other_state),
    )


@pytest.mark.unit
def test_batched_composite_rates_fall_back_for_custom_submodels():
    class OffsetLCE(LocalClusterExpansion):
        def compute(self, simulation_state, event):
            return super().compute(simulation_state, event) + 100.0

    state = State(occupations=[0, 1, 0, 1])
    site_model = OffsetLCE()
    site_model.cluster_site_indices = _to_numba_cluster_site_indices([[[0]]])
    site_model.keci = [1.0]
    site_model.empty_cluster = 0.0
    model = _initialized_composite(site_model, state)

    assert model._batch_evaluator is None
    np.testing.assert_array_equal(
        _batched_rates(model, BATCH_EVENTS, state),
        _scalar_rates(model, BATCH_EVENTS, state),
    )


@pytest.mark.unit
def test_batched_composite_rates_fall_back_for_other_event_lists():
    state = State(occupations=[0, 1, 0, 1])
    model = _initialized_composite(None, state)
    other_events = list(BATCH_EVENTS)

    np.testing.assert_array_equal(
        _batched_rates(model, other_events, state),
        _scalar_rates(model, other_events, state),
    )


@pytest.mark.integration
def test_batched_rates_match_scalar_rates_for_nasicon_model(tmp_path, monkeypatch):
    from pathlib import Path

    from kmcpy.simulator.config import Configuration
    from kmcpy.simulator.kmc import KMC

    files = Path(__file__).parent / "files"
    monkeypatch.chdir(tmp_path)
    config = Configuration(
        structure_file=str(files / "EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif"),
        model_file=str(files / "input" / "model.json"),
        event_file=str(files / "input" / "events.json"),
        initial_state_file=str(files / "input" / "initial_state.json"),
        mobile_ion_specie="Na",
        temperature=298,
        attempt_frequency=5e12,
        equilibration_passes=0,
        kmc_passes=3,
        supercell_shape=(2, 1, 1),
        site_mapping={"Na": ["Na", "X"], "Zr": "Zr", "Si": ["Si", "P"], "O": "O"},
        convert_to_primitive_cell=True,
        elementary_hop_distance=3.47782,
        random_seed=7,
        name="batched",
    )
    kmc = KMC.from_config(config)
    kmc.run()

    assert kmc.model._batch_evaluator is not None
    events = kmc.event_lib.events
    state = kmc.simulation_state
    scalar = np.array(
        [
            kmc.model.compute_probability(
                event=event,
                runtime_config=config.runtime_config,
                simulation_state=state,
            )
            for event in events
        ]
    )
    batched = kmc.model.compute_probabilities(
        events=events,
        event_indices=list(range(len(events))),
        runtime_config=config.runtime_config,
        simulation_state=state,
    )
    np.testing.assert_array_equal(batched, scalar)
    # The incrementally maintained rates agree with a full recomputation.
    np.testing.assert_array_equal(kmc.prob_list, scalar)


@pytest.mark.unit
def test_propose_returns_python_scalars_from_the_rng_stream():
    prob_cum_list = np.cumsum([1.0, 2.0, 3.0])
    event_index, dt = _propose(prob_cum_list, np.random.default_rng(3))

    reference = np.random.default_rng(3)
    first, second = reference.random(), reference.random()
    assert isinstance(event_index, int) and isinstance(dt, float)
    assert event_index == int(
        np.searchsorted(prob_cum_list / prob_cum_list[-1], first, side="right")
    )
    assert dt == (-1.0 / prob_cum_list[-1]) * np.log(second)
