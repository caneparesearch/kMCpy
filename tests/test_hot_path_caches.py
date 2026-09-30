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
