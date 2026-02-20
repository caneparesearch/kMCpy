import numpy as np

from kmcpy.simulator.built_in_properties import compute_transport_properties



def test_compute_transport_properties_values():
    displacement = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    hop_counter = np.array([1, 2], dtype=np.int64)

    metrics = compute_transport_properties(
        displacement,
        hop_counter,
        sim_time=2.0,
        dimension=3,
        n_mobile_ion_specie=2,
        elementary_hop_distance=1.0,
        volume=10.0,
        mobile_ion_charge=1.0,
        temperature=300.0,
    )

    assert np.isclose(metrics["msd"], 1.0)
    assert np.isclose(metrics["jump_diffusivity"], (2.0 / 24.0) * 1e-16)
    assert np.isclose(metrics["tracer_diffusivity"], (1.0 / 12.0) * 1e-16)
    assert np.isclose(metrics["havens_ratio"], 1.0)
    assert np.isclose(metrics["correlation_factor"], 0.75)

    k = 8.617333262145 * 10 ** (-2)
    expected_conductivity = ((2.0 / 24.0) * 1e-16) * (2.0 / 10.0) / (k * 300.0) * 1.602 * 10**11
    assert np.isclose(metrics["conductivity"], expected_conductivity)



def test_compute_transport_properties_disabled_fields_emit_nan():
    displacement = np.array([[1.0, 0.0, 0.0]])
    hop_counter = np.array([1], dtype=np.int64)

    metrics = compute_transport_properties(
        displacement,
        hop_counter,
        sim_time=1.0,
        dimension=3,
        n_mobile_ion_specie=1,
        elementary_hop_distance=1.0,
        volume=1.0,
        mobile_ion_charge=1.0,
        temperature=300.0,
        enabled={"conductivity": False, "havens_ratio": False},
    )

    assert np.isnan(metrics["conductivity"])
    assert np.isnan(metrics["havens_ratio"])
    assert np.isfinite(metrics["jump_diffusivity"])



def test_compute_transport_properties_zero_hop_correlation_factor():
    displacement = np.zeros((2, 3), dtype=float)
    hop_counter = np.array([0, 0], dtype=np.int64)

    metrics = compute_transport_properties(
        displacement,
        hop_counter,
        sim_time=1.0,
        dimension=3,
        n_mobile_ion_specie=2,
        elementary_hop_distance=1.0,
        volume=1.0,
        mobile_ion_charge=1.0,
        temperature=300.0,
    )

    assert np.isclose(metrics["correlation_factor"], 0.0)
