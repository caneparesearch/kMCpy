import numpy as np

from kmcpy.simulator.property import BUILTIN_PROPERTY_UNITS, compute_transport_properties
from kmcpy.units import (
    ANGSTROM_SQUARED_TO_CM_SQUARED,
    BOLTZMANN_CONSTANT_MEV_PER_K,
    CONDUCTIVITY_MS_PER_CM_FACTOR,
    TRANSPORT_PROPERTY_UNITS,
)



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
    assert np.isclose(
        metrics["jump_diffusivity"],
        (2.0 / 24.0) * ANGSTROM_SQUARED_TO_CM_SQUARED,
    )
    assert np.isclose(
        metrics["tracer_diffusivity"],
        (1.0 / 12.0) * ANGSTROM_SQUARED_TO_CM_SQUARED,
    )
    assert np.isclose(metrics["havens_ratio"], 1.0)
    assert np.isclose(metrics["correlation_factor"], 0.75)

    expected_conductivity = (
        (2.0 / 24.0)
        * ANGSTROM_SQUARED_TO_CM_SQUARED
        * (2.0 / 10.0)
        / (BOLTZMANN_CONSTANT_MEV_PER_K * 300.0)
        * CONDUCTIVITY_MS_PER_CM_FACTOR
    )
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



def test_compute_transport_properties_zero_hop_correlation_factor_is_zero():
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

    assert metrics["correlation_factor"] == 0.0


def test_builtin_property_units_are_explicit():
    assert BUILTIN_PROPERTY_UNITS == {
        key: TRANSPORT_PROPERTY_UNITS[key]
        for key in (
            "msd",
            "jump_diffusivity",
            "tracer_diffusivity",
            "conductivity",
            "havens_ratio",
            "correlation_factor",
        )
    }
    assert TRANSPORT_PROPERTY_UNITS["jump_diffusivity"] == "cm^2/s"
    assert TRANSPORT_PROPERTY_UNITS["conductivity"] == "mS/cm"


def test_compute_transport_properties_uses_master_compatible_correlation_factor():
    displacement = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [99.0, 0.0, 0.0],
        ]
    )
    hop_counter = np.array([2, 3, 0], dtype=np.int64)

    metrics = compute_transport_properties(
        displacement,
        hop_counter,
        sim_time=1.0,
        dimension=3,
        n_mobile_ion_specie=3,
        elementary_hop_distance=1.0,
        volume=1.0,
        mobile_ion_charge=1.0,
        temperature=300.0,
    )

    expected = (4.0 / 2.0 + 9.0 / 3.0 + 0.0) / 3.0
    assert np.isclose(metrics["correlation_factor"], expected)
