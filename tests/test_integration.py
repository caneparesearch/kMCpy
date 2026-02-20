#!/usr/bin/env python
"""Integration tests for SimulationConfig, KMC, and EventLib APIs."""


def test_simulation_config_integration():
    """Test that strict SimulationConfig integration works properly."""
    from kmcpy.simulator.config import RuntimeConfig, SimulationConfig, SystemConfig
    from kmcpy.simulator.kmc import KMC
    from tests.test_utils import create_nasicon_config, create_temperature_series

    runtime = RuntimeConfig(
        name="Runtime_Test",
        temperature=400.0,
        attempt_frequency=1e13,
        random_seed=42,
        equilibration_passes=1000,
        kmc_passes=5000,
    )
    system = SystemConfig(
        structure_file="test.cif",
        event_file="fake.json",
        dimension=3,
        elementary_hop_distance=2.5,
        mobile_ion_charge=1.0,
        mobile_ion_specie="Na",
        supercell_shape=(2, 2, 2),
        fitting_results_file="fake.json",
        fitting_results_site_file="fake.json",
        cluster_expansion_file="fake.json",
        cluster_expansion_site_file="fake.json",
        event_dependencies="fake.csv",
    )

    config = SimulationConfig(system_config=system, runtime_config=runtime)
    assert config.name == "Runtime_Test"
    assert config.temperature == 400.0
    assert config.kmc_passes == 5000

    modified = config.with_runtime_changes(temperature=600.0, name="Modified_Config")
    assert modified.temperature == 600.0
    assert modified.name == "Modified_Config"
    assert modified.attempt_frequency == config.attempt_frequency

    nasicon_config = create_nasicon_config(
        name="Test_NASICON",
        temperature=573.0,
        data_dir="fake_dir",
    )
    assert nasicon_config.name == "Test_NASICON"
    assert nasicon_config.temperature == 573.0
    assert nasicon_config.mobile_ion_specie == "Na"

    temp_series = create_temperature_series(nasicon_config, [300, 400, 500])
    assert len(temp_series) == 3
    assert temp_series[0].temperature == 300
    assert temp_series[1].temperature == 400
    assert temp_series[2].temperature == 500

    assert hasattr(KMC, "from_config")
    assert hasattr(KMC, "run")

    try:
        KMC.from_config(config)
        assert False, "Expected missing-file error"
    except Exception as exc:
        assert "unknown parameters" not in str(exc).lower()


def test_parameter_serialization():
    """Test strict parameter serialization and deserialization."""
    from kmcpy.simulator.config import SimulationConfig

    config = SimulationConfig(
        structure_file="fake.cif",
        name="Serialization_Test",
        temperature=450.0,
        attempt_frequency=1e13,
        equilibration_passes=1000,
        kmc_passes=5000,
        dimension=3,
        elementary_hop_distance=2.0,
        mobile_ion_charge=1.0,
        mobile_ion_specie="Na",
        supercell_shape=(2, 2, 2),
        fitting_results_file="fake.json",
        fitting_results_site_file="fake.json",
        cluster_expansion_file="fake.json",
        cluster_expansion_site_file="fake.json",
        event_file="fake.json",
        event_dependencies="fake.csv",
    )

    config_dict = config.to_dict()

    assert "name" in config_dict
    assert "temperature" in config_dict
    assert "attempt_frequency" in config_dict
    assert "equilibration_passes" in config_dict
    assert "kmc_passes" in config_dict
    assert "structure_file" in config_dict

    assert config_dict["equilibration_passes"] == config.equilibration_passes
    assert config_dict["kmc_passes"] == config.kmc_passes
    assert config_dict["elementary_hop_distance"] == config.elementary_hop_distance
    assert config_dict["mobile_ion_charge"] == config.mobile_ion_charge

    new_config = SimulationConfig.from_dict(config_dict)
    assert new_config.name == config.name
    assert new_config.temperature == config.temperature
    assert new_config.attempt_frequency == config.attempt_frequency
    assert new_config.equilibration_passes == config.equilibration_passes
    assert new_config.kmc_passes == config.kmc_passes


def test_eventlib_integration():
    """Test that EventLib is properly integrated."""
    from kmcpy.event import EventLib, Event

    event_lib = EventLib()

    event = Event(
        mobile_ion_indices=(0, 1),
        local_env_indices=[2, 3, 4],
    )
    event_lib.add_event(event)

    assert len(event_lib) > 0
    assert hasattr(event_lib, "generate_event_dependencies")
    assert hasattr(event_lib, "get_dependency_statistics")
    assert hasattr(event_lib, "save_event_dependencies_to_file")
