#!/usr/bin/env python
"""Integration tests for Configuration, KMC, and EventLib APIs."""

import pytest


def test_simulation_config_integration():
    """Test that strict Configuration integration works properly."""
    from kmcpy.simulator.config import RuntimeConfig, Configuration, SystemConfig
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
        model_file="fake_model.json",
    )

    config = Configuration(system_config=system, runtime_config=runtime)
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


def test_configuration_field_serialization():
    """Test strict configuration field serialization and deserialization."""
    from kmcpy.simulator.config import Configuration

    config = Configuration(
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
        model_file="fake_model.json",
        event_file="fake.json",
        property_sampling_interval=200,
        builtin_property_enabled={"conductivity": False},
        property_callbacks=[
            {
                "callable": "tests.test_kmc_state_tracker_boundaries:yaml_config_test_callback",
                "name": "cfg_callback",
                "interval": 50,
            }
        ],
    )

    config_dict = config.as_dict()

    assert "name" in config_dict
    assert "temperature" in config_dict
    assert "attempt_frequency" in config_dict
    assert "equilibration_passes" in config_dict
    assert "kmc_passes" in config_dict
    assert "structure_file" not in config_dict
    assert "model_file" not in config_dict
    assert "event_file" not in config_dict
    assert "property_sampling_interval" in config_dict
    assert "builtin_property_enabled" in config_dict
    assert "property_callbacks" in config_dict

    input_dict = config.as_input_dict()
    assert input_dict["structure_file"] == "fake.cif"
    assert input_dict["model_file"] == "fake_model.json"
    assert input_dict["event_file"] == "fake.json"

    assert config_dict["equilibration_passes"] == config.equilibration_passes
    assert config_dict["kmc_passes"] == config.kmc_passes
    assert config_dict["elementary_hop_distance"] == config.elementary_hop_distance
    assert config_dict["mobile_ion_charge"] == config.mobile_ion_charge
    assert config.field_units()["temperature"] == "K"
    assert config.field_units()["attempt_frequency"] == "Hz"
    assert config.field_units()["elementary_hop_distance"] == "Angstrom"

    new_config = Configuration.from_dict(config_dict)
    assert new_config.name == config.name
    assert new_config.temperature == config.temperature
    assert new_config.attempt_frequency == config.attempt_frequency
    assert new_config.equilibration_passes == config.equilibration_passes
    assert new_config.kmc_passes == config.kmc_passes
    assert new_config.property_sampling_interval == 200
    assert new_config.builtin_property_enabled["conductivity"] is False


def test_configuration_pymatgen_style_file_api(tmp_path):
    """Test as_dict/from_dict and to/from_file as the primary serialization API."""
    from monty.serialization import loadfn

    from kmcpy.simulator.config import Configuration

    config = Configuration(
        structure_file="fake.cif",
        model_file="fake_model.json",
        event_file="fake_events.json",
        initial_state_file="fake_state.json",
        mobile_ion_specie="Na",
        temperature=450.0,
        kmc_passes=5000,
    )

    record_file = tmp_path / "record.json"
    config.to(record_file)
    record_payload = loadfn(record_file, cls=None)

    assert "structure_file" not in record_payload
    assert "model_file" not in record_payload
    assert "event_file" not in record_payload
    assert "initial_state_file" not in record_payload

    input_file = tmp_path / "input.yaml"
    config.to(input_file, include_loader_paths=True, section="kmc", task_type="default")
    reloaded = Configuration.from_file(input_file)

    assert reloaded.structure_file == "fake.cif"
    assert reloaded.model_file == "fake_model.json"
    assert reloaded.event_file == "fake_events.json"
    assert reloaded.initial_state_file == "fake_state.json"


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


def test_eventlib_bundled_format():
    """Test EventLib bundled format with embedded dependencies."""
    import tempfile
    import os
    from kmcpy.event import EventLib, Event

    # Create event library with multiple events
    event_lib = EventLib()
    event1 = Event(mobile_ion_indices=(0, 1), local_env_indices=[2, 3, 4])
    event2 = Event(mobile_ion_indices=(1, 2), local_env_indices=[0, 3, 5])
    event_lib.add_event(event1)
    event_lib.add_event(event2)

    from pymatgen.core import Lattice, Structure
    from kmcpy.structure import ActiveSiteOrder

    index_structure = Structure(
        Lattice.cubic(10.0),
        ["Na"] * 6,
        [[i, 0, 0] for i in range(6)],
        coords_are_cartesian=True,
    )
    index_map = ActiveSiteOrder.from_structure_and_mapping(
        index_structure, {"Na": ["Na", "X"]}
    )
    event_lib.set_index_metadata(index_map)

    # Generate dependencies
    event_lib.generate_event_dependencies()
    assert event_lib.has_event_dependencies()

    # Save in bundled format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        bundled_file = f.name

    try:
        event_lib.to(bundled_file)

        # Load back and verify deps are preserved (not regenerated)
        loaded = EventLib.from_file(bundled_file)
        assert len(loaded) == 2
        assert loaded.has_event_dependencies()

        # Verify structure matches
        assert loaded.events[0].mobile_ion_indices == (0, 1)
        assert loaded.events[1].mobile_ion_indices == (1, 2)
    finally:
        os.unlink(bundled_file)


def test_eventlib_legacy_format():
    """Test EventLib can still load legacy list-format event files."""
    import tempfile
    import os
    import json
    from kmcpy.event import EventLib

    # Create legacy format file (just a list of events)
    legacy_events = [
        {"mobile_ion_indices": [0, 1], "local_env_indices": [2, 3, 4]},
        {"mobile_ion_indices": [1, 2], "local_env_indices": [0, 3, 5]},
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        legacy_file = f.name
        json.dump(legacy_events, f)

    try:
        with pytest.raises(ValueError, match="Legacy list-format event files"):
            EventLib.from_file(legacy_file)
    finally:
        os.unlink(legacy_file)
