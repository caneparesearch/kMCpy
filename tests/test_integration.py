#!/usr/bin/env python
"""
Test script to verify KMC + EventLib integration with SimulationCondition support.
"""

import sys
import os
sys.path.insert(0, '..')

def test_simulation_condition_integration():
    """Test that SimulationCondition integration works properly."""
    from kmcpy.simulator.condition import (
        SimulationCondition, 
        SimulationConfig,
    )
    from tests.test_utils import create_nasicon_config, create_temperature_series
    from kmcpy.simulator.kmc import KMC
    
    print("✓ SimulationCondition imports successful")
    
    # Test basic SimulationCondition
    basic_condition = SimulationCondition(
        name="Test_Condition",
        temperature=400.0,
        attempt_frequency=1e13,
        random_seed=42
    )
    
    assert basic_condition.name == "Test_Condition"
    assert basic_condition.temperature == 400.0
    assert basic_condition.attempt_frequency == 1e13
    assert basic_condition.random_seed == 42
    print("✓ Basic SimulationCondition creation works")
    
    # Test parameter validation
    try:
        bad_condition = SimulationCondition(
            name="Bad_Condition",
            temperature=-100.0,  # Invalid
            attempt_frequency=1e13
        )
        assert False, "Expected validation error"
    except ValueError:
        print("✓ Parameter validation works")
    
    # Test SimulationConfig (merged with KMC functionality)
    kmc_condition = SimulationConfig(
        structure_file="test.cif",  # Required parameter
        name="KMC_Test",
        temperature=573.0,
        attempt_frequency=1e13,
        equilibration_passes=1000,
        kmc_passes=5000,
        dimension=3,
        elementary_hop_distance=2.5,
        mobile_ion_charge=1.0,
        random_seed=123
    )
    
    assert kmc_condition.equilibration_passes == 1000
    assert kmc_condition.kmc_passes == 5000
    assert kmc_condition.dimension == 3
    print("✓ SimulationConfig creation works")
    
    # Test SimulationConfig creation
    config = SimulationConfig(
        structure_file="fake.cif",  # Required parameter with correct name
        name="Test_Config",
        temperature=500.0,
        attempt_frequency=1e13,
        equilibration_passes=500,
        kmc_passes=2000,
        dimension=3,
        elementary_hop_distance=2.0,
        mobile_ion_charge=1.0,
        mobile_ion_specie="Na",
        supercell_shape=(2, 2, 2),  # Use tuple
        
        # File paths with correct parameter names
        fitting_results_file="fake.json",
        fitting_results_site_file="fake.json",
        cluster_expansion_file="fake.json",
        cluster_expansion_site_file="fake.json",
        event_file="fake.json",
        event_dependencies="fake.csv"
    )
    
    assert config.name == "Test_Config"
    assert config.temperature == 500.0
    print("✓ SimulationConfig creation works")
    
    # Test parameter modification using the new API
    modified_config = config.with_runtime_changes(
        temperature=600.0,
        name="Modified_Config"
    )
    
    assert modified_config.temperature == 600.0
    assert modified_config.name == "Modified_Config"
    assert modified_config.attempt_frequency == config.attempt_frequency  # Should be unchanged
    print("✓ Parameter modification works")
    
    # Test convenience function
    nasicon_config = create_nasicon_config(
        name="Test_NASICON",
        temperature=573.0,
        data_dir="fake_dir"
    )
    
    assert nasicon_config.name == "Test_NASICON"
    assert nasicon_config.temperature == 573.0
    assert nasicon_config.mobile_ion_specie == "Na"
    print("✓ Convenience function works")
    
    # Test temperature series
    temp_series = create_temperature_series(nasicon_config, [300, 400, 500])
    assert len(temp_series) == 3
    assert temp_series[0].temperature == 300
    assert temp_series[1].temperature == 400
    assert temp_series[2].temperature == 500
    print("✓ Temperature series creation works")
    
    # Test KMC integration methods exist
    assert hasattr(KMC, 'from_config'), "Missing from_config method"
    assert hasattr(KMC, 'run'), "Missing run method"
    print("✓ KMC integration methods exist")
    
    # Test that KMC methods accept SimulationConfig (will fail due to missing files)
    try:
        kmc_instance = KMC.from_config(config)
        assert False, "Expected file not found error"
    except Exception as e:
        # Should fail due to missing files, not parameter issues
        assert "temperature" not in str(e).lower(), f"Parameter issue detected: {e}"
        print("✓ KMC.from_config accepts SimulationConfig")
    
    try:
        # Test using the consolidated run method with config
        kmc_instance = KMC.from_config(config)
        tracker = kmc_instance.run(config)
        assert False, "Expected file not found error"
    except Exception as e:
        # Should fail due to missing files, not parameter issues
        assert "temperature" not in str(e).lower(), f"Parameter issue detected: {e}"
        print("✓ KMC.run accepts SimulationConfig")
    
    print("✅ SimulationCondition integration test passed!")

def test_parameter_serialization():
    """Test parameter serialization and deserialization."""
    from kmcpy.simulator.condition import SimulationConfig
    
    # Create a config
    config = SimulationConfig(
        structure_file="fake.cif",  # Required parameter with correct name
        name="Serialization_Test",
        temperature=450.0,
        attempt_frequency=1e13,
        equilibration_passes=1000,
        kmc_passes=5000,
        dimension=3,
        elementary_hop_distance=2.0,
        mobile_ion_charge=1.0,
        mobile_ion_specie="Na",
        supercell_shape=(2, 2, 2),  # Use tuple instead of list
        
        # File paths with correct parameter names
        fitting_results_file="fake.json",
        fitting_results_site_file="fake.json",
        cluster_expansion_file="fake.json",
        cluster_expansion_site_file="fake.json",
        event_file="fake.json",
        event_dependencies="fake.csv"
    )
    
    # Test serialization
    config_dict = config.to_dict()
    
    # Check that dictionary contains expected keys (using legacy key names from RuntimeConfig.to_dict)
    assert 'name' in config_dict
    assert 'temperature' in config_dict
    assert 'equ_pass' in config_dict  # Legacy key name from RuntimeConfig.to_dict
    assert 'kmc_pass' in config_dict  # Legacy key name from RuntimeConfig.to_dict
    assert 'structure_file' in config_dict  # Clean key from SystemConfig.to_dict
    
    # Test parameter mapping - use the actual dict we created
    assert config_dict['equ_pass'] == config.equilibration_passes
    assert config_dict['kmc_pass'] == config.kmc_passes
    # These are system parameters and use clean names
    assert config_dict['elementary_hop_distance'] == config.elementary_hop_distance
    assert config_dict['mobile_ion_charge'] == config.mobile_ion_charge
    
    print("✓ Parameter serialization works correctly")
    
    # Test that we can recreate config from from_dict
    new_config = SimulationConfig.from_dict(config_dict)
    assert new_config.name == config.name
    assert new_config.temperature == config.temperature
    assert new_config.attempt_frequency == config.attempt_frequency
    
    print("✓ Parameter deserialization works correctly")
    print("✅ Parameter serialization test passed!")

def test_eventlib_integration():
    """Test that EventLib is properly integrated."""
    from kmcpy.event import EventLib, Event
    
    # Create a small EventLib test
    event_lib = EventLib()
    
    # Add a test event
    event = Event(
        mobile_ion_indices=(0, 1),
        local_env_indices=[2, 3, 4]
    )
    event_lib.add_event(event)
    
    # Verify that event was added
    assert len(event_lib) > 0, "No events in EventLib"
    
    # Test that EventLib has the required methods
    assert hasattr(event_lib, 'generate_event_dependencies'), "Missing generate_event_dependencies method"
    assert hasattr(event_lib, 'get_dependency_statistics'), "Missing get_dependency_statistics method"
    assert hasattr(event_lib, 'save_event_dependencies_to_file'), "Missing save_event_dependencies_to_file method"
    
    print(f"✓ EventLib has {len(event_lib)} events and all required methods")
    print("✓ EventLib integration test passed!")

if __name__ == "__main__":
    print("Testing KMC + EventLib Integration with SimulationCondition Support")
    print("=" * 70)
    
    try:
        test_simulation_condition_integration()
        print()
        test_parameter_serialization()
        print()
        test_eventlib_integration()
        
        print("\n" + "=" * 70)
        print("🎉 All tests passed! KMC + EventLib + SimulationCondition integration is working.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
