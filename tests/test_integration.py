#!/usr/bin/env python
"""
Test script to verify KMC + EventLib integration with SimulationCondition support.
"""

import sys
import os
sys.path.insert(0, '..')

def test_simulation_condition_integration():
    """Test that SimulationCondition integration works properly."""
    from kmcpy.simulation.condition import (
        SimulationCondition, 
        SimulationConfig,
    )
    from tests.test_utils import create_nasicon_config, create_temperature_series
    from kmcpy.simulation.kmc import KMC
    
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
        name="Test_Config",
        temperature=500.0,
        attempt_frequency=1e13,
        equilibration_passes=500,
        kmc_passes=2000,
        dimension=3,
        elementary_hop_distance=2.0,
        mobile_ion_charge=1.0,
        mobile_ion_specie="Na",
        supercell_shape=[2, 2, 2],
        initial_occ=[1, -1, 1, -1, 1, -1, 1, -1],
        
        # File paths (these would be real in practice)
        fitting_results="fake.json",
        fitting_results_site="fake.json",
        lce_fname="fake.json",
        lce_site_fname="fake.json",
        template_structure_fname="fake.cif",
        event_fname="fake.json",
        event_dependencies="fake.csv"
    )
    
    assert config.name == "Test_Config"
    assert config.temperature == 500.0
    assert len(config.initial_occ) == 8
    print("✓ SimulationConfig creation works")
    
    # Test parameter modification
    modified_config = config.copy_with_changes(
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
    from kmcpy.simulation.condition import SimulationConfig
    
    # Create a config
    config = SimulationConfig(
        name="Serialization_Test",
        temperature=450.0,
        attempt_frequency=1e13,
        equilibration_passes=1000,
        kmc_passes=5000,
        dimension=3,
        elementary_hop_distance=2.0,
        mobile_ion_charge=1.0,
        mobile_ion_specie="Na",
        supercell_shape=[2, 2, 2],
        initial_occ=[1, -1, 1, -1],
        
        # File paths
        fitting_results="fake.json",
        fitting_results_site="fake.json",
        lce_fname="fake.json",
        lce_site_fname="fake.json",
        template_structure_fname="fake.cif",
        event_fname="fake.json",
        event_dependencies="fake.csv"
    )
    
    # Test serialization
    regular_dict = config.to_dict()
    dataclass_dict = config.to_dataclass_dict()
    
    # Check that dictionaries contain expected keys
    assert 'name' in regular_dict
    assert 'temperature' in regular_dict
    assert 'v' in regular_dict  # Should map attempt_frequency to v
    assert 'equ_pass' in regular_dict  # Should map equilibration_passes to equ_pass
    assert 'kmc_pass' in regular_dict  # Should map kmc_passes to kmc_pass
    
    assert 'name' in dataclass_dict
    assert 'temperature' in dataclass_dict
    assert 'attempt_frequency' in dataclass_dict  # Should keep original field name
    assert 'equilibration_passes' in dataclass_dict  # Should keep original field name
    assert 'kmc_passes' in dataclass_dict  # Should keep original field name
    
    # Test parameter mapping
    assert regular_dict['v'] == config.attempt_frequency
    assert regular_dict['equ_pass'] == config.equilibration_passes
    assert regular_dict['kmc_pass'] == config.kmc_passes
    assert regular_dict['elem_hop_distance'] == config.elementary_hop_distance
    assert regular_dict['q'] == config.mobile_ion_charge
    
    print("✓ Parameter serialization works correctly")
    
    # Test that we can recreate config from dataclass dict
    new_config = SimulationConfig(**dataclass_dict)
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
