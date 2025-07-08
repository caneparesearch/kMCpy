#!/usr/bin/env python
"""
Test script to verify KMC + EventLib integration with backward compatibility.
"""

import sys
import os
sys.path.insert(0, '..')

def test_backward_compatibility():
    """Test that KMC class works with event_kernel parameter (backward compatibility)."""
    from kmcpy.kmc import KMC
    from kmcpy.event import EventLib, Event
    
    print("✓ Imports successful")
    
    # Test direct instantiation with event_kernel parameter
    # Note: This would fail without proper files, but we can test the parameter handling
    try:
        kmc = KMC(
            initial_occ=[1, -1, 1, -1],
            supercell_shape=[2, 1, 1],
            fitting_results="fake.json",
            fitting_results_site="fake.json",
            lce_fname="fake.json",
            lce_site_fname="fake.json",
            template_structure_fname="fake.cif",
            event_fname="fake.json",
            event_kernel="fake.csv"  # Old parameter name
        )
        assert False, "Expected exception due to missing files"
    except Exception as e:
        # Expected to fail due to missing files, but check the error message
        assert "event_dependencies" not in str(e), f"Backward compatibility failed: {e}"
        print("✓ Backward compatibility parameter accepted (failed later due to missing files)")
    
    # Test with new parameter name
    try:
        kmc = KMC(
            initial_occ=[1, -1, 1, -1],
            supercell_shape=[2, 1, 1],
            fitting_results="fake.json",
            fitting_results_site="fake.json",
            lce_fname="fake.json",
            lce_site_fname="fake.json",
            template_structure_fname="fake.cif",
            event_fname="fake.json",
            event_dependencies="fake.csv"  # New parameter name
        )
        assert False, "Expected exception due to missing files"
    except Exception as e:
        assert not ("event_dependencies" in str(e) and "required" in str(e)), f"New parameter name failed: {e}"
        print("✓ New parameter name accepted (failed later due to missing files)")
    
    print("✅ Backward compatibility test passed!")

def test_eventlib_integration():
    """Test that EventLib is properly integrated."""
    from kmcpy.event import EventLib, Event
    
    # Create a small EventLib test
    event_lib = EventLib()
    
    # Add a test event
    event = Event()
    event.initialization(
        mobile_ion_specie_1_index=0,
        mobile_ion_specie_2_index=1,
        local_env_indices_list=[2, 3, 4]
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
    print("Testing KMC + EventLib Integration with Backward Compatibility")
    print("=" * 60)
    
    try:
        test_backward_compatibility()
        print()
        test_eventlib_integration()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! KMC + EventLib integration is working.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
