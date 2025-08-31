"""
Tests for SimulationState with Occupation integration.
"""

import pytest
import tempfile
import os
from kmcpy.simulator.state import SimulationState
from kmcpy.structure.basis import Occupation


class TestSimulationStateIntegration:
    """Test SimulationState integration with Occupation objects."""
    
    def test_init_with_occupation_object(self):
        """Test initialization with Occupation object."""
        occ = Occupation([-1, 1, -1, 1], basis='chebyshev')
        state = SimulationState(occupations=occ)
        
        assert isinstance(state.occupations, Occupation)
        assert state.occupations.values == [-1, 1, -1, 1]
        assert state.occupations.basis == 'chebyshev'
        assert state.time == 0.0
        assert state.step == 0
    
    def test_init_requires_occupation_object(self):
        """Test that SimulationState requires an Occupation object."""
        from kmcpy.structure.basis import Occupation
        
        # Should raise TypeError with raw list
        with pytest.raises(TypeError, match="occupations must be an Occupation object"):
            SimulationState(occupations=[-1, 1, -1, 1])
            
        # Should work with Occupation object
        occ = Occupation([-1, 1, -1, 1], basis='chebyshev')
        state = SimulationState(occupations=occ)
        
        assert isinstance(state.occupations, Occupation)
        assert state.occupations.values == [-1, 1, -1, 1]
        assert state.occupations.basis == 'chebyshev'
    
    def test_occupation_methods_integration(self):
        """Test that occupation-based methods work correctly."""
        occ = Occupation([-1, 1, -1, -1], basis='chebyshev')
        state = SimulationState(occupations=occ)
        
        # Test methods that use Occupation functionality
        assert state.get_mobile_species_sites() == [1]
        assert state.get_vacant_sites() == [0, 2, 3]
        assert state.count_mobile_species() == 1
    
    def test_apply_event_uses_occupation_methods(self):
        """Test that apply_event uses Occupation KMC methods."""
        occ = Occupation([-1, 1, -1, -1], basis='chebyshev')
        state = SimulationState(occupations=occ)
        
        # Create mock event
        class MockEvent:
            def __init__(self, from_site, to_site):
                self.mobile_ion_indices = (from_site, to_site)
        
        event = MockEvent(from_site=1, to_site=2)
        state.apply_event(event, dt=0.5)
        
        # Check that occupation was updated correctly
        assert state.occupations.values == [-1, -1, 1, -1]
        assert state.time == 0.5
        assert state.step == 1
    
    def test_copy_preserves_occupation_object(self):
        """Test that copy creates new Occupation objects."""
        occ = Occupation([-1, 1, -1, 1], basis='chebyshev')
        state = SimulationState(occupations=occ, time=1.0, step=5)
        
        copied_state = state.copy()
        
        # Check that data is preserved
        assert copied_state.occupations.values == state.occupations.values
        assert copied_state.occupations.basis == state.occupations.basis
        assert copied_state.time == state.time
        assert copied_state.step == state.step
        
        # Check that they are different objects
        assert copied_state.occupations is not state.occupations
    
    def test_checkpoint_save_load_with_basis(self):
        """Test checkpoint functionality preserves basis information."""
        occ = Occupation([-1, 1, -1, 1], basis='chebyshev')
        state = SimulationState(occupations=occ, time=2.5, step=100)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Save and load checkpoint
            state.save_checkpoint(temp_file)
            loaded_state = SimulationState.load_checkpoint(temp_file)
            
            # Verify all data is preserved
            assert loaded_state.occupations.values == state.occupations.values
            assert loaded_state.occupations.basis == state.occupations.basis
            assert loaded_state.time == state.time
            assert loaded_state.step == state.step
            
            # Verify it's a proper Occupation object
            assert isinstance(loaded_state.occupations, Occupation)
            
        finally:
            os.unlink(temp_file)
    
    def test_checkpoint_backward_compatibility(self):
        """Test that old checkpoints without basis info still work."""
        import json
        
        # Create an old-style checkpoint without basis info
        old_checkpoint_data = {
            'occupations': [-1, 1, -1, 1],
            'time': 1.5,
            'step': 50,
            'n_mobile_species': 2
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(old_checkpoint_data, f)
            temp_file = f.name
        
        try:
            # Load old checkpoint
            loaded_state = SimulationState.load_checkpoint(temp_file)
            
            # Should default to chebyshev basis
            assert loaded_state.occupations.values == [-1, 1, -1, 1]
            assert loaded_state.occupations.basis == 'chebyshev'
            assert loaded_state.time == 1.5
            assert loaded_state.step == 50
            
        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    pytest.main([__file__])
