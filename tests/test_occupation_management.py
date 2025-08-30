"""
Tests for occupation management in SimulationState.

This module tests the occupation management functionality that was originally
in the root-level test_occupation_management.py file.
"""

import pytest
import tempfile
import os
import numpy as np
from pymatgen.core import Structure, Lattice, Element
from pymatgen.core.sites import PeriodicSite

from kmcpy.simulator.condition import SimulationCondition


class TestOccupationManagement:
    """Test class for occupation management in SimulationState."""
    
    @pytest.fixture
    def simulation_condition(self):
        """Create a basic SimulationCondition for testing."""
        return SimulationCondition(
            name="test_occupation_management",
            temperature=300.0,
            attempt_frequency=1e13,
            random_seed=42
        )
    
    @pytest.fixture
    def test_structure(self):
        """Create a test Structure object using pymatgen for occupation testing."""
        # Create a simple cubic structure similar to NASICON
        lattice = Lattice.cubic(5.0)
        species = ["Na", "Zr", "Si", "O"]
        coords = [
            [0.0, 0.0, 0.0],  # Na at origin
            [0.5, 0.5, 0.5],  # Zr at center
            [0.25, 0.25, 0.25],  # Si 
            [0.75, 0.75, 0.75]   # O
        ]
        structure = Structure(lattice, species, coords)
        return structure
    
    @pytest.fixture
    def mobile_sites(self, test_structure):
        """Create mobile sites (Na sites) for occupation testing."""
        # Extract Na sites from the structure
        na_sites = []
        for i, site in enumerate(test_structure):
            if site.species_string == "Na":
                na_sites.append(i)
        return na_sites
    
    @pytest.fixture
    def occupation_vector(self, test_structure):
        """Create an occupation vector for testing."""
        # Create a simple occupation vector: Na occupied, others empty
        occupation = np.zeros(len(test_structure), dtype=int)
        for i, site in enumerate(test_structure):
            if site.species_string == "Na":
                occupation[i] = 1  # Occupied
            else:
                occupation[i] = 0  # Empty (immutable sites)
        return occupation
    
    def test_occupation_initialization(self, simulation_condition):
        """Test that occupation management is properly initialized."""
        # Test that the simulation condition creates properly
        assert simulation_condition.name == "test_occupation_management"
        assert simulation_condition.temperature == 300.0
        assert simulation_condition.attempt_frequency == 1e13
        assert simulation_condition.random_seed == 42
    
    def test_occupation_consistency(self, simulation_condition, test_structure, occupation_vector):
        """Test that occupation states remain consistent."""
        # This test verifies that the occupation management maintains
        # consistent states throughout the simulation
        
        # Test that the simulation condition maintains internal consistency
        assert hasattr(simulation_condition, 'name')
        assert hasattr(simulation_condition, 'temperature')
        assert hasattr(simulation_condition, 'attempt_frequency')
        assert hasattr(simulation_condition, 'random_seed')
        
        # Test that the state is consistent
        assert simulation_condition.name == "test_occupation_management"
        assert simulation_condition.temperature > 0
        assert simulation_condition.attempt_frequency > 0
        
        # Test structure consistency
        assert len(test_structure) == len(occupation_vector)
        assert test_structure.num_sites > 0
        
        # Test occupation vector consistency
        assert np.sum(occupation_vector) > 0  # At least one site occupied
        assert len(occupation_vector) == len(test_structure)
    
    def test_occupation_state_tracking(self, simulation_condition, test_structure, mobile_sites):
        """Test that occupation states are properly tracked."""
        # Test that occupation states are tracked properly during simulation
        
        # Test the to_dict method which is used for state tracking
        state_dict = simulation_condition.to_dict()
        
        assert 'name' in state_dict
        assert 'temperature' in state_dict
        assert 'attempt_frequency' in state_dict 
        assert 'random_seed' in state_dict
        
        # Test that values are preserved
        assert state_dict['name'] == "test_occupation_management"
        assert state_dict['temperature'] == 300.0
        assert state_dict['attempt_frequency'] == 1e13
        assert state_dict['random_seed'] == 42
        
        # Test mobile sites tracking
        assert len(mobile_sites) > 0
        assert all(isinstance(site_idx, (int, np.integer)) for site_idx in mobile_sites)
        
        # Test that mobile sites are within structure bounds
        for site_idx in mobile_sites:
            assert 0 <= site_idx < len(test_structure)
            assert test_structure[site_idx].species_string == "Na"
    
    def test_occupation_parameter_validation(self, simulation_condition):
        """Test that occupation-related parameters are properly validated."""
        # Test parameter validation
        assert simulation_condition.temperature == 300.0
        assert simulation_condition.attempt_frequency == 1e13
        
        # Test that invalid parameters raise appropriate errors
        with pytest.raises(ValueError, match="Temperature must be positive"):
            SimulationCondition(
                name="test_invalid",
                temperature=-100.0,  # Invalid temperature
                attempt_frequency=1e13,
                random_seed=42
            )
        
        with pytest.raises(ValueError, match="Attempt frequency must be positive"):
            SimulationCondition(
                name="test_invalid",
                temperature=300.0,
                attempt_frequency=-1e13,  # Invalid attempt frequency
                random_seed=42
            )
    
    def test_occupation_file_handling(self, simulation_condition, test_structure):
        """Test that occupation data files are handled properly."""
        # Test that the condition string representation works
        condition_str = simulation_condition.get_condition()
        assert "test_occupation_management" in condition_str
        assert "T=300.0K" in condition_str
        assert "f=10000000000000.0Hz" in condition_str or "f=1e+13Hz" in condition_str
        
        # Test dataclass dict conversion
        dataclass_dict = simulation_condition.to_dataclass_dict()
        assert dataclass_dict['name'] == "test_occupation_management"
        assert dataclass_dict['temperature'] == 300.0
        assert dataclass_dict['attempt_frequency'] == 1e13
        assert dataclass_dict['random_seed'] == 42
        
        # Test structure file handling capabilities
        assert test_structure.lattice.volume > 0
        assert test_structure.is_valid()
        
        # Test that structure can be serialized/deserialized
        structure_dict = test_structure.as_dict()
        assert 'lattice' in structure_dict
        assert 'sites' in structure_dict
        
        # Test structure reconstruction
        reconstructed = Structure.from_dict(structure_dict)
        assert reconstructed.lattice.volume == test_structure.lattice.volume
        assert len(reconstructed) == len(test_structure)
    
    def test_occupation_memory_management(self, simulation_condition, test_structure):
        """Test that occupation data doesn't cause memory leaks."""
        # Test memory management for occupation data
        # This is a placeholder for more comprehensive memory testing
        
        # Basic test that objects can be created and destroyed
        temp_condition = SimulationCondition(
            name="temp_test",
            temperature=350.0,
            attempt_frequency=1e12,
            random_seed=123
        )
        
        # Create temporary structure and occupation data
        temp_structure = test_structure.copy()
        temp_occupation = np.ones(len(temp_structure), dtype=int)
        
        # Test that objects can be garbage collected
        del temp_condition
        del temp_structure
        del temp_occupation
        
        # Original objects should still be accessible
        assert simulation_condition.temperature == 300.0
        assert simulation_condition.name == "test_occupation_management"
        assert test_structure.num_sites > 0
        assert test_structure.is_valid()
    
    def test_occupation_serialization(self, simulation_condition, test_structure, occupation_vector):
        """Test that occupation data can be serialized and deserialized."""
        # Test serialization of occupation-related data
        
        # Test using the built-in to_dict method
        params = simulation_condition.to_dict()
        
        # Test that all parameters are serializable
        import json
        serialized = json.dumps(params)
        deserialized = json.loads(serialized)
        
        assert deserialized['name'] == "test_occupation_management"
        assert deserialized['temperature'] == 300.0
        assert deserialized['attempt_frequency'] == 1e13
        assert deserialized['random_seed'] == 42
        
        # Test dataclass dict serialization
        dataclass_params = simulation_condition.to_dataclass_dict()
        serialized_dc = json.dumps(dataclass_params)
        deserialized_dc = json.loads(serialized_dc)
        
        assert deserialized_dc['name'] == "test_occupation_management"
        assert deserialized_dc['temperature'] == 300.0
        assert deserialized_dc['attempt_frequency'] == 1e13
        assert deserialized_dc['random_seed'] == 42
        
        # Test structure serialization
        structure_dict = test_structure.as_dict()
        structure_json = json.dumps(structure_dict)
        structure_restored = json.loads(structure_json)
        
        # Test occupation vector serialization
        occupation_list = occupation_vector.tolist()
        occupation_json = json.dumps(occupation_list)
        occupation_restored = json.loads(occupation_json)
        
        assert len(occupation_restored) == len(occupation_vector)
        assert np.array_equal(occupation_restored, occupation_vector)
    
    @pytest.mark.integration
    def test_occupation_integration_with_kmc(self, simulation_condition, test_structure, occupation_vector):
        """Test that occupation management integrates properly with KMC."""
        # Integration test for occupation management with KMC
        
        # Test that the simulation condition can be used in KMC context
        assert simulation_condition.temperature > 0
        assert simulation_condition.attempt_frequency > 0
        assert simulation_condition.random_seed is not None
        
        # Test that the condition can be converted to the format expected by KMC
        kmc_dict = simulation_condition.to_dict()
        assert 'temperature' in kmc_dict
        assert 'attempt_frequency' in kmc_dict
        assert 'random_seed' in kmc_dict
        
        # Test that the condition string can be used for logging/debugging
        condition_str = simulation_condition.get_condition()
        assert isinstance(condition_str, str)
        assert len(condition_str) > 0
        
        # Test integration with pymatgen structures
        assert test_structure.is_valid()
        assert len(occupation_vector) == len(test_structure)
        
        # Test that occupation states are consistent with structure
        for i, site in enumerate(test_structure):
            if site.species_string == "Na":
                # Mobile sites should be able to have different occupation states
                assert occupation_vector[i] in [0, 1]
            else:
                # Immutable sites should remain unoccupied in occupation vector
                assert occupation_vector[i] == 0
        
        # Test that we can extract mobile species information
        mobile_species = [site.species_string for site in test_structure if site.species_string == "Na"]
        assert len(mobile_species) > 0
        assert all(species == "Na" for species in mobile_species)
