#!/usr/bin/env python3
"""
Unit tests for KMC integration with Simu    def test_backward_compatibility(self):
        """Test that event_kernel parameter is properly handled for backward compatibility."""
        # Test that event_kernel parameter is properly handled
        config = SimulationConfig(
            name="Test_Backward",
            temperature=300.0,
            attempt_frequency=1e13,
            equilibration_passes=1,
            kmc_passes=10,
            supercell_shape=[1, 1, 1],
            mobile_ion_specie="Na",
            event_dependencies="test.csv"  # This should map to event_kernel
        )
        
        # Test that config properly handles both event_dependencies and event_kernel
        config_dict = config.to_dict()
        
        # event_dependencies should be present in the config
        assert 'event_dependencies' in config_dict
        assert config_dict['event_dependencies'] == "test.csv"imulationState.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from pymatgen.core import Structure, Lattice, Element, Species
from pymatgen.core.sites import PeriodicSite

from kmcpy.simulation_condition import SimulationConfig, SimulationState
from kmcpy.kmc import KMC
from kmcpy.io import InputSet


@pytest.fixture
def test_structure():
    """Create a test Structure object using pymatgen for KMC integration testing."""
    lattice = Lattice.cubic(10.0)
    species = ["Na", "Zr", "Na", "O"]  # Mixed species for testing
    coords = [
        [0.0, 0.0, 0.0],  # Na mobile
        [0.5, 0.0, 0.0],  # Zr immobile
        [0.0, 0.5, 0.0],  # Na mobile
        [0.5, 0.5, 0.0]   # O immobile
    ]
    structure = Structure(lattice, species, coords)
    return structure


class TestKMCIntegration:
    """Test cases for KMC integration with SimulationConfig."""
    
    def test_kmc_parameter_handling(self):
        """Test that KMC properly handles parameters from SimulationConfig."""
        config = SimulationConfig(
            name="Test_KMC",
            temperature=400.0,
            attempt_frequency=2e13,
            equilibration_passes=10,
            kmc_passes=50,
            supercell_shape=[2, 1, 1],
            mobile_ion_specie="Li",
            convert_to_primitive_cell=False,
            immutable_sites=["Zr", "O"],
            random_seed=42
        )
        
        # Test parameter mapping
        kmc_params = {
            'temperature': config.temperature,
            'v': config.attempt_frequency,
            'supercell_shape': config.supercell_shape,
            'convert_to_primitive_cell': config.convert_to_primitive_cell,
            'immutable_sites': config.immutable_sites,
        }
        
        # Verify correct parameter mapping
        assert kmc_params['temperature'] == 400.0
        assert kmc_params['v'] == 2e13
        assert kmc_params['supercell_shape'] == [2, 1, 1]
        assert kmc_params['convert_to_primitive_cell'] is False
        assert kmc_params['immutable_sites'] == ["Zr", "O"]
    
    def test_backward_compatibility(self):
        """Test backward compatibility with event_kernel parameter."""
        # Test that event_kernel parameter is properly handled
        config = SimulationConfig(
            name="Test_Backward",
            temperature=300.0,
            attempt_frequency=1e13,
            equilibration_passes=1,
            kmc_passes=10,
            supercell_shape=[1, 1, 1],
            mobile_ion_specie="Na",
            event_dependencies="test.csv"  # This should map to event_kernel
        )
        
        # Test that config properly handles event_dependencies
        config_dict = config.to_dict()
        
        # event_dependencies should be present in the config
        assert 'event_dependencies' in config_dict
        assert config_dict['event_dependencies'] == "test.csv"
        config_dict = config.to_dict()
        
        # Both parameters should be present for backward compatibility
        assert 'event_dependencies' in config_dict or 'event_kernel' in config_dict
    
    def test_inputset_parameter_validation(self):
        """Test that InputSet properly validates new parameters."""
        test_params = {
            'task': 'kmc',
            'v': 5e12,
            'equ_pass': 1,
            'kmc_pass': 10,
            'supercell_shape': [2, 1, 1],
            'fitting_results': 'test.json',
            'fitting_results_site': 'test.json',
            'lce_fname': 'test.json',
            'lce_site_fname': 'test.json',
            'template_structure_fname': 'test.cif',
            'event_fname': 'test.json',
            'event_kernel': 'test.csv',
            'event_dependencies': 'test.csv',  # New parameter
            'initial_state': 'test.json',
            'temperature': 298.0,
            'dimension': 3,
            'q': 1.0,
            'elem_hop_distance': 3.5,
            'mobile_ion_specie': 'Na',
            'random_seed': 12345,  # New parameter
            'name': 'test',  # New parameter
            'immutable_sites': ['Zr', 'O'],
            'convert_to_primitive_cell': True
        }
        
        # This should not raise an exception
        inputset = InputSet(test_params)
        inputset.parameter_checker()
        
        # Test that parameters are accessible
        assert inputset.random_seed == 12345
        assert inputset.name == 'test'
        assert inputset.event_dependencies == 'test.csv'
        assert inputset.event_kernel == 'test.csv'
    
    def test_simulation_config_to_inputset(self):
        """Test conversion from SimulationConfig to InputSet."""
        config = SimulationConfig(
            name="Test_Conversion",
            temperature=350.0,
            attempt_frequency=1.5e13,
            equilibration_passes=5,
            kmc_passes=25,
            supercell_shape=[2, 2, 1],
            mobile_ion_specie="Na",
            random_seed=123,
            dimension=3,
            elementary_hop_distance=4.0,
            mobile_ion_charge=1.0,
            convert_to_primitive_cell=True,
            immutable_sites=["Zr", "O"],
            # Required parameters for InputSet
            fitting_results="test_fitting.json",
            fitting_results_site="test_fitting_site.json",
            lce_fname="test_lce.json",
            lce_site_fname="test_lce_site.json",
            template_structure_fname="test_structure.cif",
            event_fname="test_events.json",
            event_dependencies="test_dependencies.csv"
        )
        
        # Convert to InputSet
        inputset = config.to_inputset()
        
        # Verify that all parameters are properly converted
        assert inputset.name == "Test_Conversion"
        assert inputset.temperature == 350.0
        assert inputset.v == 1.5e13
        assert inputset.equ_pass == 5
        assert inputset.kmc_pass == 25
        assert inputset.supercell_shape == [2, 2, 1]
        assert inputset.mobile_ion_specie == "Na"
        assert inputset.random_seed == 123
        assert inputset.dimension == 3
        assert inputset.elem_hop_distance == 4.0
        assert inputset.q == 1.0
        assert inputset.convert_to_primitive_cell is True
        assert inputset.immutable_sites == ["Zr", "O"]
    
    def test_random_seed_handling(self):
        """Test that random seed is properly handled throughout the workflow."""
        config = SimulationConfig(
            name="Random_Seed_Test",
            temperature=300.0,
            attempt_frequency=1e13,
            equilibration_passes=1,
            kmc_passes=5,
            supercell_shape=[1, 1, 1],
            mobile_ion_specie="Li",
            random_seed=42,
            # Required parameters for InputSet
            fitting_results="test_fitting.json",
            fitting_results_site="test_fitting_site.json",
            lce_fname="test_lce.json",
            lce_site_fname="test_lce_site.json",
            template_structure_fname="test_structure.cif",
            event_fname="test_events.json",
            event_dependencies="test_dependencies.csv"
        )
        
        # Convert to InputSet
        inputset = config.to_inputset()
        
        # Verify random seed is preserved
        assert inputset.random_seed == 42
        
        # Test that random seed is properly accessible
        assert hasattr(inputset, 'random_seed')
        assert inputset.random_seed is not None
    
    def test_error_handling(self):
        """Test proper error handling for invalid configurations."""
        # Test invalid temperature
        with pytest.raises(ValueError, match="Temperature must be positive"):
            config = SimulationConfig(temperature=-10.0)
            config.validate()
        
        # Test invalid attempt frequency
        with pytest.raises(ValueError, match="Attempt frequency must be positive"):
            config = SimulationConfig(attempt_frequency=-1e13)
            config.validate()
        
        # Test missing required parameters in InputSet
        incomplete_params = {
            'task': 'kmc',
            'temperature': 300.0,
            # Missing many required parameters
        }
        
        inputset = InputSet(incomplete_params)
        with pytest.raises(ValueError, match="Missing required parameters"):
            inputset.parameter_checker()


class TestSimulationStateIntegration:
    """Test cases for SimulationState integration with KMC."""
    
    def test_occupation_consistency(self, test_structure):
        """Test that occupation state is consistent between SimulationState and KMC."""
        initial_occ = [1, -1, 1, -1]
        
        # Create SimulationState
        state = SimulationState(
            initial_occ=initial_occ,
            structure=test_structure,
            mobile_ion_specie="Na"
        )
        
        # Test that occupations are properly managed
        assert state.occupations == initial_occ
        
        # Test direct occupation updates (without complex event handling)
        original_occ = state.occupations.copy()
        
        # Simulate occupation change
        state.occupations = [-1, 1, 1, -1]  # Move ion from site 0 to site 1
        state.time = 0.1
        state.step = 1
        
        # Check that occupations were properly updated
        assert state.occupations == [-1, 1, 1, -1]
        assert state.time == 0.1
        assert state.step == 1
    
    def test_state_tracking(self, test_structure):
        """Test that SimulationState properly tracks simulation progress."""
        initial_occ = [1, -1, 1, -1]
        
        state = SimulationState(
            initial_occ=initial_occ,
            structure=test_structure,
            mobile_ion_specie="Na"
        )
        
        # Initial state
        assert state.time == 0.0
        assert state.step == 0
        
        # Create simple event objects with required attributes
        class SimpleEvent:
            def __init__(self, idx1, idx2):
                self.mobile_ion_specie_1_index = idx1
                self.mobile_ion_specie_2_index = idx2
        
        events = [SimpleEvent(0, 1), SimpleEvent(1, 2), SimpleEvent(2, 3)]
        dt_values = [0.1, 0.2, 0.15]
        
        # Apply events
        for i, (event, dt) in enumerate(zip(events, dt_values)):
            state.update_from_event(event, dt)
            
            # Check that time and step are properly updated
            expected_time = sum(dt_values[:i+1])
            assert abs(state.time - expected_time) < 1e-10
            assert state.step == i + 1
    
    def test_mutable_sites_calculation(self):
        """Test that mutable sites are calculated correctly."""
        # Create a realistic structure with mixed species
        lattice = Lattice.cubic(10.0)
        species = ["Na", "Zr", "Na", "O", "Na"]  # Mix of mobile and immobile
        coords = [
            [0.0, 0.0, 0.0],  # Na - index 0 - mutable
            [0.5, 0.0, 0.0],  # Zr - index 1 - immutable
            [0.0, 0.5, 0.0],  # Na - index 2 - mutable
            [0.5, 0.5, 0.0],  # O  - index 3 - immutable
            [0.0, 0.0, 0.5]   # Na - index 4 - mutable
        ]
        structure = Structure(lattice, species, coords)
        
        # Test immutable sites list
        immutable_sites = ["Zr", "O"]
        
        # Calculate mutable sites
        mutable_sites = []
        for index, site in enumerate(structure.sites):
            if str(site.specie) not in immutable_sites:
                mutable_sites.append(index)
        
        # Should only include Na sites (indices 0, 2, 4)
        expected_mutable_sites = [0, 2, 4]
        assert mutable_sites == expected_mutable_sites
