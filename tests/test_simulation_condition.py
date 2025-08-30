#!/usr/bin/env python3
"""
Unit tests for SimulationCondition, SimulationConfig, and SimulationState classes.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

from kmcpy.simulator.condition import SimulationCondition, SimulationConfig
from kmcpy.simulator.state import SimulationState
from kmcpy.io.io import InputSet


class TestSimulationCondition:
    """Test cases for SimulationCondition base class."""
    
    def test_basic_initialization(self):
        """Test basic initialization of SimulationCondition."""
        condition = SimulationCondition(
            name="test_condition",
            temperature=300.0,
            attempt_frequency=1e13,
            random_seed=42
        )
        
        assert condition.name == "test_condition"
        assert condition.temperature == 300.0
        assert condition.attempt_frequency == 1e13
        assert condition.random_seed == 42
    
    def test_default_values(self):
        """Test default values for SimulationCondition."""
        condition = SimulationCondition()
        
        assert condition.name == "DefaultSimulation"
        assert condition.temperature == 300.0
        assert condition.attempt_frequency == 1e13
        assert condition.random_seed is None
    
    def test_validation(self):
        """Test validation of SimulationCondition parameters."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            SimulationCondition(temperature=-10.0)
        
        with pytest.raises(ValueError, match="Attempt frequency must be positive"):
            SimulationCondition(attempt_frequency=-1e13)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        condition = SimulationCondition(
            name="test",
            temperature=400.0,
            attempt_frequency=2e13,
            random_seed=123
        )
        
        expected = {
            'name': 'test',
            'temperature': 400.0,
            'attempt_frequency': 2e13,
            'random_seed': 123
        }
        
        assert condition.to_dict() == expected


class TestSimulationConfig:
    """Test cases for SimulationConfig class."""
    
    def test_initialization(self):
        """Test SimulationConfig initialization."""
        config = SimulationConfig(
            name="NASICON_Test",
            temperature=573.0,
            attempt_frequency=1e13,
            equilibration_passes=100,
            kmc_passes=1000,
            supercell_shape=[2, 1, 1],
            mobile_ion_specie="Na",
            initial_occ=[1, -1, 1, -1],
            immutable_sites=["Zr", "O"]
        )
        
        assert config.name == "NASICON_Test"
        assert config.temperature == 573.0
        assert config.equilibration_passes == 100
        assert config.kmc_passes == 1000
        assert config.supercell_shape == [2, 1, 1]
        assert config.mobile_ion_specie == "Na"
        assert config.initial_occ == [1, -1, 1, -1]
        assert config.immutable_sites == ["Zr", "O"]
    
    def test_default_values(self):
        """Test default values for SimulationConfig."""
        config = SimulationConfig()
        
        assert config.supercell_shape == [1, 1, 1]
        assert config.initial_occ == []
        assert config.mobile_ion_specie == "Li"
        assert config.dimension == 3
        assert config.mobile_ion_charge == 1.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = SimulationConfig(
            name="Test",
            temperature=300.0,
            attempt_frequency=1e13,
            equilibration_passes=10,
            kmc_passes=100,
            supercell_shape=[2, 2, 2],
            mobile_ion_specie="Li",
            fitting_results="test.json",
            lce_fname="lce.json",
            template_structure_fname="structure.cif",
            event_fname="events.json",
            event_dependencies="deps.csv"
        )
        
        result = config.to_dict()
        
        assert result['name'] == 'Test'
        assert result['temperature'] == 300.0
        assert result['attempt_frequency'] == 1e13
        assert result['equ_pass'] == 10
        assert result['kmc_pass'] == 100
        assert result['supercell_shape'] == [2, 2, 2]
        assert result['mobile_ion_specie'] == 'Li'
        assert result['fitting_results'] == 'test.json'
        assert result['lce_fname'] == 'lce.json'
        assert result['template_structure_fname'] == 'structure.cif'
        assert result['event_fname'] == 'events.json'
        assert result['event_dependencies'] == 'deps.csv'
        assert result['event_kernel'] == 'deps.csv'  # Backward compatibility
        assert result['task'] == 'kmc'
    
    def test_to_inputset_conversion(self):
        """Test conversion to InputSet."""
        config = SimulationConfig(
            name="Test",
            temperature=300.0,
            attempt_frequency=1e13,
            equilibration_passes=10,
            kmc_passes=100,
            supercell_shape=[2, 1, 1],
            mobile_ion_specie="Na",
            random_seed=42,
            initial_occ=[1, -1, 1, -1],  # Test with initial_occ
            immutable_sites=["Zr", "O"],
            # Add required file parameters
            fitting_results='test_fitting.json',
            fitting_results_site='test_fitting_site.json',
            lce_fname='test_lce.json',
            lce_site_fname='test_lce_site.json',
            template_structure_fname='test_structure.cif',
            event_fname='test_events.json',
            event_dependencies='test_dependencies.csv'
        )
        
        # This will create a temporary file for initial_state
        inputset = config.to_inputset()
        
        assert hasattr(inputset, 'name')
        assert inputset.name == 'Test'
        assert inputset.temperature == 300.0
        assert inputset.attempt_frequency == 1e13
        assert inputset.equ_pass == 10
        assert inputset.kmc_pass == 100
        assert inputset.supercell_shape == [2, 1, 1]
        assert inputset.mobile_ion_specie == 'Na'
        assert inputset.random_seed == 42
        assert inputset.immutable_sites == ["Zr", "O"]
    
    def test_validation(self):
        """Test validation of SimulationConfig."""
        config = SimulationConfig(
            name="Test",
            temperature=300.0,
            attempt_frequency=1e13,
            equilibration_passes=10,
            kmc_passes=100,
            # Add required file parameters
            fitting_results='test_fitting.json',
            fitting_results_site='test_fitting_site.json',
            lce_fname='test_lce.json',
            lce_site_fname='test_lce_site.json',
            template_structure_fname='test_structure.cif',
            event_fname='test_events.json',
            event_dependencies='test_dependencies.csv',
            initial_occ=[1, -1, 1, -1]  # Required: initial_occ or initial_state
        )
        
        # Should not raise an exception
        config.validate()
        
        # Test with invalid temperature
        config.temperature = -10.0
        with pytest.raises(ValueError):
            config.validate()


@pytest.fixture
def test_structure():
    """Create a test Structure object using pymatgen."""
    try:
        from pymatgen.core import Structure, Lattice
        lattice = Lattice.cubic(10.0)
        species = ["Na", "Na", "Na", "Na"]
        coords = [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0]
        ]
        structure = Structure(lattice, species, coords)
        return structure
    except ImportError:
        # Fallback if pymatgen not available
        class BasicStructure:
            def __init__(self):
                self.frac_coords = np.array([
                    [0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.5, 0.5, 0.0]
                ])
                self.volume = 1000.0
        return BasicStructure()


class TestSimulationState:
    """Test cases for SimulationState class."""
    
    def test_initialization(self):
        """Test SimulationState initialization."""
        initial_occ = [1, -1, 1, -1]
        
        state = SimulationState(
            occupations=initial_occ,
        )
        
        assert np.array_equal(state.occupations, initial_occ)
        assert state.time == 0.0
        assert state.step == 0
    
    def test_occupation_management(self):
        """Test occupation state management."""
        initial_occ = [1, -1, 1, -1]
        
        state = SimulationState(
            occupations=initial_occ,
        )
        
        # Test initial occupations
        assert np.array_equal(state.occupations, [1, -1, 1, -1])
        
        # Test occupation update
        state.occupations[0] = -1
        state.occupations[1] = 1
        assert np.array_equal(state.occupations, [-1, 1, 1, -1])
    
    def test_update_from_event(self):
        """Test updating state from an event."""
        initial_occ = [1, -1, 1, -1]
        
        state = SimulationState(
            occupations=initial_occ,
        )
        
        # Apply event directly without needing event object
        from_site, to_site = 0, 1
        dt = 0.1
        
        # Update from event
        state.apply_event(from_site, to_site, dt)
        
        # Check that occupations were flipped
        assert np.array_equal(state.occupations, [-1, 1, 1, -1])
        assert state.time == 0.1
        assert state.step == 1

        
    def test_copy(self):
        """Test copying simulation state."""
        initial_occ = [1, -1, 1, -1]
        
        state = SimulationState(
            occupations=initial_occ,
        )
        
        # Update original state
        state.time = 10.0
        state.step = 5
        
        # Create copy
        state_copy = state.copy()
        
        # Check that copy is independent
        assert np.array_equal(state_copy.occupations, state.occupations)
        assert state_copy.time == state.time
        assert state_copy.step == state.step
        
        # Modify copy
        state_copy.occupations[0] = -1
        state_copy.time = 20.0
        
        # Original should be unchanged
        assert state.occupations[0] == 1
        assert state.time == 10.0
