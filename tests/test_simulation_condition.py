#!/usr/bin/env python3
"""
Unit tests for SimulationCondition, SimulationConfig, and SimulationState classes.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

from kmcpy.simulator.condition import SimulationCondition
from kmcpy.simulator.config import SimulationConfig, SystemConfig, RuntimeConfig
from kmcpy.simulator.state import SimulationState


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
        """Test SimulationConfig initialization with new clean API."""
        # Test the new smart constructor - no need to create separate configs
        config = SimulationConfig(
            structure_file="test.cif",
            supercell_shape=(2, 1, 1),
            mobile_ion_specie="Na",
            immutable_sites=("Zr", "O"),
            name="NASICON_Test",
            temperature=573.0,
            attempt_frequency=1e13,
            equilibration_passes=100,
            kmc_passes=1000
        )
        
        # Access properties directly - much cleaner!
        assert config.name == "NASICON_Test"
        assert config.temperature == 573.0
        assert config.equilibration_passes == 100
        assert config.kmc_passes == 1000
        assert config.supercell_shape == (2, 1, 1)
        assert config.mobile_ion_specie == "Na"
        assert config.immutable_sites == ("Zr", "O")
        
        # Can still access the underlying configs if needed
        assert config.runtime_config.name == "NASICON_Test"
        assert config.system_config.mobile_ion_specie == "Na"
    
    def test_default_values(self):
        """Test default values for SimulationConfig."""
        config = SimulationConfig(structure_file="test.cif")
        
        # Test defaults through direct property access
        assert config.supercell_shape == (1, 1, 1)
        assert config.mobile_ion_specie == "Li"
        assert config.dimension == 3
        assert config.temperature == 300.0
        assert config.kmc_passes == 10000
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = SimulationConfig(
            structure_file="structure.cif",
            supercell_shape=(2, 2, 2),
            mobile_ion_specie="Li",
            cluster_expansion_file="lce.json",
            event_file="events.json",
            fitting_results_file="test.json",
            name="Test",
            temperature=300.0,
            attempt_frequency=1e13,
            equilibration_passes=10,
            kmc_passes=100
        )
        
        result = config.to_dict()
        
        # Check runtime parameters (uses legacy key names)
        assert result['name'] == 'Test'
        assert result['temperature'] == 300.0
        assert result['equ_pass'] == 10  # Legacy key name
        assert result['kmc_pass'] == 100  # Legacy key name
        assert result['random_seed'] is None
        
        # Check system parameters 
        assert result['supercell_shape'] == [2, 2, 2]  # Converted to list
        assert result['mobile_ion_specie'] == 'Li'
        assert result['structure_file'] == 'structure.cif'
        assert result['cluster_expansion_file'] == 'lce.json'
        assert result['event_file'] == 'events.json'
        assert result['fitting_results_file'] == 'test.json'
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'name': 'Test',
            'temperature': 300.0,
            'attempt_frequency': 1e13,
            'equilibration_passes': 10,
            'kmc_passes': 100,
            'structure_file': 'structure.cif',
            'supercell_shape': (2, 1, 1),
            'mobile_ion_specie': 'Na',
            'cluster_expansion_file': 'lce.json',
            'event_file': 'events.json'
        }
        
        config = SimulationConfig.from_dict(config_dict)
        
        # Test direct property access - much cleaner
        assert config.name == 'Test'
        assert config.temperature == 300.0
        assert config.attempt_frequency == 1e13
        assert config.equilibration_passes == 10
        assert config.kmc_passes == 100
        assert config.structure_file == 'structure.cif'
        assert config.supercell_shape == (2, 1, 1)
        assert config.mobile_ion_specie == 'Na'
        assert config.system_config.cluster_expansion_file == 'lce.json'
        assert config.system_config.event_file == 'events.json'
    
    def test_validation(self):
        """Test validation of SimulationConfig components."""
        # Test valid configuration with new API
        config = SimulationConfig(
            structure_file="test_structure.cif",
            cluster_expansion_file="test_lce.json",
            event_file="test_events.json",
            name="Test",
            temperature=300.0,
            equilibration_passes=10,
            kmc_passes=100
        )
        
        # Should not raise an exception
        assert config.temperature == 300.0
        
        # Test validation happens during construction - invalid temperature
        with pytest.raises(ValueError, match="Temperature must be positive"):
            SimulationConfig(
                structure_file="test.cif",
                name="Test",
                temperature=-10.0,  # Invalid negative temperature
                equilibration_passes=10,
                kmc_passes=100
            )


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
        from kmcpy.structure.basis import Occupation
        initial_occ = Occupation([1, -1, 1, -1], basis='chebyshev')
        
        state = SimulationState(
            occupations=initial_occ,
        )
        
        assert state.occupations.values == [1, -1, 1, -1]
        assert state.time == 0.0
        assert state.step == 0
    
    def test_occupation_management(self):
        """Test occupation state management."""
        from kmcpy.structure.basis import Occupation
        initial_occ = Occupation([1, -1, 1, -1], basis='chebyshev')
        
        state = SimulationState(
            occupations=initial_occ,
        )
        
        # Test initial occupations
        assert state.occupations.values == [1, -1, 1, -1]
        
        # Test occupation update
        state.occupations.flip_inplace([0])  # Flip site 0: 1 -> -1
        state.occupations.flip_inplace([1])  # Flip site 1: -1 -> 1
        assert state.occupations.values == [-1, 1, 1, -1]
    
    def test_update_from_event(self):
        """Test updating state from an event."""
        from kmcpy.structure.basis import Occupation
        initial_occ = Occupation([1, -1, 1, -1], basis='chebyshev')
        
        state = SimulationState(
            occupations=initial_occ,
        )
        
        # Create event object for the update
        from kmcpy.event.event import Event
        event = Event(
            mobile_ion_indices=(0, 1),  # from_site=0, to_site=1
            local_env_indices=[2, 3, 4]  # local environment (not used in apply_event)
        )
        dt = 0.1
        
        # Update from event object
        state.apply_event(event, dt)
        
        # Check that occupations were flipped
        assert state.occupations.values == [-1, 1, 1, -1]
        assert state.time == 0.1
        assert state.step == 1

        
    def test_copy(self):
        """Test copying simulation state."""
        from kmcpy.structure.basis import Occupation
        initial_occ = Occupation([1, -1, 1, -1], basis='chebyshev')
        
        state = SimulationState(
            occupations=initial_occ,
        )
        
        # Update original state
        state.time = 10.0
        state.step = 5
        
        # Create copy
        state_copy = state.copy()
        
        # Check that copy is independent
        assert state_copy.occupations.values == [1, -1, 1, -1]
        assert state_copy.time == state.time
        assert state_copy.step == state.step
        
        # Modify copy
        state_copy.occupations.values[0] = -1
        state_copy.time = 20.0
        
        # Original should be unchanged
        assert state.occupations.values[0] == 1
        assert state.time == 10.0
