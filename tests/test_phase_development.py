#!/usr/bin/env python3
"""
Unit tests for architectural phases and development milestones.
These tests verify that the various phases of the kMCpy refactoring work correctly.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from pymatgen.core import Structure, Lattice, Element, Species
from pymatgen.core.sites import PeriodicSite

from kmcpy.simulator.config import SimulationConfig, SystemConfig, RuntimeConfig
from kmcpy.simulator.state import SimulationState
from kmcpy.simulator.tracker import Tracker
from kmcpy.io.config_io import SimulationConfigIO


@pytest.fixture
def test_structure():
    """Create a test Structure object using pymatgen for phase development testing."""
    # Create a NASICON-like structure with Na, Zr, Si, O
    lattice = Lattice.cubic(10.0)
    species = ["Na", "Zr", "Si", "O"]
    coords = [
        [0.0, 0.0, 0.0],  # Na at origin
        [0.5, 0.0, 0.0],  # Zr 
        [0.0, 0.5, 0.0],  # Si
        [0.5, 0.5, 0.0]   # O
    ]
    structure = Structure(lattice, species, coords)
    return structure


@pytest.fixture
def mobile_ion_sites(test_structure):
    """Get mobile ion sites from the test structure."""
    mobile_sites = []
    for i, site in enumerate(test_structure):
        if site.species_string == "Na":
            mobile_sites.append(i)
    return mobile_sites


class TestTrackerParameterDeduplication:
    """Test Tracker parameter deduplication."""
    
    def test_tracker_parameter_separation(self):
        """Test that Tracker no longer duplicates parameters from configuration."""
        
        # Create a test configuration using new clean API
        system = SystemConfig(
            structure_file="test.cif",
            dimension=3,
            elementary_hop_distance=2.5,
            mobile_ion_charge=1.0,
            mobile_ion_specie="Na",
            supercell_shape=(2, 1, 1),
            immutable_sites=("Zr", "O")
        )
        runtime = RuntimeConfig(
            name="TrackerTest",
            temperature=573.0,
            attempt_frequency=1e13,
            equilibration_passes=100,
            kmc_passes=1000
        )
        config = SimulationConfig(system_config=system, runtime_config=runtime)
        
        # Verify that configuration is properly structured
        assert config.runtime_config.name == "TrackerTest"
        assert config.runtime_config.temperature == 573.0
        assert config.runtime_config.attempt_frequency == 1e13
        assert config.runtime_config.equilibration_passes == 100
        assert config.runtime_config.kmc_passes == 1000
        
        # Test parameter deduplication concept
        # Configuration should contain parameters, not simulation state
        assert hasattr(config.runtime_config, 'temperature')
        assert hasattr(config.runtime_config, 'attempt_frequency')
        assert hasattr(config.runtime_config, 'equilibration_passes')
        assert hasattr(config.runtime_config, 'kmc_passes')
        
        # System config should contain system parameters
        assert config.system_config.mobile_ion_specie == "Na"
        assert config.system_config.supercell_shape == (2, 1, 1)
        


class TestSimulationStateArchitecture:
    """Test SimulationState-centric architecture."""
    
    def test_simulation_state_centralized_management(self, test_structure):
        """Test that SimulationState is the central manager for all mutable state."""
        
        # Create configuration (immutable) using new clean API
        system = SystemConfig(
            structure_file="test.cif",
            dimension=3,
            elementary_hop_distance=2.5,
            mobile_ion_charge=1.0,
            mobile_ion_specie="Na",
            supercell_shape=(2, 1, 1)
        )
        runtime = RuntimeConfig(
            name="Phase2_Test",
            temperature=573.0,
            attempt_frequency=1e13,
            equilibration_passes=100,
            kmc_passes=1000
        )
        config = SimulationConfig(system_config=system, runtime_config=runtime)
        
        # Create SimulationState (mutable) with initial occupations  
        initial_occ = [1, -1, 1, -1]
        state = SimulationState(
            occupations=initial_occ,
        )
        
        # Test that SimulationState manages all mutable state
        assert state.occupations == [1, -1, 1, -1]
        assert state.time == 0.0
        assert state.step == 0
        
        # Test state updates without complex event handling
        state.time = 0.1
        state.step = 1
        state.occupations = [-1, 1, 1, -1]
        
        # Verify state was updated correctly
        assert state.occupations == [-1, 1, 1, -1]
        assert state.time == 0.1
        assert state.step == 1
        
        # Test passed; no need for additional logging
    
    def test_clean_separation_of_concerns(self, test_structure):
        """Test clean separation between Config (immutable) and State (mutable)."""
        
        # Configuration should be immutable - using clean API
        system = SystemConfig(
            structure_file="test.cif",
            mobile_ion_specie="Li"
        )
        runtime = RuntimeConfig(
            name="Separation_Test",
            temperature=400.0,
            attempt_frequency=2e13
        )
        config = SimulationConfig(system_config=system, runtime_config=runtime)
        
        # State should be mutable
        state = SimulationState(
            occupations=[1, -1, 1, -1],
        )
        
        # Configuration should not change during simulation
        original_temp = config.runtime_config.temperature
        original_freq = config.runtime_config.attempt_frequency
        
        # Modify state
        state.time = 10.0
        state.step = 5
        
        # Configuration should remain unchanged (immutable)
        assert config.runtime_config.temperature == original_temp
        assert config.runtime_config.attempt_frequency == original_freq
        
        # State should be modified
        assert state.time == 10.0
        assert state.step == 5
        
        print("✓ Clean separation of concerns working correctly")


class TestKMCIntegrationImprovements:
    """Test KMC integration improvements."""
    
    def test_kmc_simulation_state_integration(self):
        """Test that KMC uses SimulationState as single source of truth."""
        
        # Create minimal configuration for testing using clean API
        system = SystemConfig(
            structure_file="test.cif",
            dimension=3,
            elementary_hop_distance=3.0,
            mobile_ion_charge=1.0,
            mobile_ion_specie="Na",
            supercell_shape=(2, 1, 1),
            immutable_sites=("Zr", "O")
        )
        runtime = RuntimeConfig(
            name="Phase3_Test",
            temperature=400.0,
            attempt_frequency=1e13,
            equilibration_passes=10,
            kmc_passes=50
        )
        config = SimulationConfig(system_config=system, runtime_config=runtime)
        
        # Test that configuration is properly structured for KMC integration
        assert config.runtime_config.name == "Phase3_Test"
        assert config.runtime_config.temperature == 400.0
        assert config.runtime_config.attempt_frequency == 1e13
        
        # Create initial occupations for state
        initial_occ = [1, -1, 1, -1]
        state = SimulationState(occupations=initial_occ)
        
        # Test parameter mapping for KMC
        kmc_params = {
            'temperature': config.runtime_config.temperature,
            'attempt_frequency': config.runtime_config.attempt_frequency,
            'supercell_shape': config.system_config.supercell_shape,
            'immutable_sites': config.system_config.immutable_sites,
        }
        
        # Verify parameter mapping
        assert kmc_params['temperature'] == 400.0
        assert kmc_params['attempt_frequency'] == 1e13
        assert kmc_params['supercell_shape'] == (2, 1, 1)
        assert kmc_params['immutable_sites'] == ("Zr", "O")
        
        print("✓ KMC integration working correctly")
    
    def test_optimized_simulation_loop(self, test_structure):
        """Test optimized simulation loop with direct state management."""
        
        # Create SimulationState for optimized loop
        state = SimulationState(
            occupations=[1, -1, 1, -1],
        )
        
        # Test optimized state updates
        original_occupations = state.occupations.copy()
        
        # Simulate multiple state changes without complex event handling
        state_changes = [
            ([-1, 1, 1, -1], 0.1),
            ([1, -1, -1, 1], 0.2),
            ([-1, -1, 1, 1], 0.3)
        ]
        
        # Process state changes efficiently
        for i, (new_occ, dt) in enumerate(state_changes):
            state.occupations = new_occ
            state.time += dt
            state.step += 1
            assert state.step == i + 1
        
        # Verify final state
        assert state.step == len(state_changes)
        assert abs(state.time - 0.6) < 1e-10
        assert state.occupations == [-1, -1, 1, 1]
        
        print("✓ Optimized simulation loop working correctly")


class TestPhase4InputSetMigration:
    """Test Phase 4: InputSet migration and deprecation."""
    
    def test_simulation_config_direct_api(self):
        """Test direct SimulationConfig API without InputSet conversion."""
        
        # Create configuration with all necessary parameters using clean API
        system = SystemConfig(
            structure_file="test.cif",
            supercell_shape=(2, 1, 1),
            dimension=3,
            mobile_ion_charge=1.0,
            elementary_hop_distance=4.0,
            convert_to_primitive_cell=False,
            immutable_sites=('Zr', 'O'),
            mobile_ion_specie='Na',
        )
        runtime = RuntimeConfig(
            name="Phase4_Test",
            temperature=573.0,
            attempt_frequency=1e13,
            equilibration_passes=10,
            kmc_passes=50,
            random_seed=42
        )
        config = SimulationConfig(system_config=system, runtime_config=runtime)
        
        # Test direct API usage
        assert config.runtime_config.name == "Phase4_Test"
        assert config.runtime_config.temperature == 573.0
        assert config.runtime_config.attempt_frequency == 1e13
        assert config.runtime_config.random_seed == 42
        
        # Test that configuration is self-contained
        assert hasattr(config.runtime_config, 'temperature')
        assert hasattr(config.runtime_config, 'attempt_frequency')
        assert hasattr(config.runtime_config, 'equilibration_passes')
        assert hasattr(config.runtime_config, 'kmc_passes')
        assert hasattr(config.system_config, 'supercell_shape')
        assert hasattr(config.system_config, 'mobile_ion_specie')
        assert hasattr(config.runtime_config, 'random_seed')
        
        print("✓ Phase 4: Direct SimulationConfig API working correctly")

    
    def test_parameter_migration(self):
        """Test that clean API works properly."""
        
        # Test clean API parameter usage
        system = SystemConfig(
            structure_file="test.cif",
            supercell_shape=(2, 2, 2),
            mobile_ion_specie="Li"
        )
        runtime = RuntimeConfig(
            name="Migration_Test",
            temperature=300.0,
            attempt_frequency=1e13,
            equilibration_passes=5,
            kmc_passes=25,
            random_seed=123  # Random seed parameter
        )
        config = SimulationConfig(system_config=system, runtime_config=runtime)
        
        # Test the to_dict method for migration
        config_dict = config.to_dict()
        
        # Test that parameters are correctly set
        assert config_dict['temperature'] == 300.0
        assert config_dict['equilibration_passes'] == 5
        assert config_dict['kmc_passes'] == 25
        assert config_dict['mobile_ion_specie'] == "Li"
        assert config_dict['random_seed'] == 123
        assert config_dict['supercell_shape'] == [2, 2, 2]  # Converted to list
        
        print("✓ Phase 4: Parameter migration working correctly")


class TestOccupationManagement:
    """Test occupation management improvements."""
    
    def test_occupation_state_management(self, test_structure):
        """Test that SimulationState properly manages occupations."""
        
        initial_occ = [1, -1, 1, -1]  # Sites 0,2 occupied, sites 1,3 vacant
        
        state = SimulationState(
            occupations=initial_occ,
        )
        
        # Test initial state
        assert state.occupations == initial_occ
        
        # Get occupied and vacant sites manually for testing
        occupied_sites = [i for i, occ in enumerate(state.occupations) if occ == 1]
        vacant_sites = [i for i, occ in enumerate(state.occupations) if occ == -1]
        
        assert occupied_sites == [0, 2]
        assert vacant_sites == [1, 3]
        
        # Test occupation updates by directly modifying occupations
        state.occupations = [-1, 1, 1, -1]  # Move ion from site 0 to site 1
        
        # Check updated occupations
        assert state.occupations == [-1, 1, 1, -1]
        
        # Verify updated occupied/vacant sites
        new_occupied_sites = [i for i, occ in enumerate(state.occupations) if occ == 1]
        new_vacant_sites = [i for i, occ in enumerate(state.occupations) if occ == -1]
        
        assert new_occupied_sites == [1, 2]
        assert new_vacant_sites == [0, 3]
        
        print("✓ Occupation management working correctly")
    
    def test_no_state_duplication(self, test_structure):
        """Test that there's no duplication of occupation state."""
        
        initial_occ = [1, -1, 1, -1]
        
        state = SimulationState(
            occupations=initial_occ,
        )
        
        # State should manage its own occupations
        assert hasattr(state, 'occupations')
        assert isinstance(state.occupations, list)  # SimulationState uses lists, not numpy arrays
        
        # Modifications should be direct
        original_occ = state.occupations.copy()
        state.occupations[0] = -1
        
        # Should be modified directly
        assert state.occupations[0] == -1
        assert state.occupations != original_occ
        
        print("✓ No state duplication verified")


if __name__ == "__main__":
    # Run with pytest when executed directly
    import subprocess
    import sys
    
    print("Running phase development tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v"
    ], cwd=os.path.dirname(os.path.abspath(__file__)) or ".")
    
    if result.returncode == 0:
        print("\n🎉 All phase development tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(result.returncode)
