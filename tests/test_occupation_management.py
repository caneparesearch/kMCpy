"""Tests for occupation/state management with strict config and state APIs."""

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from kmcpy.simulator.config import RuntimeConfig, Configuration, SystemConfig
from kmcpy.simulator.state import State
from kmcpy.structure.active_site_order import ActiveSiteOrder


class DummyEvent:
    """Minimal event-like object for State tests."""

    def __init__(self, from_site: int, to_site: int):
        """Create a dummy migration event."""
        self.mobile_ion_indices = (from_site, to_site)


class TestOccupationManagement:
    """Validate strict runtime/config/state behavior for occupation handling."""

    @pytest.fixture
    def runtime_config(self):
        """Create a basic runtime config for testing."""
        return RuntimeConfig(
            name="test_occupation_management",
            temperature=300.0,
            attempt_frequency=1e13,
            random_seed=42,
            equilibration_passes=10,
            kmc_passes=20,
        )

    @pytest.fixture
    def simulation_config(self, runtime_config):
        """Create a complete simulation config for testing."""
        system_config = SystemConfig(
            structure_file="test_structure.cif",
            mobile_ion_specie="Na",
            dimension=3,
            supercell_shape=(1, 1, 1),
            elementary_hop_distance=2.5,
            mobile_ion_charge=1.0,
            event_file="test_events.json",
        )
        return Configuration(system_config=system_config, runtime_config=runtime_config)

    @pytest.fixture
    def test_structure(self):
        """Create a simple test structure for occupation checks."""
        lattice = Lattice.cubic(5.0)
        species = ["Na", "Zr", "Si", "O"]
        coords = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
        ]
        return Structure(lattice, species, coords)

    @pytest.fixture
    def occupation_vector(self, test_structure):
        """Create an occupation vector aligned with the structure."""
        occupation = np.zeros(len(test_structure), dtype=int)
        for i, site in enumerate(test_structure):
            if site.species_string == "Na":
                occupation[i] = 0
            else:
                occupation[i] = 1
        return occupation

    def test_runtime_config_initialization(self, runtime_config):
        """Test runtime config initialization and field access."""
        assert runtime_config.name == "test_occupation_management"
        assert runtime_config.temperature == 300.0
        assert runtime_config.attempt_frequency == 1e13
        assert runtime_config.random_seed == 42

    def test_runtime_config_validation(self):
        """Test runtime parameter validation."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            RuntimeConfig(temperature=-100.0)

        with pytest.raises(ValueError, match="Attempt frequency must be positive"):
            RuntimeConfig(attempt_frequency=-1e13)

    def test_simulation_config_access(self, simulation_config):
        """Test convenient Configuration accessors."""
        assert simulation_config.temperature == 300.0
        assert simulation_config.attempt_frequency == 1e13
        assert simulation_config.mobile_ion_specie == "Na"
        assert simulation_config.dimension == 3

    def test_occupation_consistency(self, test_structure, occupation_vector):
        """Test that occupation vectors remain structure-consistent."""
        assert len(test_structure) == len(occupation_vector)
        assert test_structure.num_sites > 0
        assert np.sum(occupation_vector == 0) > 0

    def test_simulation_state_tracking(self, occupation_vector):
        """Test mutable simulation state tracking for occupation, time, and step."""
        state = State(occupations=occupation_vector.tolist())

        assert state.time == 0.0
        assert state.step == 0
        assert len(state.occupations) == len(occupation_vector)

        state.time = 1.25
        state.step = 7

        assert state.time == 1.25
        assert state.step == 7

    def test_state_from_occupations_filters_full_structure_values(self, test_structure):
        """Test State.from_occupations handles full-structure occupation input."""
        index_map = ActiveSiteOrder.from_structure_and_mapping(
            test_structure,
            {"Na": ["Na", "X"], "Zr": "Zr", "Si": ["Si", "P"], "O": "O"},
        )

        state = State.from_occupations(
            [0, 1, 1, 1],
            active_site_order=index_map,
        )

        assert state.occupations == [0, 1]

    def test_state_from_file_loads_initial_state(self, tmp_path, test_structure):
        """Test initial-state files load through State.from_file."""
        index_map = ActiveSiteOrder.from_structure_and_mapping(
            test_structure,
            {"Na": ["Na", "X"], "Zr": "Zr", "Si": ["Si", "P"], "O": "O"},
        )
        state_file = tmp_path / "initial_state.json"
        state_file.write_text('{"occupation": [0, 1, 1, 0]}', encoding="utf-8")

        state = State.from_file(str(state_file), active_site_order=index_map)

        assert state.occupations == [0, 1]

    def test_simulation_state_apply_event(self):
        """Test state updates through the strict core apply_event API."""
        state = State(occupations=[0, 1, 1, 1])
        event = DummyEvent(from_site=0, to_site=1)

        state.apply_event(event, dt=0.2)

        assert state.occupations == [1, 0, 1, 1]
        assert state.step == 1
        assert state.time == pytest.approx(0.2)

    def test_simulation_state_serialization(self, occupation_vector):
        """Test strict dict serialization for core mutable state only."""
        state = State(occupations=occupation_vector.tolist(), time=2.0, step=3)

        data = state.as_dict()
        restored = State.from_dict(data)

        assert restored.occupations == state.occupations
        assert restored.time == state.time
        assert restored.step == state.step

    @pytest.mark.integration
    def test_occupation_integration_with_config_and_state(self, simulation_config, occupation_vector):
        """Test clean integration between immutable config and mutable state."""
        state = State(occupations=occupation_vector.tolist())

        assert simulation_config.temperature > 0
        assert simulation_config.attempt_frequency > 0
        assert simulation_config.random_seed is not None

        state.step = 1
        state.time = 0.5

        assert state.step == 1
        assert state.time == pytest.approx(0.5)
        assert len(state.occupations) == len(occupation_vector)
