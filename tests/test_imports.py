import importlib
import pytest

@pytest.mark.parametrize("module_path", [
    "kmcpy.external.structure",
    "kmcpy.external.cif",
    "kmcpy.external.local_env",
    "kmcpy.simulation",  # New simulation module
    "kmcpy.kmc",
    "kmcpy.event",
    "kmcpy.io",
    "kmcpy.simulation.tracker",
])
def test_module_imports(module_path):
    """Test that modules can be imported without circular import errors."""
    try:
        importlib.import_module(module_path)
    except ImportError as e:
        pytest.fail(f"Failed to import {module_path}: {e}")

@pytest.mark.parametrize("module_path", [
    "kmcpy.external",
])
def test_top_level_imports(module_path):
    try:
        importlib.import_module(module_path)
    except ImportError as e:
        pytest.fail(f"Failed to import top-level {module_path}: {e}")

def test_simulation_condition_classes():
    """Test that SimulationCondition classes can be imported and instantiated."""
    from kmcpy.simulation.condition import (
        SimulationCondition, 
        SimulationConfig,
    )
    from tests.test_utils import create_nasicon_config, create_temperature_series
    
    # Test basic instantiation
    condition = SimulationCondition(
        name="Test",
        temperature=300.0,
        attempt_frequency=1e13
    )
    assert condition.name == "Test"
    
    kmc_condition = SimulationConfig(
        name="KMC_Test",
        temperature=400.0,
        attempt_frequency=1e13,
        equilibration_passes=1000,
        kmc_passes=5000
    )
    assert kmc_condition.equilibration_passes == 1000
    
    # Test convenience functions exist
    assert callable(create_nasicon_config)
    assert callable(create_temperature_series)

def test_kmc_simulation_condition_integration():
    """Test that KMC class has SimulationCondition integration methods."""
    from kmcpy.kmc import KMC
    
    # Test that new methods exist (updated method names)
    assert hasattr(KMC, 'from_config'), "KMC missing from_config method"
    assert hasattr(KMC, 'from_inputset'), "KMC missing from_inputset method"
    assert hasattr(KMC, 'run'), "KMC missing run method"
    
    # Test that methods are callable
    assert callable(getattr(KMC, 'from_config'))
    assert callable(getattr(KMC, 'from_inputset'))
    assert callable(getattr(KMC, 'run'))
