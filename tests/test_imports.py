import importlib
import pytest

@pytest.mark.parametrize("module_path", [
    "kmcpy.external.structure",
    "kmcpy.external.cif",
    "kmcpy.external.local_env",
    "kmcpy.simulator",  # New simulator module
    "kmcpy.simulator.kmc",
    "kmcpy.event",
    "kmcpy.io",
    "kmcpy.simulator.tracker",
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

def test_simulation_config_classes():
    """Test that simulation config classes can be imported and instantiated."""
    from kmcpy.simulator.config import (
        RuntimeConfig,
        SimulationConfig,
        SystemConfig,
    )
    from tests.test_utils import create_nasicon_config, create_temperature_series
    
    runtime_config = RuntimeConfig(
        name="Runtime_Test",
        temperature=300.0,
        attempt_frequency=1e13,
    )
    system_config = SystemConfig(
        structure_file="test.cif",
        event_file="events.json",
    )
    assert runtime_config.name == "Runtime_Test"
    
    simulation_config = SimulationConfig(
        system_config=system_config,
        runtime_config=runtime_config,
        name="KMC_Test",
        temperature=400.0,
        attempt_frequency=1e13,
        equilibration_passes=1000,
        kmc_passes=5000,
    )
    assert simulation_config.equilibration_passes == 1000
    
    # Test convenience functions exist
    assert callable(create_nasicon_config)
    assert callable(create_temperature_series)

def test_kmc_simulation_config_integration():
    """Test that KMC class has SimulationConfig integration methods."""
    from kmcpy.simulator.kmc import KMC
    
    # Test that new methods exist (updated method names)
    assert hasattr(KMC, 'from_config'), "KMC missing from_config method"
    assert hasattr(KMC, 'run'), "KMC missing run method"
    
    # Test that methods are callable
    assert callable(getattr(KMC, 'from_config'))
    assert callable(getattr(KMC, 'run'))
    
    # Verify that InputSet methods have been removed (no longer supported)
    assert not hasattr(KMC, 'from_inputset'), "KMC should not have from_inputset method (InputSet deprecated)"
