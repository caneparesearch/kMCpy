import importlib
import pytest

@pytest.mark.parametrize("module_path", [
    "kmcpy.external.structure",
    "kmcpy.external.cif",
    "kmcpy.external.local_env",
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
