from .model_file import (
    build_model_file_from_legacy_files,
    build_tabulated_model_file,
    build_tabulated_model_file_from_entries_file,
    load_model_file,
    save_model_file,
    validate_model_file,
)
from .serialization import to_json_compatible as convert

__all__ = [
    "convert",
    "NEBDataLoader",
    "NEBEntry",
    "build_model_file_from_legacy_files",
    "build_tabulated_model_file",
    "build_tabulated_model_file_from_entries_file",
    "load_model_file",
    "save_model_file",
    "validate_model_file",
]


def __getattr__(name):
    if name in {"NEBDataLoader", "NEBEntry"}:
        from .data_loader import NEBDataLoader, NEBEntry

        exports = {
            "NEBDataLoader": NEBDataLoader,
            "NEBEntry": NEBEntry,
        }
        return exports[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
