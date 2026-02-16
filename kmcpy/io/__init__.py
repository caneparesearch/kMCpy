from .serialization import to_json_compatible as convert

__all__ = [
    "convert",
    "NEBDataLoader",
    "NEBEntry",
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
