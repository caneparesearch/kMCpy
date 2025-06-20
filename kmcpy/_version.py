try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # Python <3.8

__all__ = ["__version__"]

__version__ = version("kmcpy")
