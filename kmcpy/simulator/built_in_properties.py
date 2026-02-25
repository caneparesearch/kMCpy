"""Compatibility wrapper for built-in property functions moved to `property.py`."""

from kmcpy.simulator.property import (  # noqa: F401
    BUILTIN_PROPERTY_FIELDS,
    compute_transport_properties,
)
