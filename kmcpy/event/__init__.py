"""
Event module for kMCpy

This module defines the Event class and EventLib for managing migration events in kinetic Monte Carlo simulations.
"""

from .base import Event, EventLib
from .generators import EventGenerator, NeighborInfoMatcher, ModernEventGenerator


def create_modern_event_generator():
    """
    Factory function to create a ModernEventGenerator instance.

    Returns:
        ModernEventGenerator: A new instance of ModernEventGenerator
    """
    return ModernEventGenerator()


def generate_events_modern(*args, **kwargs):
    """
    Convenience function for generating events using the modern approach.

    This creates a ModernEventGenerator instance and calls its generate_events method.
    All arguments are forwarded to ModernEventGenerator.generate_events().

    Returns:
        EventLib: Generated events with dependencies
    """
    generator = ModernEventGenerator()
    return generator.generate_events(*args, **kwargs)


__all__ = [
    'Event',
    'EventLib',
    'EventGenerator',
    'NeighborInfoMatcher',
    'ModernEventGenerator',
    'create_modern_event_generator',
    'generate_events_modern',
]
