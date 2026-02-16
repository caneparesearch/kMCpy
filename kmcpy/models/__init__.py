"""
Model module for kMCpy.

This module provides the core modeling infrastructure including base classes,
local cluster expansion models, and composite models for kinetic Monte Carlo simulations.
"""

from .base import BaseModel
from .local_cluster_expansion import LocalClusterExpansion
from .cluster import Orbit, Cluster
from .composite_lce_model import CompositeLCEModel
from .fitting.fitter import BaseFitter, LCEFitter
from .fitting.registry import register_fitter, get_fitter_for_model

__all__ = [
    'BaseModel',
    'LocalClusterExpansion',
    'CompositeLCEModel',
    'Orbit',
    'Cluster',
    'BaseFitter',
    'LCEFitter',
    'register_fitter',
    'get_fitter_for_model',
]
