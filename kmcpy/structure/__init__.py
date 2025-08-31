"""
Structure module for kMCpy.

This module provides the core modeling infrastructure including lattice structures,
occupation management, and basis functions.
"""

from .lattice_structure import LatticeStructure
from .vacancy import Vacancy
from .local_lattice_structure import LocalLatticeStructure
from .comparator import SupercellComparator
from .basis import (
    BasisFunction, 
    Occupation, 
    OccupationBasis, 
    ChebyshevBasis,
    register_basis,
    get_basis,
    BASIS_REGISTRY
)

__all__ = [
    "LatticeStructure",
    "Vacancy",
    "LocalLatticeStructure",
    "SupercellComparator",
    "BasisFunction",
    "Occupation",
    "OccupationBasis", 
    "ChebyshevBasis",
    "register_basis",
    "get_basis",
    "BASIS_REGISTRY"
]