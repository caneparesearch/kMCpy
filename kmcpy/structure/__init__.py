"""
Structure module for kMCpy.

This module provides the core modeling infrastructure including lattice structures,
occupation management, basis functions, and local environment comparison.
"""

from .lattice_structure import LatticeStructure
from .vacancy import Vacancy
from .local_lattice_structure import LocalLatticeStructure
from .comparator import SupercellComparator
from .local_environment_comparator import (
    LocalEnvironmentComparator,
    create_environment_comparator,
    match_local_environments
)
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
    "LocalEnvironmentComparator",
    "create_environment_comparator",
    "match_local_environments",
    "BasisFunction",
    "Occupation",
    "OccupationBasis", 
    "ChebyshevBasis",
    "register_basis",
    "get_basis",
    "BASIS_REGISTRY"
]