"""
Structure module for kMCpy.

This module provides the core modeling infrastructure including lattice structures,
occupation management, basis functions, and local environment comparison.
"""

from .lattice_structure import LatticeStructure
from .vacancy import Vacancy
from .active_site_index_map import ActiveSiteIndexMap
from .local_lattice_structure import LocalLatticeStructure
from .cluster import (
    Cluster,
    Orbit,
    ClusterMatcher,
    ClusterMatch,
    match_clusters,
)
from .local_site_ordering import (
    LocalSiteOrderingConvention,
    ordered_site_hash,
    ordered_site_signature,
)
from .comparator import SupercellComparator
from .local_environment_enumerator import (
    LocalEnvironmentEnumeration,
    NEBEndpointPair,
    enumerate_local_environments,
    enumerate_neb_endpoint_pairs,
    generate_neb_endpoint_pair,
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
    "ActiveSiteIndexMap",
    "LocalLatticeStructure",
    "Cluster",
    "Orbit",
    "ClusterMatcher",
    "ClusterMatch",
    "match_clusters",
    "LocalSiteOrderingConvention",
    "ordered_site_hash",
    "ordered_site_signature",
    "SupercellComparator",
    "LocalEnvironmentEnumeration",
    "NEBEndpointPair",
    "enumerate_local_environments",
    "enumerate_neb_endpoint_pairs",
    "generate_neb_endpoint_pair",
    "BasisFunction",
    "Occupation",
    "OccupationBasis", 
    "ChebyshevBasis",
    "register_basis",
    "get_basis",
    "BASIS_REGISTRY"
]
