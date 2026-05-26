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
from .sites import (
    build_site_index,
    find_site_by_wyckoff_sequence_and_label,
    find_site_by_wyckoff_sequence_label_and_supercell,
    kmc_info_key,
    make_kmc_supercell,
    site_index_key,
    structure_from_sites,
)
from .neighbors import (
    get_cutoff_neighbor_info,
    get_range_cutoff_neighbor_info,
    prepare_cutoff_neighbor_lookup,
    prepare_range_cutoff_neighbor_lookup,
    prepare_range_cutoff_neighbor_lookup_from_preset,
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
    "prepare_range_cutoff_neighbor_lookup_from_preset",
    "prepare_range_cutoff_neighbor_lookup",
    "prepare_cutoff_neighbor_lookup",
    "get_range_cutoff_neighbor_info",
    "get_cutoff_neighbor_info",
    "structure_from_sites",
    "site_index_key",
    "make_kmc_supercell",
    "kmc_info_key",
    "find_site_by_wyckoff_sequence_label_and_supercell",
    "find_site_by_wyckoff_sequence_and_label",
    "build_site_index",
    "BasisFunction",
    "Occupation",
    "OccupationBasis", 
    "ChebyshevBasis",
    "register_basis",
    "get_basis",
    "BASIS_REGISTRY"
]
