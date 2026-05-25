import pytest
from pymatgen.core import Lattice, Structure

from kmcpy.structure.cluster import Cluster, ClusterMatcher


def _sites(species, coords):
    lattice = Lattice.cubic(20)
    structure = Structure(
        lattice,
        species,
        coords,
        coords_are_cartesian=True,
    )
    return list(structure)


def _neighbor_info(sites, site_indices):
    return [
        {
            "site": site,
            "image": (0, 0, 0),
            "weight": 1.0,
            "site_index": site_index,
            "local_index": index,
        }
        for index, (site, site_index) in enumerate(zip(sites, site_indices))
    ]


def test_cluster_matcher_returns_reference_order_for_permuted_sites():
    sites = _sites(
        ["Na", "Cl", "Cl"],
        [[0, 0, 0], [1, 0, 0], [0, 2, 0]],
    )
    reference = Cluster.from_sites(sites, site_indices=[10, 20, 30])
    candidate = Cluster.from_sites(
        [sites[2], sites[0], sites[1]],
        site_indices=[30, 10, 20],
    )

    match = ClusterMatcher(reference).match(candidate, require_unique=True)

    assert match.reference_to_candidate == (1, 2, 0)
    assert match.candidate_to_reference == (2, 0, 1)
    assert match.ordered_candidate_indices == (10, 20, 30)
    assert match.is_unique


def test_cluster_matcher_rejects_species_mismatch():
    reference = Cluster(
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [1, 0, 0]],
    )
    candidate = Cluster(
        species=["Na", "Br"],
        coords=[[0, 0, 0], [1, 0, 0]],
    )

    with pytest.raises(ValueError, match="signatures do not match"):
        ClusterMatcher(reference).match(candidate)


def test_cluster_matcher_detects_ambiguous_symmetric_sites():
    reference = Cluster(
        species=["Na", "Na"],
        coords=[[0, 0, 0], [1, 0, 0]],
    )
    candidate = Cluster(
        species=["Na", "Na"],
        coords=[[1, 0, 0], [0, 0, 0]],
    )

    match = ClusterMatcher(reference).match(candidate)
    assert not match.is_unique

    with pytest.raises(ValueError, match="ambiguous"):
        ClusterMatcher(reference).match(candidate, require_unique=True)


def test_cluster_roles_disambiguate_symmetric_sites():
    reference = Cluster(
        species=["Na", "Na"],
        coords=[[0, 0, 0], [1, 0, 0]],
        site_indices=[0, 1],
        roles=["from_site", "to_site"],
    )
    candidate = Cluster(
        species=["Na", "Na"],
        coords=[[1, 0, 0], [0, 0, 0]],
        site_indices=[10, 11],
        roles=["to_site", "from_site"],
    )

    match = ClusterMatcher(reference).match(candidate, require_unique=True)

    assert match.reference_to_candidate == (1, 0)
    assert match.ordered_candidate_indices == (11, 10)
    assert match.is_unique


def test_neighbor_info_can_be_matched_directly():
    sites = _sites(
        ["Na", "Cl", "Cl"],
        [[0, 0, 0], [1, 0, 0], [0, 2, 0]],
    )
    reference_neighbors = _neighbor_info(sites, [10, 20, 30])
    candidate_neighbors = _neighbor_info([sites[2], sites[0], sites[1]], [30, 10, 20])

    match = ClusterMatcher(
        Cluster.from_neighbor_info(reference_neighbors)
    ).match(Cluster.from_neighbor_info(candidate_neighbors))

    matched = [candidate_neighbors[index] for index in match.reference_to_candidate]
    assert [neighbor["site_index"] for neighbor in matched] == [10, 20, 30]


def test_cluster_equivalence_uses_cluster_matcher():
    sites = _sites(
        ["Na", "Cl", "Cl"],
        [[0, 0, 0], [1, 0, 0], [0, 2, 0]],
    )
    cluster = Cluster([0, 1, 2], sites)
    permuted_cluster = Cluster([2, 0, 1], [sites[2], sites[0], sites[1]])

    assert cluster == permuted_cluster
