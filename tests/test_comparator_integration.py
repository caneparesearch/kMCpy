from pymatgen.core import Lattice, PeriodicSite, Structure

from kmcpy.structure import LocalLatticeStructure
from kmcpy.structure.cluster import Cluster, ClusterMatcher


def test_local_lattice_structure_exposes_cluster():
    lattice = Lattice.cubic(5.0)
    structure = Structure(
        lattice,
        ["Na", "Na", "Cl", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )
    local_env = LocalLatticeStructure(
        template_structure=structure,
        center=0,
        cutoff=4.0,
        site_mapping={"Na": ["Na", "X"], "Cl": "Cl"},
    )

    cluster = local_env.to_cluster()

    assert cluster.site_indices == tuple(local_env.site_indices)
    assert cluster.distance_matrix.shape == (
        len(local_env.structure),
        len(local_env.structure),
    )
    assert local_env.get_environment_signature() == cluster.signature


def test_cluster_matcher_reorders_neighbor_info():
    lattice = Lattice.cubic(5.0)
    sites = [
        PeriodicSite("Cl", [1, 0, 0], lattice),
        PeriodicSite("Cl", [0, 2, 0], lattice),
        PeriodicSite("Na", [0, 0, 1], lattice),
    ]
    reference = Cluster.from_sites(sites, site_indices=[1, 2, 3])
    candidate = Cluster.from_sites(
        [sites[1], sites[2], sites[0]],
        site_indices=[2, 3, 1],
    )

    match = ClusterMatcher(reference).match(candidate)

    assert match.ordered_candidate_indices == (1, 2, 3)
