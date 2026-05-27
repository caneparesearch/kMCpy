import numpy as np
from pymatgen.core import Lattice, Structure

from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
from kmcpy.structure.basis import ChebyshevBasis, Occupation
from kmcpy.structure.lattice_structure import LatticeStructure
from kmcpy.structure.local_lattice_structure import LocalLatticeStructure
from kmcpy.structure import enumerate_local_environments


def test_chebyshev_basis_uses_q_minus_one_functions_for_multicomponent_sites():
    basis = ChebyshevBasis(max_states=4)

    assert basis.uses_state_indices
    assert basis.valid_values == {0, 1, 2, 3}
    assert basis.num_site_basis_functions(4) == 3
    np.testing.assert_allclose(
        basis.site_basis_values(4),
        np.array(
            [
                [-1.0, 1.0, -1.0],
                [-1.0 / 3.0, -7.0 / 9.0, 23.0 / 27.0],
                [1.0 / 3.0, -7.0 / 9.0, -23.0 / 27.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    )


def test_lattice_structure_maps_multicomponent_species_to_distinct_states():
    lattice = Lattice.cubic(4.0)
    template = Structure(
        lattice,
        ["Al", "Al"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    model = LatticeStructure(
        template_structure=template,
        site_mapping={"Al": ["Al", "X", "Mg", "Si"]},
    )
    structure = Structure(
        lattice,
        ["Al", "Mg"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )

    occ = model.get_occ_from_structure(structure)

    assert model.basis.uses_state_indices
    assert occ.array_equal([0, 2])
    assert model.occupation_value_for_species(0, "Si") == 3
    assert model.species_for_occupation_value(1, 2).symbol == "Mg"


def test_local_environment_enumeration_allows_more_than_two_species():
    lattice = Lattice.cubic(10.0)
    structure = Structure(
        lattice,
        ["Na", "Si"],
        [[0, 0, 0], [1, 0, 0]],
        coords_are_cartesian=True,
    )
    model = LatticeStructure(
        template_structure=structure,
        site_mapping={"Na": ["Na", "X"], "Si": ["Si", "P", "Ge"]},
    )

    results = enumerate_local_environments(
        model,
        center=0,
        cutoff=2.0,
        variable_species=["Si", "P", "Ge"],
        variable_site_indices=[1],
    )

    assert [result.label for result in results] == ["1:Si", "1:P", "1:Ge"]
    assert [result.full_occupation[1] for result in results] == [0, 1, 2]


def test_lce_expands_multicomponent_chebyshev_decorated_features():
    lattice = Lattice.cubic(10.0)
    template = Structure(
        lattice,
        ["Al", "Al"],
        [[0, 0, 0], [1, 0, 0]],
        coords_are_cartesian=True,
    )
    local_lattice = LocalLatticeStructure(
        template_structure=template,
        site_mapping={"Al": ["Al", "X", "Mg"]},
        center=0,
        cutoff=2.0,
        local_site_order={
            "name": "cartesian_x",
            "sort_keys": ["cartesian_x"],
            "exclude_center_site": False,
        },
    )
    model = LocalClusterExpansion()
    model.build(local_lattice, cutoff_cluster=[3.0, 0.0, 0.0])

    structure = Structure(
        lattice,
        ["Al", "Mg"],
        [[0, 0, 0], [1, 0, 0]],
        coords_are_cartesian=True,
    )
    _, corr = model.get_occ_corr_from_structure(structure)

    assert model.basis.uses_state_indices
    assert len(model.cluster_site_indices) == 6
    assert len(model.get_orbit_fingerprints()) == 6
    np.testing.assert_allclose(corr, np.array([0.0, 2.0, -1.0, -1.0, 1.0, 1.0]))

    payload = model.as_dict()
    reloaded = LocalClusterExpansion.from_dict(payload)
    reloaded_corr = np.empty(len(reloaded.cluster_site_indices))
    reloaded._calculate_correlation(
        reloaded_corr,
        Occupation([0, 2], basis=reloaded.basis, validate=False).data,
    )
    np.testing.assert_allclose(reloaded_corr, corr)
