"""LatticeStructure as the disordered structure a simulation runs on."""

import pytest
from pymatgen.core import Lattice, Structure

import kmcpy
from kmcpy.structure.lattice_structure import LatticeStructure


@pytest.fixture
def disordered():
    """Li half-occupied, Si/P mixed, O fixed."""
    return Structure(
        Lattice.cubic(4.0),
        [{"Li": 0.5}, {"Si": 0.5, "P": 0.5}, "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0]],
    )


@pytest.mark.unit
def test_partial_occupancies_define_allowed_species(disordered):
    lattice = LatticeStructure(disordered)

    assert lattice.site_mapping == {"Li": ["Li", "X"], "Si": ["Si", "P"], "O": ["O"]}
    assert lattice.template_structure.is_ordered
    assert lattice.n_active_sites == 2
    assert lattice.mobile_species == ["Li"]


@pytest.mark.unit
def test_site_mapping_lists_only_varying_species(disordered):
    ordered = Structure(disordered.lattice, ["Li", "Si", "O"], disordered.frac_coords)
    lattice = LatticeStructure(ordered, {"Li": ["Li", "X"]})

    assert lattice.site_mapping == {"Li": ["Li", "X"], "Si": ["Si"], "O": ["O"]}
    assert lattice.n_active_sites == 1


@pytest.mark.unit
def test_make_supercell_works_like_pymatgen(disordered):
    lattice = LatticeStructure(disordered)
    supercell = lattice.make_supercell((2, 1, 1), in_place=False)

    assert lattice.supercell_shape == (1, 1, 1)
    assert supercell.supercell_shape == (2, 1, 1)
    assert supercell.n_active_sites == 2 * lattice.n_active_sites
    assert len(supercell.template_structure) == len(lattice.template_structure)

    # In place by default; shapes multiply; ints and diagonal matrices work.
    assert lattice.make_supercell([[1, 0, 0], [0, 2, 0], [0, 0, 1]]) is lattice
    assert lattice.supercell_shape == (1, 2, 1)
    assert lattice.active_site_order.supercell_shape == (1, 2, 1)
    assert lattice.make_supercell(2).supercell_shape == (2, 4, 2)

    with pytest.raises(ValueError, match="diagonal"):
        lattice.make_supercell([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="three positive integers"):
        lattice.make_supercell((0, 1, 1))


@pytest.mark.unit
def test_hop_cutoff_longer_than_supercell_is_rejected(disordered):
    # The 8 Angstrom supercell axis folds the +4 and -4 Angstrom Li neighbors
    # onto one site, and a 4 Angstrom axis folds a site onto itself.
    lattice = LatticeStructure(disordered).make_supercell((2, 1, 1))
    with pytest.raises(ValueError, match="supercell_shape .* too small"):
        kmcpy.HopEvents(cutoff=4.1).generate(lattice)


@pytest.mark.unit
def test_occupations_and_ordered_structures_round_trip(disordered):
    supercell = LatticeStructure(disordered).make_supercell((2, 1, 1))
    occupations = [0, 1, 1, 0]  # Li, vacancy, P, Si

    structure = supercell.structure_from_occupations(occupations)

    assert structure.is_ordered
    assert structure.composition.as_dict() == {"Li": 1, "P": 1, "Si": 1, "O": 2}
    assert supercell.occupations_from_structure(structure) == occupations


@pytest.mark.unit
def test_serialization_keeps_supercell_shape(disordered, tmp_path):
    supercell = LatticeStructure(disordered).make_supercell((2, 1, 1))

    restored = LatticeStructure.from_dict(supercell.as_dict())
    assert restored.supercell_shape == (2, 1, 1)
    assert restored.site_mapping == supercell.site_mapping

    supercell.to(tmp_path / "lattice.json")
    from_file = LatticeStructure.from_file(tmp_path / "lattice.json")
    assert from_file.active_site_order.fingerprint == supercell.active_site_order.fingerprint


@pytest.mark.integration
def test_ordered_structure_is_an_initial_state(disordered, tmp_path):
    # 3x3x3 keeps every neighbor within the hop cutoff a distinct site.
    supercell = LatticeStructure(disordered).make_supercell(3)
    occupations = [index % 2 for index in range(supercell.n_active_sites)]
    settings = dict(
        events=kmcpy.HopEvents(cutoff=4.1),
        mobile_ion_charge=1.0,
        model=kmcpy.LocalBarrierModel.constant_barrier(300.0),
        temperature=300.0,
        kmc_passes=1,
        equilibration_passes=0,
        random_seed=1,
        output_dir=tmp_path,
    )

    from_structure = kmcpy.Simulation(
        supercell, state=supercell.structure_from_occupations(occupations), **settings
    ).build()
    from_list = kmcpy.Simulation(supercell, state=occupations, **settings).build()

    assert list(from_structure.simulation_state.occupations) == occupations
    assert list(from_list.simulation_state.occupations) == occupations
