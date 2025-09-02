import pytest
import numpy as np
from pymatgen.core import Structure, Lattice
from kmcpy.structure.lattice_structure import LatticeStructure

@pytest.fixture
def simple_lattice_model():
    """A fixture to create a simple LatticeStructure for testing."""
    lattice = Lattice.cubic(3.0)
    template_structure = Structure(
        lattice,
        ["Na", "Na"],
        [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    specie_site_mapping = {"Na": ["Na", "X"]}  # X represents a vacancy
    return LatticeStructure(template_structure=template_structure, specie_site_mapping=specie_site_mapping)

def test_get_occ_from_structure_perfect_match(simple_lattice_model):
    """Test get_occ_from_structure with perfect match (no vacancies)."""
    model = simple_lattice_model
    structure = model.template_structure.copy()
    occ = model.get_occ_from_structure(structure)
    expected_occ = [model.basis.occupied_value] * 2
    assert occ.array_equal(expected_occ)

def test_get_occ_from_structure_with_vacancy(simple_lattice_model):
    """Test get_occ_from_structure with a vacancy."""
    model = simple_lattice_model
    structure = model.template_structure.copy()
    structure.remove_sites([1])  # Remove one Na to create a vacancy
    occ = model.get_occ_from_structure(structure)
    # Site 0 is still occupied, site 1 is now vacant
    expected_occ = [model.basis.occupied_value, model.basis.vacant_value]
    assert occ.array_equal(expected_occ)

def test_get_occ_from_structure_supercell(simple_lattice_model):
    """Test get_occ_from_structure with a supercell."""
    model = simple_lattice_model
    supercell_structure = model.template_structure.copy()
    supercell_structure.make_supercell([2, 1, 1])
    occ = model.get_occ_from_structure(supercell_structure)
    # All sites should be occupied in the supercell
    expected_occ = [model.basis.occupied_value] * 4
    assert occ.array_equal(expected_occ)

def test_get_occ_from_structure_supercell_with_vacancies(simple_lattice_model):
    """Test get_occ_from_structure with a supercell that has vacancies."""
    model = simple_lattice_model
    supercell_structure = model.template_structure.copy()
    supercell_structure.make_supercell([2, 1, 1])
    supercell_structure.remove_sites([1, 3])  # Remove two sites to create vacancies
    occ = model.get_occ_from_structure(supercell_structure)
    # Sites 0 and 2 remain occupied, sites 1 and 3 become vacant
    expected_occ = [
        model.basis.occupied_value,   # Site 0: occupied
        model.basis.vacant_value,     # Site 1: vacant (removed)
        model.basis.occupied_value,   # Site 2: occupied
        model.basis.vacant_value      # Site 3: vacant (removed)
    ]
    assert occ.array_equal(expected_occ)
