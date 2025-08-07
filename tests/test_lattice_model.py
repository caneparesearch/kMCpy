import pytest
import numpy as np
from pymatgen.core import Structure, Lattice
from kmcpy.models.lattice_model import LatticeModel

@pytest.fixture
def simple_lattice_model():
    """A fixture to create a simple LatticeModel for testing."""
    lattice = Lattice.cubic(3.0)
    template_structure = Structure(
        lattice,
        ["Na", "Na"],
        [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    specie_site_mapping = {"Na": ["Na", "X"]}  # X represents a vacancy
    return LatticeModel(template_structure=template_structure, specie_site_mapping=specie_site_mapping)

def test_get_occ_from_structure_perfect_match(simple_lattice_model):
    """Test get_occ_from_structure with a structure that perfectly matches the template."""
    model = simple_lattice_model
    structure = model.template_structure.copy()
    occ = model.get_occ_from_structure(structure)
    expected_occ = np.array([model.basis.basis_function[0]] * 2)
    assert np.array_equal(occ, expected_occ)

def test_get_occ_from_structure_with_vacancy(simple_lattice_model):
    """Test get_occ_from_structure with a vacancy."""
    model = simple_lattice_model
    structure = model.template_structure.copy()
    structure.remove_sites([1])  # Remove one Na to create a vacancy
    occ = model.get_occ_from_structure(structure)
    expected_occ = np.array([model.basis.basis_function[0], model.basis.basis_function[1]])
    assert np.array_equal(occ, expected_occ)

def test_get_occ_from_structure_supercell(simple_lattice_model):
    """Test get_occ_from_structure with a supercell."""
    model = simple_lattice_model
    supercell_structure = model.template_structure.copy()
    supercell_structure.make_supercell([2, 1, 1])
    occ = model.get_occ_from_structure(supercell_structure)
    expected_occ = np.array([model.basis.basis_function[0]] * 4)
    assert np.array_equal(occ, expected_occ)

def test_get_occ_from_structure_supercell_with_vacancies(simple_lattice_model):
    """Test get_occ_from_structure with a supercell that has vacancies."""
    model = simple_lattice_model
    supercell_structure = model.template_structure.copy()
    supercell_structure.make_supercell([2, 1, 1])
    supercell_structure.remove_sites([1, 3])  # Remove two sites to create vacancies
    occ = model.get_occ_from_structure(supercell_structure)
    expected_occ = np.array([
        model.basis.basis_function[0], 
        model.basis.basis_function[1],
        model.basis.basis_function[0],
        model.basis.basis_function[1]
    ])
    assert np.array_equal(occ, expected_occ)
