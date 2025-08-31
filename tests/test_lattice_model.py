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
    """Test get_occ_from_structure with a structure that perfectly matches the template."""
    model = simple_lattice_model
    structure = model.template_structure.copy()
    occ = model.get_occ_from_structure(structure)
    # With chebyshev basis, occupied sites have value 1
    expected_values = np.array([1] * 2)  # All sites occupied
    assert np.array_equal(occ.values, expected_values)
    assert occ.basis == 'chebyshev'

def test_get_occ_from_structure_with_vacancy(simple_lattice_model):
    """Test get_occ_from_structure with a vacancy."""
    model = simple_lattice_model
    structure = model.template_structure.copy()
    structure.remove_sites([1])  # Remove one Na to create a vacancy
    occ = model.get_occ_from_structure(structure)
    # With chebyshev basis: occupied=1, vacant=-1
    expected_values = np.array([1, -1])  # First site occupied, second vacant
    assert np.array_equal(occ.values, expected_values)
    assert occ.basis == 'chebyshev'

def test_get_occ_from_structure_supercell(simple_lattice_model):
    """Test get_occ_from_structure with a supercell."""
    model = simple_lattice_model
    supercell_structure = model.template_structure.copy()
    supercell_structure.make_supercell([2, 1, 1])
    occ = model.get_occ_from_structure(supercell_structure)
    # All sites occupied in supercell (4 sites total)
    expected_values = np.array([1] * 4)
    assert np.array_equal(occ.values, expected_values)
    assert occ.basis == 'chebyshev'

def test_get_occ_from_structure_supercell_with_vacancies(simple_lattice_model):
    """Test get_occ_from_structure with a supercell that has vacancies."""
    model = simple_lattice_model
    supercell_structure = model.template_structure.copy()
    supercell_structure.make_supercell([2, 1, 1])
    supercell_structure.remove_sites([1, 3])  # Remove two sites to create vacancies
    occ = model.get_occ_from_structure(supercell_structure)
    # Pattern: occupied, vacant, occupied, vacant
    expected_values = np.array([1, -1, 1, -1])
    assert np.array_equal(occ.values, expected_values)
    assert occ.basis == 'chebyshev'
