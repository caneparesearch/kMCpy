import pytest
import numpy as np
from pymatgen.core import Structure, Lattice
from kmcpy.structure.lattice_structure import LatticeStructure
from kmcpy.structure.local_env import LocalLatticeStructure

@pytest.fixture
def global_lattice_model_and_env():
    """A fixture to create a global LatticeStructure and a LocalLatticeStructure for testing."""
    # Use a large lattice to avoid periodic boundary issues in the test
    lattice = Lattice.cubic(10.0)
    template_structure = Structure(
        lattice,
        ["Na", "Cl", "Na", "Br"],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        coords_are_cartesian=True
    )
    specie_site_mapping = {"Na": ["Na", "X"], "Cl": ["Cl"], "Br": ["Br"]}  # X represents a vacancy
    
    # Create the global model
    global_model = LatticeStructure(
        template_structure=template_structure,
        specie_site_mapping=specie_site_mapping
    )
    
    # Create a local environment centered on the first Na atom
    local_env = LocalLatticeStructure(
        template_structure=template_structure,
        specie_site_mapping=specie_site_mapping,
        center=[0, 0, 0],
        cutoff=1.5
    )
    
    return global_model, local_env

def test_local_environment_setup(global_lattice_model_and_env):
    """Test that the LocalLatticeStructure is set up correctly."""
    _, local_env = global_lattice_model_and_env
    
    # The local environment should contain the Na at [0,0,0] and the Cl at [1,0,0]
    assert len(local_env.structure) == 2
    # The species should be Cl and Na, sorted by species name
    assert local_env.structure[0].species_string == "Cl"
    assert local_env.structure[1].species_string == "Na"
    # Check that the global indices are correct (1 for Cl, 0 for Na)
    assert local_env.site_indices == [1, 0]

def test_get_local_occupation(global_lattice_model_and_env):
    """Test getting occupation for a local environment from a full structure."""
    global_model, local_env = global_lattice_model_and_env
    
    # This structure is identical to the template
    structure_to_test = global_model.template_structure.copy()
    
    # 1. Get the full occupation vector from the global model
    full_occ = global_model.get_occ_from_structure(structure_to_test)
    
    # 2. Get the local occupation by slicing the full vector
    local_occ = full_occ[local_env.site_indices]
    
    # The local environment is perfect, so occupation should be all zeros (or the first basis function)
    expected_local_occ = np.array([
        global_model.basis.basis_function[0], 
        global_model.basis.basis_function[0]
    ])
    
    assert np.array_equal(local_occ, expected_local_occ)

def test_get_local_occupation_with_vacancy(global_lattice_model_and_env):
    """Test local occupation when there's a vacancy in the local environment."""
    global_model, local_env = global_lattice_model_and_env
    
    # Create a structure where the Cl atom is missing (a vacancy)
    structure_with_vacancy = global_model.template_structure.copy()
    structure_with_vacancy.remove_sites([1])  # Remove the Cl atom at index 1
    
    # 1. Get the full occupation vector
    full_occ = global_model.get_occ_from_structure(structure_with_vacancy)
    
    # 2. Get the local occupation by slicing
    local_occ = full_occ[local_env.site_indices]
    
    # The Cl site (index 1 in global, index 0 in local) is vacant.
    # The Na site (index 0 in global, index 1 in local) is present.
    expected_local_occ = np.array([
        global_model.basis.basis_function[1],  # Vacancy
        global_model.basis.basis_function[0]   # Present
    ])
    
    assert np.array_equal(local_occ, expected_local_occ)
