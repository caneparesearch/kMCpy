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

@pytest.fixture
def complex_lattice_model():
    """A fixture to create a more complex LatticeStructure for testing."""
    lattice = Lattice.cubic(5.0)
    template_structure = Structure(
        lattice,
        ["Na", "O", "Na", "Cl", "Cl"], 
        [[0, 0, 0], [0.25, 0, 0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )
    specie_site_mapping = {
        "Na": ["Na", "X"], 
        "O": ["O", "X"], 
        "Cl": ["Cl", "X"]
    }  # X represents a vacancy
    return LatticeStructure(template_structure=template_structure, specie_site_mapping=specie_site_mapping)

def test_get_occ_from_structure_perfect_match(simple_lattice_model):
    """Test get_occ_from_structure with perfect match (no vacancies)."""
    model = simple_lattice_model
    structure = model.template_structure.copy()
    occ = model.get_occ_from_structure(structure)
    expected_occ = [model.basis.occupied_value] * 2
    assert occ.array_equal(expected_occ), f"Expected {expected_occ}, got {occ.values}"

def test_get_occ_from_structure_with_vacancy(simple_lattice_model):
    """Test get_occ_from_structure with a vacancy."""
    model = simple_lattice_model
    structure = model.template_structure.copy()
    structure.remove_sites([0])  # Remove one Na to create a vacancy
    occ = model.get_occ_from_structure(structure)
    # Site 0 is now vacant (removed), site 1 is still occupied
    expected_occ = [model.basis.vacant_value, model.basis.occupied_value]
    assert occ.array_equal(expected_occ), f"Expected {expected_occ}, got {occ.values}"

def test_get_occ_from_structure_multiple_vacancies(simple_lattice_model):
    """Test get_occ_from_structure with multiple vacancies."""
    model = simple_lattice_model
    structure = model.template_structure.copy()
    structure.remove_sites([0, 1])  # Remove both Na to create vacancies
    occ = model.get_occ_from_structure(structure)
    # Both sites should be vacant
    expected_occ = [model.basis.vacant_value, model.basis.vacant_value]
    assert occ.array_equal(expected_occ), f"Expected {expected_occ}, got {occ.values}"

def test_get_occ_from_structure_supercell(simple_lattice_model):
    """Test get_occ_from_structure with a supercell."""
    model = simple_lattice_model
    supercell_structure = model.template_structure.copy()
    supercell_structure.make_supercell([2, 1, 1])
    occ = model.get_occ_from_structure(supercell_structure, sc_matrix=np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]]))
    # All sites should be occupied in the supercell
    expected_occ = [model.basis.occupied_value] * 4
    assert occ.array_equal(expected_occ), f"Expected {expected_occ}, got {occ.values}"

def test_get_occ_from_structure_supercell_with_vacancies(simple_lattice_model):
    """Test get_occ_from_structure with a supercell that has vacancies."""
    model = simple_lattice_model
    supercell_structure = model.template_structure.copy()
    supercell_structure.make_supercell([2, 1, 1])
    supercell_structure.remove_sites([1, 3])  # Remove two sites to create vacancies
    occ = model.get_occ_from_structure(supercell_structure, sc_matrix=np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]]))
    # Sites 0 and 2 remain occupied, sites 1 and 3 become vacant
    expected_occ = [
        model.basis.occupied_value,   # Site 0: occupied
        model.basis.vacant_value,     # Site 1: vacant (removed)
        model.basis.occupied_value,   # Site 2: occupied
        model.basis.vacant_value      # Site 3: vacant (removed)
    ]
    assert occ.array_equal(expected_occ), f"Expected {expected_occ}, got {occ.values}"

def test_get_occ_from_structure_complex_structure(complex_lattice_model):
    """Test get_occ_from_structure with a more complex structure."""
    model = complex_lattice_model
    structure = model.template_structure.copy()
    occ = model.get_occ_from_structure(structure)
    # All sites should be occupied since no vacancies
    expected_occ = [model.basis.occupied_value] * 5
    assert occ.array_equal(expected_occ), f"Expected {expected_occ}, got {occ.values}"

def test_get_occ_from_structure_complex_with_vacancies(complex_lattice_model):
    """Test get_occ_from_structure with a complex structure with selective vacancies."""
    model = complex_lattice_model
    structure = model.template_structure.copy()
    # Remove some specific sites to create vacancies
    structure.remove_sites([1, 4])  # Remove O at [0.25,0,0] and Cl at [0.5,0.5,0.5]
    occ = model.get_occ_from_structure(structure)
    
    expected_occ = [
        model.basis.occupied_value,   # Na at [0,0,0]: occupied
        model.basis.vacant_value,     # O at [0.25,0,0]: vacant (removed)
        model.basis.occupied_value,   # Na at [0.5,0.5,0]: occupied
        model.basis.occupied_value,   # Cl at [0.5,0,0]: occupied
        model.basis.vacant_value      # Cl at [0.5,0.5,0.5]: vacant (removed)
    ]
    assert occ.array_equal(expected_occ), f"Expected {expected_occ}, got {occ.values}"

def test_get_occ_from_structure_basis_consistency(simple_lattice_model):
    """Test that the occupation uses the correct basis values."""
    model = simple_lattice_model
    structure = model.template_structure.copy()
    structure.remove_sites([0])  # Remove one site
    occ = model.get_occ_from_structure(structure)
    
    # Check that the values match the basis
    assert all(val in model.basis.valid_values for val in occ.values), \
        f"Occupation values {occ.values} not all in valid basis values {model.basis.valid_values}"
    
    # Check specific values
    assert occ[0] == model.basis.vacant_value, f"Site 0 should be vacant ({model.basis.vacant_value}), got {occ[0]}"
    assert occ[1] == model.basis.occupied_value, f"Site 1 should be occupied ({model.basis.occupied_value}), got {occ[1]}"

def test_get_occ_from_structure_error_handling(simple_lattice_model):
    """Test error handling for incompatible structures."""
    model = simple_lattice_model
    
    # Create a completely different structure that shouldn't match
    different_lattice = Lattice.cubic(10.0)  # Very different lattice parameter
    incompatible_structure = Structure(
        different_lattice,
        ["K", "F"],  # Different species
        [[0, 0, 0], [0.5, 0.5, 0.5]]
    )

    with pytest.raises(ValueError, match="No mapping found"):
        model.get_occ_from_structure(incompatible_structure)

def test_get_occ_from_structure_automatic_supercell_detection(simple_lattice_model):
    """Test automatic supercell detection without explicit sc_matrix."""
    model = simple_lattice_model
    supercell_structure = model.template_structure.copy()
    supercell_structure.make_supercell([2, 2, 1])  # 2x2x1 supercell

    # Don't provide sc_matrix - let it be detected automatically
    occ = model.get_occ_from_structure(supercell_structure)

    # Should have 8 sites all occupied
    expected_occ = [model.basis.occupied_value] * 8
    assert occ.array_equal(expected_occ), f"Expected {expected_occ}, got {occ.values}"

def test_get_occ_from_structure_partial_occupancy_supercell(simple_lattice_model):
    """Test supercell with partial occupancy (some vacancies)."""
    model = simple_lattice_model
    supercell_structure = model.template_structure.copy()
    supercell_structure.make_supercell([2, 2, 1])  # 2x2x1 supercell = 8 sites
    
    # Remove every other site to create a checkerboard pattern
    supercell_structure.remove_sites([1, 3, 5, 7])
    
    occ = model.get_occ_from_structure(supercell_structure, 
                                     sc_matrix=np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]]))
    
    # Should have alternating occupied/vacant pattern
    expected_occ = [
        model.basis.occupied_value,   # Site 0: occupied
        model.basis.vacant_value,     # Site 1: vacant
        model.basis.occupied_value,   # Site 2: occupied  
        model.basis.vacant_value,     # Site 3: vacant
        model.basis.occupied_value,   # Site 4: occupied
        model.basis.vacant_value,     # Site 5: vacant
        model.basis.occupied_value,   # Site 6: occupied
        model.basis.vacant_value      # Site 7: vacant
    ]
    assert occ.array_equal(expected_occ), f"Expected {expected_occ}, got {occ.values}"

def test_get_occ_from_structure_tolerance_sensitivity(simple_lattice_model):
    """Test that tolerance settings work correctly."""
    model = simple_lattice_model
    structure = model.template_structure.copy()
    
    # Slightly perturb one atom position
    structure.translate_sites([0], [0.001, 0.001, 0.001])  # Small displacement
    
    # Should work with default tolerance
    occ = model.get_occ_from_structure(structure, tol=0.1)
    expected_occ = [model.basis.occupied_value] * 2
    assert occ.array_equal(expected_occ), f"Expected {expected_occ}, got {occ.values}"
    
    # Should fail with very tight tolerance
    with pytest.raises(ValueError, match="No mapping found"):
        model.get_occ_from_structure(structure, tol=0.0001)

def test_get_occ_from_structure_basis_type_consistency(simple_lattice_model):
    """Test that different basis types work consistently."""
    model = simple_lattice_model
    structure = model.template_structure.copy()
    structure.remove_sites([0])  # Create vacancy at site 0
    
    occ = model.get_occ_from_structure(structure)
    
    # Check that the occupation object has the correct basis
    assert occ.basis == model.basis.name, f"Expected basis {model.basis.name}, got {occ.basis}"
    
    # Check values are from the correct basis
    assert occ[0] == model.basis.vacant_value
    assert occ[1] == model.basis.occupied_value
    
    # Test conversion to different basis
    if model.basis.name == 'chebyshev':
        occ_converted = occ.to_basis('occupation')
        assert occ_converted[0] == 0  # vacant in occupation basis
        assert occ_converted[1] == 1  # occupied in occupation basis

def test_get_occ_from_structure_like_test_py_example(complex_lattice_model):
    """Test similar to the working example in test.py."""
    model = complex_lattice_model
    
    # Create an input structure similar to test.py
    lattice = Lattice.cubic(5.0)
    input_structure = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    supercell_shape = (2, 2, 1)  
    input_structure.make_supercell(scaling_matrix=supercell_shape)
    
    # Get occupation from this input structure 
    occ = model.get_occ_from_structure(input_structure, sc_matrix=np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]]))
    
    # Check that we get reasonable results
    assert len(occ) == len(model.template_structure) * 4  # 2x2x1 supercell = 4x template sites
    assert all(val in model.basis.valid_values for val in occ.values)
    
    # Count occupied vs vacant sites
    n_occupied = occ.count_occupied()
    n_vacant = occ.count_vacant()
    assert n_occupied + n_vacant == len(occ)
    assert n_occupied > 0  # Should have some occupied sites

def test_get_occ_from_structure_different_species_subset(complex_lattice_model):
    """Test with input structure that's a subset with different species composition."""
    model = complex_lattice_model
    
    # Create structure with only some of the species from template
    lattice = Lattice.cubic(5.0)
    partial_structure = Structure(
        lattice, 
        ["Na", "Cl", "Cl"], 
        [[0, 0, 0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )
    
    occ = model.get_occ_from_structure(partial_structure)
    
    # Should have 5 sites total (template size)
    assert len(occ) == 5
    
    # Should have 3 occupied sites and 2 vacant
    assert occ.count_occupied() == 3
    assert occ.count_vacant() == 2
    
    # Check specific pattern
    expected_pattern = [
        model.basis.occupied_value,   # Na at [0,0,0] 
        model.basis.vacant_value,     # O at [0.25,0,0] - not in input
        model.basis.vacant_value,     # Na at [0.5,0.5,0] - not in input
        model.basis.occupied_value,   # Cl at [0.5,0,0]
        model.basis.occupied_value    # Cl at [0.5,0.5,0.5]
    ]
    assert occ.array_equal(expected_pattern), f"Expected {expected_pattern}, got {occ.values}"
