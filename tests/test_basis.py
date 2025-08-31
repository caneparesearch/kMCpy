"""
Unit tests for basis functions and Occupation class in kmcpy.structure.basis.

This test suite covers:
- OccupationBasis and ChebyshevBasis functionality
- Occupation class with both basis types
- Basis conversion operations
- Edge cases and error conditions
"""

import pytest
import numpy as np
from kmcpy.structure.basis import OccupationBasis, ChebyshevBasis, Occupation


class TestOccupationBasis:
    """Test the OccupationBasis class."""
    
    def test_init(self):
        """Test OccupationBasis initialization."""
        basis = OccupationBasis()
        assert basis.vacant_value == 0
        assert basis.occupied_value == 1
        assert basis.valid_values == {0, 1}
        assert basis.basis_function == [0, 1]
    
    def test_is_occupied(self):
        """Test occupation detection."""
        basis = OccupationBasis()
        assert basis.is_occupied(1) is True
        assert basis.is_occupied(0) is False
    
    def test_is_vacant(self):
        """Test vacancy detection."""
        basis = OccupationBasis()
        assert basis.is_vacant(0) is True
        assert basis.is_vacant(1) is False
    
    def test_flip_value(self):
        """Test value flipping."""
        basis = OccupationBasis()
        assert basis.flip_value(0) == 1
        assert basis.flip_value(1) == 0
    
    def test_basis_conversions(self):
        """Test conversion to/from Chebyshev basis."""
        basis = OccupationBasis()
        # To Chebyshev
        assert basis.to_chebyshev(0) == -1  # vacant -> -1
        assert basis.to_chebyshev(1) == 1   # occupied -> +1
        # From Chebyshev
        assert basis.from_chebyshev(-1) == 0  # -1 -> vacant
        assert basis.from_chebyshev(1) == 1   # +1 -> occupied


class TestChebyshevBasis:
    """Test the ChebyshevBasis class."""
    
    def test_init(self):
        """Test ChebyshevBasis initialization."""
        basis = ChebyshevBasis()
        assert basis.vacant_value == -1
        assert basis.occupied_value == 1
        assert basis.valid_values == {-1, 1}
        assert basis.basis_function == [-1, 1]
    
    def test_is_occupied(self):
        """Test occupation detection."""
        basis = ChebyshevBasis()
        assert basis.is_occupied(1) is True
        assert basis.is_occupied(-1) is False
    
    def test_is_vacant(self):
        """Test vacancy detection."""
        basis = ChebyshevBasis()
        assert basis.is_vacant(-1) is True
        assert basis.is_vacant(1) is False
    
    def test_flip_value(self):
        """Test value flipping."""
        basis = ChebyshevBasis()
        assert basis.flip_value(-1) == 1
        assert basis.flip_value(1) == -1
    
    def test_basis_conversions(self):
        """Test conversion to/from occupation basis."""
        basis = ChebyshevBasis()
        # To occupation
        assert basis.to_occupation(-1) == 0  # vacant -> 0
        assert basis.to_occupation(1) == 1   # occupied -> 1
        # From occupation
        assert basis.from_occupation(0) == -1  # 0 -> vacant
        assert basis.from_occupation(1) == 1   # 1 -> occupied


class TestOccupation:
    """Test the Occupation class."""
    
    def test_init_chebyshev(self):
        """Test initialization with Chebyshev basis."""
        occ = Occupation([-1, 1, -1, 1], basis='chebyshev')
        assert occ.basis == 'chebyshev'
        assert occ.values == [-1, 1, -1, 1]
        assert len(occ) == 4
    
    def test_init_occupation(self):
        """Test initialization with occupation basis."""
        occ = Occupation([0, 1, 0, 1], basis='occupation')
        assert occ.basis == 'occupation'
        assert occ.values == [0, 1, 0, 1]
        assert len(occ) == 4
    
    def test_init_validation_chebyshev(self):
        """Test validation with Chebyshev basis."""
        # Valid values
        Occupation([-1, 1], basis='chebyshev')  # Should not raise
        
        # Invalid values
        with pytest.raises(ValueError, match="Invalid values"):
            Occupation([0, 1], basis='chebyshev')  # 0 not valid in Chebyshev
    
    def test_init_validation_occupation(self):
        """Test validation with occupation basis."""
        # Valid values
        Occupation([0, 1], basis='occupation')  # Should not raise
        
        # Invalid values
        with pytest.raises(ValueError, match="Invalid values"):
            Occupation([-1, 1], basis='occupation')  # -1 not valid in occupation
    
    def test_init_invalid_basis(self):
        """Test invalid basis raises error."""
        with pytest.raises(ValueError, match="Unknown basis"):
            Occupation([0, 1], basis='invalid')
    
    def test_properties(self):
        """Test basic properties."""
        occ = Occupation([-1, 1, -1], basis='chebyshev')
        assert occ.basis == 'chebyshev'
        assert np.array_equal(occ.data, np.array([-1, 1, -1]))
        assert occ.values == [-1, 1, -1]
        assert len(occ) == 3
    
    def test_indexing_and_slicing(self):
        """Test indexing and slicing operations."""
        occ = Occupation([-1, 1, -1, 1], basis='chebyshev')
        
        # Single index
        assert occ[0] == -1
        assert occ[1] == 1
        
        # Slicing returns new Occupation
        sub_occ = occ[1:3]
        assert isinstance(sub_occ, Occupation)
        assert sub_occ.values == [1, -1]
        assert sub_occ.basis == 'chebyshev'
    
    def test_setitem(self):
        """Test setting values."""
        occ = Occupation([-1, -1], basis='chebyshev')
        occ[0] = 1
        assert occ[0] == 1
        
        # Setting invalid value should raise error
        with pytest.raises(ValueError):
            occ[1] = 0  # Invalid for Chebyshev basis
    
    def test_iteration(self):
        """Test iteration over occupation values."""
        occ = Occupation([-1, 1, -1], basis='chebyshev')
        values = list(occ)
        assert values == [-1, 1, -1]
    
    def test_equality(self):
        """Test equality comparison."""
        occ1 = Occupation([-1, 1], basis='chebyshev')
        occ2 = Occupation([-1, 1], basis='chebyshev')
        occ3 = Occupation([0, 1], basis='occupation')
        occ4 = Occupation([1, -1], basis='chebyshev')
        
        assert occ1 == occ2  # Same values and basis
        assert occ1 != occ3  # Different basis
        assert occ1 != occ4  # Different values
        assert occ1 != "not an occupation"  # Different type
    
    def test_string_representations(self):
        """Test string representations."""
        occ = Occupation([-1, 1], basis='chebyshev')
        assert "Occupation([-1, 1], basis='chebyshev')" in repr(occ)
        assert "Chebyshev occupation: [-1, 1]" in str(occ)
    
    def test_count_operations_chebyshev(self):
        """Test counting operations with Chebyshev basis."""
        occ = Occupation([-1, 1, -1, 1, 1], basis='chebyshev')
        assert occ.count_occupied() == 3  # Three +1 values
        assert occ.count_vacant() == 2    # Two -1 values
    
    def test_count_operations_occupation(self):
        """Test counting operations with occupation basis."""
        occ = Occupation([0, 1, 0, 1, 1], basis='occupation')
        assert occ.count_occupied() == 3  # Three 1 values
        assert occ.count_vacant() == 2    # Two 0 values
    
    def test_get_indices_chebyshev(self):
        """Test getting indices with Chebyshev basis."""
        occ = Occupation([-1, 1, -1, 1], basis='chebyshev')
        assert occ.get_occupied_indices() == [1, 3]  # +1 at indices 1, 3
        assert occ.get_vacant_indices() == [0, 2]    # -1 at indices 0, 2
    
    def test_get_indices_occupation(self):
        """Test getting indices with occupation basis."""
        occ = Occupation([0, 1, 0, 1], basis='occupation')
        assert occ.get_occupied_indices() == [1, 3]  # 1 at indices 1, 3
        assert occ.get_vacant_indices() == [0, 2]    # 0 at indices 0, 2
    
    def test_flip_operations(self):
        """Test flipping operations."""
        occ = Occupation([-1, 1, -1], basis='chebyshev')
        
        # Flip single index - returns new object
        flipped = occ.flip(0)
        assert flipped.values == [1, 1, -1]
        assert occ.values == [-1, 1, -1]  # Original unchanged
        
        # Flip multiple indices
        flipped = occ.flip([0, 2])
        assert flipped.values == [1, 1, 1]
        
        # Flip in-place
        occ.flip_inplace(1)
        assert occ.values == [-1, -1, -1]
    
    def test_basis_conversion(self):
        """Test basis conversion."""
        # Chebyshev to occupation
        cheb_occ = Occupation([-1, 1, -1, 1], basis='chebyshev')
        occ_occ = cheb_occ.to_basis('occupation')
        assert occ_occ.basis == 'occupation'
        assert occ_occ.values == [0, 1, 0, 1]
        
        # Occupation to Chebyshev  
        occ_occ = Occupation([0, 1, 0, 1], basis='occupation')
        cheb_occ = occ_occ.to_basis('chebyshev')
        assert cheb_occ.basis == 'chebyshev'
        assert cheb_occ.values == [-1, 1, -1, 1]
        
        # Same basis - should return equivalent object
        same = cheb_occ.to_basis('chebyshev')
        assert same == cheb_occ
        
        # Invalid target basis
        with pytest.raises(ValueError, match="Unknown basis"):
            cheb_occ.to_basis('invalid')
    
    def test_copy(self):
        """Test copying."""
        occ = Occupation([-1, 1], basis='chebyshev')
        copied = occ.copy()
        
        assert copied == occ
        assert copied is not occ  # Different objects
        assert copied._data is not occ._data  # Different arrays
        
        # Modifying copy doesn't affect original
        copied.flip_inplace(0)
        assert copied != occ
    
    def test_factory_methods_chebyshev(self):
        """Test factory methods with Chebyshev basis."""
        # zeros (all vacant = -1)
        occ = Occupation.zeros(3, basis='chebyshev')
        assert occ.values == [-1, -1, -1]
        assert occ.count_vacant() == 3
        
        # ones (all occupied = +1)
        occ = Occupation.ones(3, basis='chebyshev')
        assert occ.values == [1, 1, 1]
        assert occ.count_occupied() == 3
    
    def test_factory_methods_occupation(self):
        """Test factory methods with occupation basis."""
        # zeros (all vacant = 0)
        occ = Occupation.zeros(3, basis='occupation')
        assert occ.values == [0, 0, 0]
        assert occ.count_vacant() == 3
        
        # ones (all occupied = 1)
        occ = Occupation.ones(3, basis='occupation')
        assert occ.values == [1, 1, 1]
        assert occ.count_occupied() == 3
    
    def test_random_factory(self):
        """Test random factory method."""
        # Test with fixed seed for reproducibility
        occ1 = Occupation.random(10, basis='chebyshev', fill_fraction=0.5, seed=42)
        occ2 = Occupation.random(10, basis='chebyshev', fill_fraction=0.5, seed=42)
        assert occ1 == occ2  # Same seed should give same result
        
        # Test different fill fractions
        empty = Occupation.random(10, basis='chebyshev', fill_fraction=0.0, seed=42)
        assert empty.count_occupied() == 0
        
        full = Occupation.random(10, basis='chebyshev', fill_fraction=1.0, seed=42)
        assert full.count_occupied() == 10
        
        # Test with occupation basis
        occ_basis = Occupation.random(5, basis='occupation', fill_fraction=0.5, seed=42)
        assert occ_basis.basis == 'occupation'
        assert all(v in [0, 1] for v in occ_basis.values)
    
    def test_numpy_array_input(self):
        """Test initialization with numpy arrays."""
        arr = np.array([-1, 1, -1])
        occ = Occupation(arr, basis='chebyshev')
        assert occ.values == [-1, 1, -1]
    
    def test_tuple_input(self):
        """Test initialization with tuples."""
        occ = Occupation((-1, 1, -1), basis='chebyshev')
        assert occ.values == [-1, 1, -1]
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Empty occupation
        empty = Occupation([], basis='chebyshev')
        assert len(empty) == 0
        assert empty.count_occupied() == 0
        assert empty.count_vacant() == 0
        assert empty.get_occupied_indices() == []
        assert empty.get_vacant_indices() == []
        
        # Single site
        single = Occupation([1], basis='chebyshev')
        assert len(single) == 1
        assert single.count_occupied() == 1
        
        # Large array
        large = Occupation.zeros(1000, basis='chebyshev')
        assert len(large) == 1000
        assert large.count_vacant() == 1000
    
    def test_validation_disabled(self):
        """Test that validation can be disabled."""
        # This should not raise an error even with invalid values
        occ = Occupation([0, 2, -5], basis='chebyshev', validate=False)
        assert occ.values == [0, 2, -5]
    
    def test_from_json_factory(self):
        """Test from_json factory method for loading occupation data from JSON files."""
        import tempfile
        import json
        import os
        
        # Test data with occupation basis (0/1 values)
        # 12 elements = 2 site types * 6 supercell volume
        test_data = {
            'occupation': [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
        }
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            # Test loading with Chebyshev basis (should convert 0->-1, 1->1)
            occ_cheb = Occupation.from_json(
                initial_state_file=temp_file,
                supercell_shape=[2, 3, 1],  # 2x3x1 = 6 supercell volume
                select_sites=[0],  # Select only first site type (6 sites)
                basis='chebyshev'
            )
            
            assert occ_cheb.basis == 'chebyshev'
            # Should have 6 values (first site type across supercell)
            assert len(occ_cheb.values) == 6
            # Values should be converted to Chebyshev basis
            expected_cheb = [-1, 1, -1, 1, 1, -1]  # First site type values converted
            assert occ_cheb.values == expected_cheb
            
            # Test loading with occupation basis (no conversion)
            occ_occ = Occupation.from_json(
                initial_state_file=temp_file,
                supercell_shape=[2, 3, 1],
                select_sites=[1],  # Select only second site type (6 sites)
                basis='occupation'
            )
            
            assert occ_occ.basis == 'occupation'
            assert len(occ_occ.values) == 6
            expected_occ = [1, 0, 0, 1, 1, 0]  # Second site type values unchanged
            assert occ_occ.values == expected_occ
            
            # Test selecting both site types (should get all 12 values)
            occ_both = Occupation.from_json(
                initial_state_file=temp_file,
                supercell_shape=[2, 3, 1],
                select_sites=[0, 1],  # Select both site types
                basis='chebyshev'
            )
            
            assert len(occ_both.values) == 12
            expected_both = [-1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1]
            assert occ_both.values == expected_both
            
        finally:
            os.unlink(temp_file)
    
    def test_from_json_errors(self):
        """Test error handling in from_json method."""
        import tempfile
        import json
        import os
        
        # Test file not found
        with pytest.raises(FileNotFoundError):
            Occupation.from_json(
                initial_state_file='nonexistent.json',
                supercell_shape=[1, 1, 1],
                select_sites=[0]
            )
        
        # Test missing 'occupation' key
        test_data_bad = {'wrong_key': [0, 1]}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data_bad, f)
            temp_file = f.name
        
        try:
            with pytest.raises(KeyError, match="'occupation' key not found"):
                Occupation.from_json(
                    initial_state_file=temp_file,
                    supercell_shape=[1, 1, 1],
                    select_sites=[0]
                )
        finally:
            os.unlink(temp_file)
        
        # Test incompatible supercell shape
        test_data = {'occupation': [0, 1, 0]}  # 3 elements
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="incompatible with supercell shape"):
                Occupation.from_json(
                    initial_state_file=temp_file,
                    supercell_shape=[2, 2, 1],  # Volume = 4, doesn't divide 3
                    select_sites=[0]
                )
        finally:
            os.unlink(temp_file)
    
    def test_kmc_methods(self):
        """Test KMC-specific methods in Occupation class."""
        occ = Occupation([-1, 1, -1, -1], basis='chebyshev')
        
        # Test swap_sites (immutable)
        swapped = occ.swap_sites(0, 1)
        assert swapped.values == [1, -1, -1, -1]
        assert occ.values == [-1, 1, -1, -1]  # Original unchanged
        
        # Test swap_sites_inplace (mutable)
        occ_mut = occ.copy()
        occ_mut.swap_sites_inplace(0, 1)
        assert occ_mut.values == [1, -1, -1, -1]
        
        # Test apply_event_inplace (simulating ion hopping)
        occ_event = occ.copy()
        occ_event.apply_event_inplace(1, 2)  # Move from occupied site 1 to vacant site 2
        assert occ_event.values == [-1, -1, 1, -1]  # Site 1 becomes vacant, site 2 occupied
        
        # Test with occupation basis
        occ_occ = Occupation([0, 1, 0, 0], basis='occupation')
        occ_occ.apply_event_inplace(1, 2)  # Move from occupied site 1 to vacant site 2
        assert occ_occ.values == [0, 0, 1, 0]  # Site 1 becomes vacant, site 2 occupied


class TestBasisIntegration:
    """Test integration between basis classes and Occupation."""
    
    def test_roundtrip_conversion(self):
        """Test roundtrip conversion between bases."""
        original_cheb = Occupation([-1, 1, -1, 1], basis='chebyshev')
        
        # Convert to occupation and back
        occ_version = original_cheb.to_basis('occupation')
        back_to_cheb = occ_version.to_basis('chebyshev')
        
        assert back_to_cheb == original_cheb
        
        # Test the other direction
        original_occ = Occupation([0, 1, 0, 1], basis='occupation')
        cheb_version = original_occ.to_basis('chebyshev')
        back_to_occ = cheb_version.to_basis('occupation')
        
        assert back_to_occ == original_occ
    
    def test_consistent_operations_across_bases(self):
        """Test that operations give consistent results across bases."""
        # Create equivalent occupations in both bases
        cheb_occ = Occupation([-1, 1, -1, 1, 1], basis='chebyshev')
        occ_occ = cheb_occ.to_basis('occupation')
        
        # Counts should be the same
        assert cheb_occ.count_occupied() == occ_occ.count_occupied()
        assert cheb_occ.count_vacant() == occ_occ.count_vacant()
        
        # Indices should be the same
        assert cheb_occ.get_occupied_indices() == occ_occ.get_occupied_indices()
        assert cheb_occ.get_vacant_indices() == occ_occ.get_vacant_indices()
        
        # Flipping should give equivalent results
        cheb_flipped = cheb_occ.flip(0)
        occ_flipped = occ_occ.flip(0)
        assert cheb_flipped.to_basis('occupation') == occ_flipped


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
