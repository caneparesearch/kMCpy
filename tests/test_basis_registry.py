"""
Unit tests for basis functions registry system and generalized Occupation class.

Tests the registry system, built-in basis functions (OccupationBasis, ChebyshevBasis),
and the generalized Occupation class that works with any registered basis function.
"""

import pytest
import numpy as np
from typing import List, Union

from kmcpy.structure.basis import (
    BasisFunction,
    OccupationBasis,
    ChebyshevBasis,
    Occupation,
    register_basis,
    get_basis,
    BASIS_REGISTRY
)


class TestBasisFunctionRegistry:
    """Test the basis function registry system."""
    
    def test_registry_contains_built_in_functions(self):
        """Test that built-in basis functions are registered."""
        assert 'occupation' in BASIS_REGISTRY
        assert 'chebyshev' in BASIS_REGISTRY
    
    def test_get_basis_returns_correct_instances(self):
        """Test that get_basis returns correct instances."""
        occ_basis = get_basis('occupation')
        cheb_basis = get_basis('chebyshev')
        
        assert isinstance(occ_basis, OccupationBasis)
        assert isinstance(cheb_basis, ChebyshevBasis)
    
    def test_get_basis_raises_for_unknown(self):
        """Test that get_basis raises ValueError for unknown basis."""
        with pytest.raises(ValueError, match="Unknown basis 'unknown'"):
            get_basis('unknown')
    
    def test_custom_basis_registration(self):
        """Test registering a custom basis function."""
        
        @register_basis('custom_test')
        class CustomTestBasis(BasisFunction):
            @property
            def name(self):
                return 'custom_test'
            
            @property
            def vacant_value(self):
                return 0.0
            
            @property
            def occupied_value(self):
                return 2.0
            
            @property
            def valid_values(self):
                return {0.0, 2.0}
            
            @property
            def basis_function(self):
                return [0.0, 2.0]
            
            def is_occupied(self, value):
                return value == 2.0
            
            def is_vacant(self, value):
                return value == 0.0
            
            def flip_value(self, value):
                return 2.0 if value == 0.0 else 0.0
        
        # Test that custom basis is registered
        assert 'custom_test' in BASIS_REGISTRY
        
        # Test that we can get and use the custom basis
        custom_basis = get_basis('custom_test')
        assert isinstance(custom_basis, CustomTestBasis)
        assert custom_basis.vacant_value == 0.0
        assert custom_basis.occupied_value == 2.0


class TestBasisFunctionInterface:
    """Test the BasisFunction abstract interface."""
    
    def test_occupation_basis_implements_interface(self):
        """Test that OccupationBasis properly implements BasisFunction."""
        basis = OccupationBasis()
        
        # Test all required properties
        assert basis.vacant_value == 0
        assert basis.occupied_value == 1
        assert basis.valid_values == {0, 1}
        assert basis.basis_function == [0, 1]
        assert basis.name == 'occupation'
        
        # Test all required methods
        assert basis.is_occupied(1)
        assert basis.is_vacant(0)
        assert basis.flip_value(0) == 1
        
        # Test convert_to method
        cheb_basis = ChebyshevBasis()
        assert basis.convert_to(0, cheb_basis) == -1  # vacant to vacant
        assert basis.convert_to(1, cheb_basis) == 1   # occupied to occupied
    
    def test_chebyshev_basis_implements_interface(self):
        """Test that ChebyshevBasis properly implements BasisFunction."""
        basis = ChebyshevBasis()
        
        # Test all required properties
        assert basis.vacant_value == -1
        assert basis.occupied_value == 1
        assert basis.valid_values == {-1, 1}
        assert basis.basis_function == [-1, 1]
        assert basis.name == 'chebyshev'
        
        # Test all required methods
        assert basis.is_occupied(1)
        assert basis.is_vacant(-1)
        assert basis.flip_value(-1) == 1
        
        # Test convert_to method
        occ_basis = OccupationBasis()
        assert basis.convert_to(-1, occ_basis) == 0  # vacant to vacant
        assert basis.convert_to(1, occ_basis) == 1   # occupied to occupied


class TestGeneralizedOccupation:
    """Test Occupation class with generalized basis system."""
    
    def test_initialization_with_string_basis(self):
        """Test creating Occupation with string basis names."""
        data = [-1, 1, -1, 1]
        occ = Occupation(data, basis='chebyshev')
        
        assert occ.basis == 'chebyshev'
        assert isinstance(occ.basis_obj, ChebyshevBasis)
        assert occ.values == data
    
    def test_initialization_with_basis_instance(self):
        """Test creating Occupation with basis function instances."""
        data = [0, 1, 0, 1]
        basis_obj = OccupationBasis()
        occ = Occupation(data, basis=basis_obj)
        
        assert occ.basis == 'occupation'
        assert occ.basis_obj is basis_obj
        assert occ.values == data
    
    def test_invalid_basis_type(self):
        """Test that invalid basis types raise errors."""
        with pytest.raises(ValueError, match="Invalid basis type"):
            Occupation([1, 0], basis=42)  # Invalid type
    
    def test_generalized_basis_conversion(self):
        """Test conversion between any registered basis types."""
        data = [-1, 1, -1]
        cheb_occ = Occupation(data, basis='chebyshev')
        
        # Convert to occupation basis using string
        occ_occ = cheb_occ.to_basis('occupation')
        assert occ_occ.basis == 'occupation'
        assert occ_occ.values == [0, 1, 0]
        
        # Convert back using basis instance
        cheb_basis = ChebyshevBasis()
        back_to_cheb = occ_occ.to_basis(cheb_basis)
        assert back_to_cheb.basis == 'chebyshev'
        assert back_to_cheb.values == data
    
    def test_factory_methods_with_basis_instances(self):
        """Test factory methods with basis function instances."""
        cheb_basis = ChebyshevBasis()
        occ_basis = OccupationBasis()
        
        # Test with Chebyshev basis
        zeros_cheb = Occupation.zeros(3, basis=cheb_basis)
        assert zeros_cheb.basis == 'chebyshev'
        assert zeros_cheb.values == [-1, -1, -1]
        
        ones_cheb = Occupation.ones(3, basis=cheb_basis)
        assert ones_cheb.values == [1, 1, 1]
        
        # Test with occupation basis
        zeros_occ = Occupation.zeros(3, basis=occ_basis)
        assert zeros_occ.basis == 'occupation'
        assert zeros_occ.values == [0, 0, 0]
        
        ones_occ = Occupation.ones(3, basis=occ_basis)
        assert ones_occ.values == [1, 1, 1]
        
        # Test random
        random_cheb = Occupation.random(5, basis=cheb_basis, fill_fraction=0.0, seed=42)
        assert all(v == cheb_basis.vacant_value for v in random_cheb.values)


class TestCustomBasisWithOccupation:
    """Test Occupation class with custom basis functions."""
    
    def test_ternary_basis_example(self):
        """Test using a custom ternary basis function with Occupation."""
        
        # Define a ternary basis function (-1, 0, 1)
        @register_basis('ternary_test')  
        class TernaryBasis(BasisFunction):
            @property
            def name(self):
                return 'ternary_test'
            
            @property
            def vacant_value(self):
                return 0
            
            @property
            def occupied_value(self):
                return 1
            
            @property
            def valid_values(self):
                return {-1, 0, 1}
            
            @property
            def basis_function(self):
                return [-1, 0, 1]
            
            def is_occupied(self, value):
                return value == 1
            
            def is_vacant(self, value):
                return value == 0
            
            def flip_value(self, value):
                # Custom flip logic: 0<->1, -1 stays -1
                if value == 0:
                    return 1
                elif value == 1:
                    return 0
                else:  # -1 stays -1 (blocked or fixed site)
                    return -1
        
        # Test custom basis with Occupation
        data = [-1, 0, 1, 0]
        occ = Occupation(data, basis='ternary_test')
        
        assert occ.basis == 'ternary_test'
        assert occ.count_vacant() == 2  # Two 0s
        assert occ.count_occupied() == 1  # One 1
        
        # Test flip operation with custom logic
        flipped = occ.flip([1, 2, 0])  # Try to flip positions 1, 2, 0
        expected = [-1, 1, 0, 0]  # -1 stays -1, 0->1, 1->0, 0 unchanged
        assert flipped.values == expected
        
        # Test factory methods work with custom basis
        zeros = Occupation.zeros(3, basis='ternary_test')
        assert zeros.values == [0, 0, 0]  # vacant_value = 0
        
        ones = Occupation.ones(3, basis='ternary_test')
        assert ones.values == [1, 1, 1]  # occupied_value = 1
    
    def test_custom_float_basis(self):
        """Test custom basis with float values."""
        
        @register_basis('float_test')
        class FloatBasis(BasisFunction):
            @property
            def name(self):
                return 'float_test'
            
            @property
            def vacant_value(self):
                return 0.0
            
            @property
            def occupied_value(self):
                return 1.0
            
            @property
            def valid_values(self):
                return {0.0, 0.5, 1.0}  # Allows partial occupancy
            
            @property
            def basis_function(self):
                return [0.0, 0.5, 1.0]
            
            def is_occupied(self, value):
                return value == 1.0
            
            def is_vacant(self, value):
                return value == 0.0
            
            def flip_value(self, value):
                if value == 0.0:
                    return 1.0
                elif value == 1.0:
                    return 0.0
                else:  # 0.5 stays 0.5
                    return 0.5
        
        # Test with float data
        data = [0.0, 0.5, 1.0]
        occ = Occupation(data, basis='float_test')
        
        assert occ.count_vacant() == 1
        assert occ.count_occupied() == 1
        assert occ.values == [0.0, 0.5, 1.0]
        
        # Test flip
        flipped = occ.flip([0, 2])
        assert flipped.values == [1.0, 0.5, 0.0]


class TestBasisConversionGeneral:
    """Test general basis conversion between any basis functions."""
    
    def test_same_basis_conversion(self):
        """Test conversion to same basis returns equivalent object."""
        data = [-1, 1, -1]
        occ = Occupation(data, basis='chebyshev')
        
        # Convert to same basis
        same_basis = occ.to_basis('chebyshev')
        assert same_basis.values == data
        assert same_basis.basis == 'chebyshev'
        assert same_basis is not occ  # Different object
    
    def test_cross_basis_conversion_consistency(self):
        """Test that conversions are consistent using convert_to interface."""
        # Create test data in both bases
        cheb_data = [-1, 1, -1, 1]
        occ_data = [0, 1, 0, 1]
        
        cheb_occ = Occupation(cheb_data, basis='chebyshev')
        occ_occ = Occupation(occ_data, basis='occupation')
        
        # Convert both ways
        cheb_to_occ = cheb_occ.to_basis('occupation')
        occ_to_cheb = occ_occ.to_basis('chebyshev')
        
        # Should be equivalent to original data
        assert cheb_to_occ.values == occ_data
        assert occ_to_cheb.values == cheb_data
        
        # Test round-trip conversion
        round_trip_cheb = cheb_to_occ.to_basis('chebyshev')
        round_trip_occ = occ_to_cheb.to_basis('occupation')
        
        assert round_trip_cheb.values == cheb_data
        assert round_trip_occ.values == occ_data


if __name__ == '__main__':
    pytest.main([__file__])
