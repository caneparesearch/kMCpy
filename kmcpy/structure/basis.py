"""
Basis functions and occupation management for lattice models in kMCpy.

This module provides different basis functions for converting between species and numerical values
used in cluster expansion and other lattice-based calculations, as well as an Occupation class
for managing site occupation states with support for custom basis functions via registry.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Iterator, Dict, Type, Any


# Basis function registry for extensibility
BASIS_REGISTRY: Dict[str, Type['BasisFunction']] = {}


class BasisFunction(ABC):
    """
    Abstract base class for all basis functions.
    
    This defines the interface that all basis functions must implement,
    allowing users to create custom basis functions that work with the
    Occupation class.
    """
    
    def __init__(self):
        self.name = self.__class__.__name__.lower().replace('basis', '')
    
    @property
    @abstractmethod
    def vacant_value(self) -> Union[int, float]:
        """Value representing a vacant site."""
        pass
    
    @property
    @abstractmethod  
    def occupied_value(self) -> Union[int, float]:
        """Value representing an occupied site."""
        pass
    
    @property
    @abstractmethod
    def valid_values(self) -> set:
        """Set of all valid values in this basis."""
        pass
    
    @property
    @abstractmethod
    def basis_function(self) -> List[Union[int, float]]:
        """List defining the basis function values."""
        pass
    
    @abstractmethod
    def is_occupied(self, value: Union[int, float]) -> bool:
        """Check if a value represents an occupied site."""
        pass
    
    @abstractmethod
    def is_vacant(self, value: Union[int, float]) -> bool:
        """Check if a value represents a vacant site."""
        pass
    
    @abstractmethod
    def flip_value(self, value: Union[int, float]) -> Union[int, float]:
        """Flip between occupied and vacant."""
        pass
    
    def convert_to(self, value: Union[int, float], target_basis: 'BasisFunction') -> Union[int, float]:
        """
        Convert a value to another basis function.
        
        Default implementation assumes binary occupation states.
        Override for more complex conversions.
        """
        if self.is_occupied(value):
            return target_basis.occupied_value
        else:
            return target_basis.vacant_value
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(vacant={self.vacant_value}, occupied={self.occupied_value})"


def register_basis(name: str):
    """
    Decorator to register a basis function class.
    
    Args:
        name: Name to register the basis function under
    """
    def decorator(cls: Type[BasisFunction]):
        BASIS_REGISTRY[name] = cls
        return cls
    return decorator


@register_basis('occupation')
class OccupationBasis(BasisFunction):
    """
    Occupation basis function that maps between binary occupation states.
    Uses [0, 1] representation for site occupancy.
    
    In occupation basis:
    - 0 = vacant site (same as template)
    - 1 = occupied site (different from template)
    """
    
    @property
    def vacant_value(self) -> int:
        return 0
    
    @property
    def occupied_value(self) -> int:
        return 1
    
    @property
    def valid_values(self) -> set:
        return {0, 1}
    
    @property
    def basis_function(self) -> List[int]:
        return [0, 1]
    
    def is_occupied(self, value: int) -> bool:
        """Check if a value represents an occupied site."""
        return value == self.occupied_value
    
    def is_vacant(self, value: int) -> bool:
        """Check if a value represents a vacant site."""
        return value == self.vacant_value
    
    def flip_value(self, value: int) -> int:
        """Flip between occupied and vacant."""
        return 1 - value
    
    def to_chebyshev(self, value: int) -> int:
        """Convert occupation value to Chebyshev representation."""
        return -1 if value == 0 else 1
    
    def from_chebyshev(self, value: int) -> int:
        """Convert Chebyshev value to occupation representation."""
        return 0 if value == -1 else 1


@register_basis('chebyshev')
class ChebyshevBasis(BasisFunction):
    """
    Chebyshev basis function that maps between [-1, +1] representation.
    Often used in cluster expansion for better numerical properties.
    
    In Chebyshev basis:
    - -1 = vacant site (same as template)
    - +1 = occupied site (different from template)
    """
    
    @property
    def vacant_value(self) -> int:
        return -1
    
    @property
    def occupied_value(self) -> int:
        return 1
    
    @property
    def valid_values(self) -> set:
        return {-1, 1}
    
    @property
    def basis_function(self) -> List[int]:
        return [-1, 1]
    
    def is_occupied(self, value: int) -> bool:
        """Check if a value represents an occupied site."""
        return value == self.occupied_value
    
    def is_vacant(self, value: int) -> bool:
        """Check if a value represents a vacant site."""
        return value == self.vacant_value
    
    def flip_value(self, value: int) -> int:
        """Flip between occupied and vacant."""
        return -value
    
    def to_occupation(self, value: int) -> int:
        """Convert Chebyshev value to occupation representation."""
        return 0 if value == -1 else 1
    
    def from_occupation(self, value: int) -> int:
        """Convert occupation value to Chebyshev representation."""
        return -1 if value == 0 else 1


def get_basis(name: str) -> BasisFunction:
    """
    Get a basis function instance by name.
    
    Args:
        name: Name of the basis function
        
    Returns:
        Instance of the requested basis function
        
    Raises:
        ValueError: If basis name is not registered
    """
    if name not in BASIS_REGISTRY:
        raise ValueError(f"Unknown basis '{name}'. Available: {list(BASIS_REGISTRY.keys())}")
    return BASIS_REGISTRY[name]()


class Occupation:
    """
    Encapsulates site occupation data with basis conversion and validation.
    
    Provides a clean interface for managing occupation arrays with different
    basis representations. Supports any registered basis function, allowing
    users to define custom basis functions.
    
    Features:
    - Automatic validation of occupation values using basis classes
    - Basis conversion between any registered basis types
    - Common operations like counting occupied/vacant sites  
    - Immutable and mutable variants
    - Efficient numpy operations under the hood
    - Extensible via basis function registry
    """
    
    def __init__(self, data: Union[List[int], Tuple[int], np.ndarray], 
                 basis: Union[str, BasisFunction] = 'chebyshev', validate: bool = True):
        """
        Initialize occupation array.
        
        Args:
            data: Occupation values as list, tuple, or numpy array
            basis: Basis function - either a string name or BasisFunction instance
            validate: Whether to validate occupation values against basis
            
        Raises:
            ValueError: If basis is invalid or data doesn't match basis constraints
        """
        if isinstance(basis, str):
            self._basis_obj = get_basis(basis)
            self._basis_name = basis
        elif isinstance(basis, BasisFunction):
            self._basis_obj = basis
            self._basis_name = basis.name
        else:
            raise ValueError(f"Invalid basis type. Must be string or BasisFunction instance")
        
        self._data = np.array(data, dtype=type(self._basis_obj.vacant_value))
        
        if validate:
            self._validate()
    
    def _validate(self):
        """Validate occupation values against the specified basis."""
        invalid = set(self._data) - self._basis_obj.valid_values
        if invalid:
            raise ValueError(f"Invalid values {invalid} for {self._basis_name} basis. "
                           f"Must be in {self._basis_obj.valid_values}")
    
    @property
    def basis(self) -> str:
        """Get the basis type name."""
        return self._basis_name
    
    @property
    def basis_obj(self) -> BasisFunction:
        """Get the basis function object."""
        return self._basis_obj
    
    @property 
    def data(self) -> np.ndarray:
        """Get the underlying numpy array (read-only view)."""
        return self._data.copy()  # Return copy to maintain immutability
    
    @property
    def array(self) -> np.ndarray:
        """Get the underlying numpy array (alias for data, for backward compatibility)."""
        return self.data
    
    @property
    def values(self) -> List[int]:
        """Get occupation values as a list."""
        return self._data.tolist()
    
    def __len__(self) -> int:
        """Return number of sites."""
        return len(self._data)
    
    def __getitem__(self, index: Union[int, slice, np.ndarray, List[int]]) -> Union[int, 'Occupation']:
        """Get occupation value(s) at index."""
        if isinstance(index, slice):
            return Occupation(self._data[index], basis=self._basis_obj, validate=False)
        elif isinstance(index, (list, tuple, np.ndarray)):
            # Handle array-like indexing (returns new Occupation object)
            return Occupation(self._data[index], basis=self._basis_obj, validate=False)
        else:
            # Single index (returns scalar)
            return self._data[index].item()
    
    def __setitem__(self, index: Union[int, slice], value: Union[int, List[int]]):
        """Set occupation value(s) at index."""
        self._data[index] = value
        if hasattr(self, '_validate'):  # Only validate if not in constructor
            self._validate()
    
    def __iter__(self) -> Iterator[int]:
        """Iterate over occupation values."""
        return iter(self._data.tolist())
    
    def __eq__(self, other) -> bool:
        """Check equality with another Occupation object."""
        if not isinstance(other, Occupation):
            return False
        return self._basis_name == other._basis_name and np.array_equal(self._data, other._data)
    
    def __ne__(self, other) -> bool:
        """Check inequality with another Occupation object."""
        return not self.__eq__(other)
    
    def equivalent_to(self, other: 'Occupation') -> bool:
        """
        Check if two Occupation objects represent the same occupation pattern,
        regardless of basis type.
        
        This is useful for comparing occupations that might be in different bases
        but represent the same physical state.
        
        Args:
            other: Another Occupation object to compare with
            
        Returns:
            True if both objects represent the same occupation pattern
        """
        if not isinstance(other, Occupation):
            return False
        
        # Convert both to occupation basis for comparison
        self_occ = self.to_basis('occupation')
        other_occ = other.to_basis('occupation')
        return np.array_equal(self_occ._data, other_occ._data)
    
    def array_equal(self, array: Union[List, Tuple, np.ndarray]) -> bool:
        """
        Check if the underlying data array equals the given array.
        
        This is useful for unit tests where you want to compare the raw data
        without creating another Occupation object.
        
        Args:
            array: Array-like object to compare with
            
        Returns:
            True if the arrays are equal
        """
        return np.array_equal(self._data, np.array(array))
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Occupation({self.values}, basis='{self._basis_name}')"
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        return f"{self._basis_name.title()} occupation: {self.values}"
    
    # Convenience methods for common operations using basis objects
    def count_occupied(self) -> int:
        """Count occupied sites using basis object."""
        return int(np.sum([self._basis_obj.is_occupied(val) for val in self._data]))
    
    def count_vacant(self) -> int:
        """Count vacant sites using basis object."""
        return int(np.sum([self._basis_obj.is_vacant(val) for val in self._data]))
    
    def get_occupied_indices(self) -> List[int]:
        """Get indices of occupied sites."""
        return [i for i, val in enumerate(self._data) if self._basis_obj.is_occupied(val)]
    
    def get_vacant_indices(self) -> List[int]:
        """Get indices of vacant sites."""
        return [i for i, val in enumerate(self._data) if self._basis_obj.is_vacant(val)]
    
    def flip(self, indices: Union[int, List[int]]) -> 'Occupation':
        """
        Return new Occupation with flipped values at specified indices.
        
        Args:
            indices: Site index or list of indices to flip
            
        Returns:
            New Occupation object with flipped values
        """
        new_data = self._data.copy()
        if isinstance(indices, int):
            indices = [indices]
        
        for idx in indices:
            new_data[idx] = self._basis_obj.flip_value(new_data[idx])
                
        return Occupation(new_data, basis=self._basis_obj, validate=False)
    
    def flip_inplace(self, indices: Union[int, List[int]]) -> None:
        """
        Flip values at specified indices in-place.
        
        Args:
            indices: Site index or list of indices to flip
        """
        if isinstance(indices, int):
            indices = [indices]
        
        for idx in indices:
            self._data[idx] = self._basis_obj.flip_value(self._data[idx])
    
    def to_basis(self, target_basis: Union[str, BasisFunction]) -> 'Occupation':
        """
        Convert to different basis representation using basis objects.
        
        Args:
            target_basis: Target basis - either string name or BasisFunction instance
            
        Returns:
            New Occupation object in target basis
        """
        if isinstance(target_basis, str):
            target_basis_obj = get_basis(target_basis)
        else:
            target_basis_obj = target_basis
            
        if target_basis_obj.__class__ == self._basis_obj.__class__:
            return Occupation(self._data, basis=target_basis_obj, validate=False)
        
        # General conversion using basis function interface
        converted = np.array([self._basis_obj.convert_to(val, target_basis_obj) for val in self._data])
        
        return Occupation(converted, basis=target_basis_obj, validate=False)
    
    def copy(self) -> 'Occupation':
        """Create a deep copy of this Occupation."""
        return Occupation(self._data.copy(), basis=self._basis_obj, validate=False)
    
    @classmethod
    def zeros(cls, n_sites: int, basis: Union[str, BasisFunction] = 'chebyshev') -> 'Occupation':
        """
        Create Occupation with all sites vacant using appropriate basis.
        
        Args:
            n_sites: Number of sites
            basis: Basis function or name
            
        Returns:
            Occupation object with all sites vacant
        """
        if isinstance(basis, str):
            basis_obj = get_basis(basis)
        else:
            basis_obj = basis
            
        data = np.full(n_sites, basis_obj.vacant_value, dtype=type(basis_obj.vacant_value))
        return cls(data, basis=basis_obj, validate=False)
    
    @classmethod  
    def ones(cls, n_sites: int, basis: Union[str, BasisFunction] = 'chebyshev') -> 'Occupation':
        """
        Create Occupation with all sites occupied using appropriate basis.
        
        Args:
            n_sites: Number of sites
            basis: Basis function or name
            
        Returns:
            Occupation object with all sites occupied
        """
        if isinstance(basis, str):
            basis_obj = get_basis(basis)
        else:
            basis_obj = basis
            
        data = np.full(n_sites, basis_obj.occupied_value, dtype=type(basis_obj.occupied_value))
        return cls(data, basis=basis_obj, validate=False)
    
    @classmethod
    def random(cls, n_sites: int, basis: Union[str, BasisFunction] = 'chebyshev', 
               fill_fraction: float = 0.5, seed: int = None) -> 'Occupation':
        """
        Create random Occupation using appropriate basis values.
        
        Args:
            n_sites: Number of sites
            basis: Basis function or name
            fill_fraction: Fraction of sites to occupy (0.0 to 1.0)
            seed: Random seed for reproducibility
            
        Returns:
            Random Occupation object
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random boolean array
        occupied = np.random.random(n_sites) < fill_fraction
        
        if isinstance(basis, str):
            basis_obj = get_basis(basis)
        else:
            basis_obj = basis
            
        data = np.where(occupied, basis_obj.occupied_value, basis_obj.vacant_value)
        return cls(data, basis=basis_obj, validate=False)
