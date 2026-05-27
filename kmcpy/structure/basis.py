"""
Basis functions and occupation management for lattice models in kMCpy.

This module provides different basis functions for converting between species and numerical values
used in cluster expansion and other lattice-based calculations, as well as an Occupation class
for managing site occupation states with support for custom basis functions via registry.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Iterator, Dict, Type, Any

from monty.json import MSONable


# Basis function registry for extensibility
BASIS_REGISTRY: Dict[str, Type['BasisFunction']] = {}


class BasisFunction(MSONable, ABC):
    """
    Abstract base class for all basis functions.
    
    This defines the interface that all basis functions must implement,
    allowing users to create custom basis functions that work with the
    Occupation class.
    """
    
    def __init__(self):
        self.name = self.__class__.__name__.lower().replace('basis', '')

    @property
    def uses_state_indices(self) -> bool:
        """Whether occupations store discrete species-state indices."""
        return False
    
    @property
    @abstractmethod
    def match_value(self) -> Union[int, float]:
        """Value representing specie matches template structure."""
        pass
    
    @property
    @abstractmethod  
    def mismatch_value(self) -> Union[int, float]:
        """Value representing specie doesn't match template structure."""
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

    def num_site_basis_functions(self, n_states: int) -> int:
        """Return number of non-constant site basis functions."""
        return 1

    def state_value(self, state_index: int, n_states: int | None = None) -> Union[int, float]:
        """Return the occupation value used for a species-state index."""
        return self.match_value if int(state_index) == 0 else self.mismatch_value

    def site_basis_values(self, n_states: int) -> np.ndarray:
        """Return basis values with shape ``(n_states, n_basis_functions)``."""
        return np.array(
            [
                [self.state_value(state_index, n_states)]
                for state_index in range(int(n_states))
            ],
            dtype=float,
        )

    def as_dict(self) -> dict:
        """Serialize the basis definition."""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BasisFunction":
        """Create a basis function from a Monty-style payload."""
        if not isinstance(data, dict):
            raise ValueError("BasisFunction.from_dict expects a dictionary")

        target_cls: Type[BasisFunction] = cls
        if cls is BasisFunction:
            class_name = data.get("@class")
            for registered_cls in BASIS_REGISTRY.values():
                if registered_cls.__name__ == class_name:
                    target_cls = registered_cls
                    break
            else:
                basis_name = data.get("name")
                if basis_name in BASIS_REGISTRY:
                    target_cls = BASIS_REGISTRY[basis_name]
                else:
                    raise ValueError(f"Unknown basis payload: {data}")

        kwargs = {
            key: value
            for key, value in data.items()
            if not key.startswith("@") and key != "name"
        }
        return target_cls(**kwargs)
    
    def flip(self, value: Union[int, float]) -> Union[int, float]:
        """Flip between match and mismatch values."""
        return self.mismatch_value if value == self.match_value else self.match_value
    
    def convert_to(self, value: Union[int, float], target_basis: 'BasisFunction') -> Union[int, float]:
        """Convert a value to another basis function."""
        if value == self.mismatch_value:
            return target_basis.mismatch_value
        else:
            return target_basis.match_value
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(match={self.match_value}, mismatch={self.mismatch_value})"


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
    - 0 = vacant (no atom present)
    - 1 = occupied (atom present)

    This is the most intuitive binary representation.
    """

    @property
    def match_value(self) -> int:
        """Value when site matches template (has atom = occupied)."""
        return 1

    @property
    def mismatch_value(self) -> int:
        """Value when site doesn't match template (no atom = vacant)."""
        return 0

    # Physical interpretation properties
    @property
    def vacant_value(self) -> int:
        """Value representing vacant site (no atom)."""
        return 0

    @property
    def occupied_value(self) -> int:
        """Value representing occupied site (atom present)."""
        return 1

    @property
    def valid_values(self) -> set:
        return {0, 1}

    @property
    def basis_function(self) -> List[int]:
        return [0, 1]

    def flip(self, value: int) -> int:
        """Flip between match and mismatch."""
        return 1 - value

    def flip_value(self, value: int) -> int:
        """Alias for flip for backward compatibility."""
        return self.flip(value)

    def is_occupied(self, value: int) -> bool:
        """Check if value represents occupied state."""
        return value == self.occupied_value

    def is_vacant(self, value: int) -> bool:
        """Check if value represents vacant state."""
        return value == self.vacant_value

    def to_chebyshev(self, value: int) -> int:
        """Convert occupation value to the Chebyshev species-state basis."""
        return 1 if value == self.vacant_value else 0

    def from_chebyshev(self, value: int) -> int:
        """Convert the Chebyshev species-state value to occupation basis."""
        return self.occupied_value if value == 0 else self.vacant_value


@register_basis('chebyshev')
class ChebyshevBasis(BasisFunction):
    """
    Discrete site-state Chebyshev basis used for cluster expansion.

    kMCpy stores occupations as species-state indices. A site with ``q``
    allowed species has states ``0`` through ``q - 1`` and ``q - 1``
    non-constant Chebyshev basis functions.
    """

    def __init__(self, max_states: int = 2):
        super().__init__()
        if int(max_states) < 2:
            raise ValueError("ChebyshevBasis requires at least two states")
        self.max_states = int(max_states)

    @property
    def uses_state_indices(self) -> bool:
        """Chebyshev occupations are stored as species-state indices."""
        return True

    @property
    def match_value(self) -> int:
        """Value when site matches the first mapped species."""
        return 0

    @property
    def mismatch_value(self) -> int:
        """Value when site is missing or matches another allowed species."""
        return 1

    @property
    def occupied_value(self) -> int:
        """Alias for the first mapped species state."""
        return self.match_value

    @property
    def vacant_value(self) -> int:
        """Alias for the missing/second-species state."""
        return self.mismatch_value

    @property
    def valid_values(self) -> set:
        return set(range(self.max_states))

    @property
    def basis_function(self) -> List[Union[int, float, list]]:
        return self.site_basis_values(self.max_states).tolist()

    def flip(self, value: int) -> int:
        """Flip between match and mismatch."""
        return self.mismatch_value if value == self.match_value else self.match_value

    def flip_value(self, value: int) -> int:
        """Alias for flip for backward compatibility."""
        return self.flip(value)

    def is_occupied(self, value: int) -> bool:
        """Check if value represents the first mapped species state."""
        return value == self.occupied_value

    def is_vacant(self, value: int) -> bool:
        """Check if value represents the missing/second-species state."""
        return value == self.vacant_value

    def to_occupation(self, value: int) -> int:
        """
        Convert Chebyshev value to occupation basis.

        Chebyshev: 0 (first species), 1 (missing/second species)
        Occupation: 0 (vacant), 1 (occupied)
        """
        return 1 if value == self.occupied_value else 0

    def from_occupation(self, value: int) -> int:
        """
        Convert occupation value to Chebyshev basis.

        Occupation: 0 (vacant), 1 (occupied)
        Chebyshev: 0 (first species), 1 (missing/second species)
        """
        return self.occupied_value if value == 1 else self.vacant_value

    def num_site_basis_functions(self, n_states: int) -> int:
        """Return ``q - 1`` non-constant basis functions for ``q`` states."""
        n_states = int(n_states)
        if n_states < 2:
            return 1
        return n_states - 1

    def state_value(self, state_index: int, n_states: int | None = None) -> int:
        """Return the occupation value for a species-state index."""
        return int(state_index)

    def site_basis_values(self, n_states: int) -> np.ndarray:
        """
        Return discrete Chebyshev values for a site's allowed species.

        The species states are encoded as equally spaced points in ``[-1, 1]``.
        For ``q`` species, the returned columns are ``T_1`` through
        ``T_{q-1}`` evaluated at those points.
        """
        n_states = int(n_states)
        if n_states < 2:
            return np.ones((n_states, 1), dtype=float)
        encoded = np.linspace(-1.0, 1.0, n_states)
        return np.polynomial.chebyshev.chebvander(encoded, n_states - 1)[:, 1:]

    def as_dict(self) -> dict:
        payload = super().as_dict()
        payload["max_states"] = self.max_states
        return payload


def get_basis(name: str, **kwargs) -> BasisFunction:
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
    return BASIS_REGISTRY[name](**kwargs)


class Occupation:
    """
    Encapsulates site occupation data with basis conversion and validation.
    
    Provides a clean interface for managing occupation arrays with different
    basis representations. Supports any registered basis function, allowing
    users to define custom basis functions.
    
    Features:
    - Automatic validation of occupation values using basis classes
    - Basis conversion between any registered basis types
    - Common operations like counting match/mismatch sites  
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
        
        self._data = np.array(data, dtype=type(self._basis_obj.match_value))
        
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
    def count_mismatch(self) -> int:
        """Count sites with mismatch (specie doesn't match template)."""
        return int(np.sum(self._data == self._basis_obj.mismatch_value))

    def count_match(self) -> int:
        """Count sites with match (specie matches template)."""
        return int(np.sum(self._data == self._basis_obj.match_value))

    # Physical occupation counts (use basis object's definitions)
    def count_occupied(self) -> int:
        """Count sites with occupied state."""
        return int(np.sum(self._data == self._basis_obj.occupied_value))

    def count_vacant(self) -> int:
        """Count sites with vacant state."""
        return int(np.sum(self._data == self._basis_obj.vacant_value))

    def get_mismatch_indices(self) -> List[int]:
        """Get indices of sites with mismatch (specie doesn't match template)."""
        return np.where(self._data == self._basis_obj.mismatch_value)[0].tolist()

    def get_match_indices(self) -> List[int]:
        """Get indices of sites with match (specie matches template)."""
        return np.where(self._data == self._basis_obj.match_value)[0].tolist()

    def get_occupied_indices(self) -> List[int]:
        """Get indices of occupied sites."""
        return np.where(self._data == self._basis_obj.occupied_value)[0].tolist()

    def get_vacant_indices(self) -> List[int]:
        """Get indices of vacant sites."""
        return np.where(self._data == self._basis_obj.vacant_value)[0].tolist()
    
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
            # Use flip_value if available (for custom flip logic), otherwise use flip
            if hasattr(self._basis_obj, 'flip_value'):
                new_data[idx] = self._basis_obj.flip_value(new_data[idx])
            else:
                new_data[idx] = self._basis_obj.flip(new_data[idx])
                
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
            # Use flip_value if available (for custom flip logic), otherwise use flip
            if hasattr(self._basis_obj, 'flip_value'):
                self._data[idx] = self._basis_obj.flip_value(self._data[idx])
            else:
                self._data[idx] = self._basis_obj.flip(self._data[idx])
    
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
        Create Occupation with all sites vacant (no atoms).

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

        # Use vacant_value from basis object
        data = np.full(n_sites, basis_obj.vacant_value, dtype=type(basis_obj.vacant_value))
        return cls(data, basis=basis_obj, validate=False)

    @classmethod
    def ones(cls, n_sites: int, basis: Union[str, BasisFunction] = 'chebyshev') -> 'Occupation':
        """
        Create Occupation with all sites occupied (atoms present).

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

        # Use occupied_value from basis object
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
            fill_fraction: Fraction of sites to be occupied (atoms present) (0.0 to 1.0)
            seed: Random seed for reproducibility

        Returns:
            Random Occupation object
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate random boolean array for occupied sites
        occupied = np.random.random(n_sites) < fill_fraction

        if isinstance(basis, str):
            basis_obj = get_basis(basis)
        else:
            basis_obj = basis

        # Use occupied_value and vacant_value from basis object
        data = np.where(occupied, basis_obj.occupied_value, basis_obj.vacant_value)
        return cls(data, basis=basis_obj, validate=False)
