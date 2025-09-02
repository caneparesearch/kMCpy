"""
Local environment comparator for neighbor sequence matching.

This module provides functionality to match and compare local environments
by ensuring that neighbor sequences are consistently ordered according to
their distance matrices.
"""

import numpy as np
import itertools
import logging
from typing import List, Dict, Tuple, Optional, Any
from pymatgen.util.coord import get_angle

logger = logging.getLogger(__name__)


class LocalEnvironmentComparator:
    """
    A comparator for local environments that ensures consistent neighbor ordering
    based on distance matrices. This replaces the NeighborInfoMatcher functionality
    with a cleaner, more integrated approach.
    """
    
    def __init__(
        self,
        neighbor_info: List[Dict],
        rtol: float = 1e-3,
        atol: float = 1e-3
    ):
        """
        Initialize the local environment comparator.
        
        Args:
            neighbor_info: List of neighbor information dictionaries
            rtol: Relative tolerance for distance matrix comparison
            atol: Absolute tolerance for distance matrix comparison
        """
        self.neighbor_info = neighbor_info
        self.rtol = rtol
        self.atol = atol
        self._analyze_environment()
    
    def _analyze_environment(self):
        """Analyze the local environment and create reference data."""
        # Group neighbors by species
        self.neighbor_species_groups = {}
        self.neighbor_species_counts = {}
        
        for neighbor in self.neighbor_info:
            species = neighbor["site"].species_string
            if species not in self.neighbor_species_groups:
                self.neighbor_species_groups[species] = []
                self.neighbor_species_counts[species] = 0
            
            self.neighbor_species_groups[species].append(neighbor)
            self.neighbor_species_counts[species] += 1
        
        # Create environment signature
        self.signature = self._create_signature()
        
        # Build distance matrix
        self.distance_matrix = self._build_distance_matrix()
        
        # Build distance matrices for each species group
        self.species_distance_matrices = {}
        for species, neighbors in self.neighbor_species_groups.items():
            self.species_distance_matrices[species] = self._build_distance_matrix(neighbors)
    
    def _create_signature(self) -> Tuple:
        """Create a unique signature for this environment type."""
        species_counts = []
        for species in sorted(self.neighbor_species_counts.keys()):
            species_counts.append((species, self.neighbor_species_counts[species]))
        return tuple(species_counts)
    
    def _build_distance_matrix(self, neighbors: Optional[List[Dict]] = None) -> np.ndarray:
        """
        Build distance matrix from neighbor information.
        
        Args:
            neighbors: List of neighbors (uses all if None)
            
        Returns:
            2D distance matrix
        """
        if neighbors is None:
            neighbors = self.neighbor_info
        
        n = len(neighbors)
        if n == 0:
            return np.array([])
        
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                # For local environments, sites don't need jimage parameter since
                # they're already in the local coordinate system
                distance_matrix[i, j] = neighbors[i]["site"].distance(
                    neighbors[j]["site"]
                )
        
        return distance_matrix
    
    def _build_angle_matrix(self, neighbors: Optional[List[Dict]] = None) -> np.ndarray:
        """Build angle matrix from neighbor information."""
        if neighbors is None:
            neighbors = self.neighbor_info
            
        n = len(neighbors)
        if n == 0:
            return np.array([])
        
        angle_matrix = np.zeros((n, n, n))
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and i != k:
                        v1 = neighbors[j]["site"].coords - neighbors[i]["site"].coords
                        v2 = neighbors[k]["site"].coords - neighbors[i]["site"].coords
                        angle_matrix[i, j, k] = get_angle(v1, v2, "degrees")
        
        return angle_matrix
    
    def match_neighbor_sequence(
        self,
        other_neighbors: List[Dict],
        find_nearest_if_fail: bool = False
    ) -> List[Dict]:
        """
        Match and reorder another neighbor sequence to match this reference.
        
        Args:
            other_neighbors: List of neighbor info to reorder
            find_nearest_if_fail: Find best match if exact match fails
            
        Returns:
            Reordered neighbor sequence
            
        Raises:
            ValueError: If no valid matching sequence is found
        """
        other_comparator = LocalEnvironmentComparator(other_neighbors, self.rtol, self.atol)
        
        # Check if environments are compatible
        if self.signature != other_comparator.signature:
            raise ValueError(f"Environment signatures don't match: {self.signature} vs {other_comparator.signature}")
        
        # Check if already matching
        if np.allclose(self.distance_matrix, other_comparator.distance_matrix, 
                      rtol=self.rtol, atol=self.atol):
            logger.debug("Neighbor sequences already match - no reordering needed")
            return other_neighbors
        
        # Try to find matching sequence for each species group
        reordered_groups = {}
        
        for species in self.neighbor_species_groups.keys():
            reference_matrix = self.species_distance_matrices[species]
            other_matrix = other_comparator.species_distance_matrices[species]
            other_group = other_comparator.neighbor_species_groups[species]
            
            # Find the best permutation for this species group
            best_sequence = self._find_best_permutation(
                reference_matrix, other_matrix, other_group, find_nearest_if_fail
            )
            
            reordered_groups[species] = best_sequence
        
        # Combine all species groups back into a single list
        reordered_neighbors = []
        for species in sorted(reordered_groups.keys()):
            reordered_neighbors.extend(reordered_groups[species])
        
        # Validate the final result
        final_comparator = LocalEnvironmentComparator(reordered_neighbors, self.rtol, self.atol)
        if not np.allclose(self.distance_matrix, final_comparator.distance_matrix,
                          rtol=self.rtol, atol=self.atol):
            if not find_nearest_if_fail:
                raise ValueError("Could not find exact matching sequence")
            logger.warning("Using approximate match - exact match not found")
        
        return reordered_neighbors
    
    def _find_best_permutation(
        self,
        reference_matrix: np.ndarray,
        other_matrix: np.ndarray,
        other_neighbors: List[Dict],
        find_nearest_if_fail: bool = False
    ) -> List[Dict]:
        """Find the best permutation of neighbors to match reference matrix."""
        n = len(other_neighbors)
        if n == 0:
            return []
        
        if n == 1:
            return other_neighbors
        
        # Use efficient algorithm for larger groups
        best_permutation = self._efficient_permutation_search(
            reference_matrix, other_matrix, other_neighbors
        )
        
        if best_permutation is not None:
            return best_permutation
        
        if find_nearest_if_fail:
            # Fall back to brute force search for best approximate match
            return self._brute_force_best_match(
                reference_matrix, other_matrix, other_neighbors
            )
        
        raise ValueError("No matching permutation found")
    
    def _efficient_permutation_search(
        self,
        reference_matrix: np.ndarray,
        other_matrix: np.ndarray,
        other_neighbors: List[Dict]
    ) -> Optional[List[Dict]]:
        """
        Efficient permutation search using progressive building approach.
        This is much faster than brute force for large neighbor groups.
        """
        n = len(other_neighbors)
        
        # Build permutation progressively
        possible_sequences = [[i] for i in range(n)]
        
        for length in range(2, n + 1):
            new_sequences = []
            reference_submatrix = reference_matrix[:length, :length]
            
            for partial_sequence in possible_sequences:
                for next_idx in range(n):
                    if next_idx not in partial_sequence:
                        test_sequence = partial_sequence + [next_idx]
                        test_matrix = self._build_submatrix(other_matrix, test_sequence)
                        
                        if np.allclose(reference_submatrix, test_matrix, 
                                     rtol=self.rtol, atol=self.atol):
                            new_sequences.append(test_sequence)
            
            possible_sequences = new_sequences
            
            if not possible_sequences:
                return None
        
        if possible_sequences:
            best_sequence = possible_sequences[0]
            return [other_neighbors[i] for i in best_sequence]
        
        return None
    
    def _build_submatrix(self, matrix: np.ndarray, indices: List[int]) -> np.ndarray:
        """Build submatrix from given indices."""
        n = len(indices)
        submatrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                submatrix[i, j] = matrix[indices[i], indices[j]]
        
        return submatrix
    
    def _brute_force_best_match(
        self,
        reference_matrix: np.ndarray,
        other_matrix: np.ndarray,
        other_neighbors: List[Dict]
    ) -> List[Dict]:
        """Brute force search for best approximate match."""
        best_score = float('inf')
        best_permutation = None
        
        for perm in itertools.permutations(range(len(other_neighbors))):
            test_matrix = self._build_submatrix(other_matrix, list(perm))
            score = np.sum(np.abs(reference_matrix - test_matrix))
            
            if score < best_score:
                best_score = score
                best_permutation = perm
        
        if best_permutation is not None:
            logger.info(f"Best approximate match found with score: {best_score}")
            return [other_neighbors[i] for i in best_permutation]
        
        raise ValueError("Could not find any permutation")
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get comprehensive information about this local environment."""
        return {
            'signature': self.signature,
            'neighbor_count': len(self.neighbor_info),
            'species_counts': self.neighbor_species_counts,
            'distance_matrix_shape': self.distance_matrix.shape,
            'species_groups': list(self.neighbor_species_groups.keys())
        }
    
    @classmethod
    def from_local_lattice_structure(
        cls,
        local_lattice_structure,
        rtol: float = 1e-3,
        atol: float = 1e-3
    ):
        """
        Create a LocalEnvironmentComparator from a LocalLatticeStructure.
        
        This method bridges the gap between the structure classes and the 
        neighbor matching functionality.
        """
        # Convert LocalLatticeStructure to neighbor info format
        neighbor_info = []
        
        for i, site in enumerate(local_lattice_structure.structure):
            # Create neighbor info dictionary similar to get_nn_info output
            # For Molecule sites, we need to handle them differently
            neighbor_dict = {
                "site": site,
                "image": (0, 0, 0),  # Local environment has no images
                "weight": 0.0,  # Can be calculated if needed
                "site_index": i,
                "local_index": i
            }
            neighbor_info.append(neighbor_dict)
        
        return cls(neighbor_info, rtol, atol)


# Convenience functions for integration with existing code
def create_environment_comparator(neighbor_info: List[Dict], **kwargs) -> LocalEnvironmentComparator:
    """Factory function to create a LocalEnvironmentComparator."""
    return LocalEnvironmentComparator(neighbor_info, **kwargs)


def match_local_environments(
    reference_neighbors: List[Dict],
    target_neighbors: List[Dict],
    rtol: float = 1e-3,
    atol: float = 1e-3,
    find_nearest_if_fail: bool = False
) -> List[Dict]:
    """
    Convenience function to match two local environments.
    
    Args:
        reference_neighbors: Reference neighbor sequence
        target_neighbors: Target neighbors to reorder
        rtol: Relative tolerance
        atol: Absolute tolerance
        find_nearest_if_fail: Find approximate match if exact fails
        
    Returns:
        Reordered target neighbors matching reference sequence
    """
    reference_comparator = LocalEnvironmentComparator(reference_neighbors, rtol, atol)
    return reference_comparator.match_neighbor_sequence(target_neighbors, find_nearest_if_fail)
