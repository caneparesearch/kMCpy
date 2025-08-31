

"""
Clean simulation state management.

SimulationState: Pure mutable state during simulation execution.
No configuration mixed in, just the evolving state.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
import json
import numpy as np

from ..event.event import Event

if TYPE_CHECKING:
    from ..structure.basis import Occupation


class SimulationState:
    """
    Pure mutable state during KMC simulation.
    
    This contains ONLY the state that changes during simulation:
    - Current site occupations (as Occupation object with basis support)
    - Simulation time and step count  
    - Mobile species tracking data
    
    No configuration parameters - those belong in SimulationConfig.
    Generic design supports any KMC process (diffusion, reactions, etc.)
    
    The occupations are stored as an Occupation object, providing:
    - Type safety and validation
    - Basis function support (Chebyshev, occupation, custom)
    - KMC-specific operations (site flipping, event application)
    - Automatic conversion between basis representations
    """
    
    def __init__(
        self, 
        occupations: 'Occupation',
        time: float = 0.0,
        step: int = 0
    ):
        """
        Initialize simulation state.
        
        Args:
            occupations: Current site occupations as Occupation object
            time: Current simulation time
            step: Current step number
        """
        from ..structure.basis import Occupation
        
        if not isinstance(occupations, Occupation):
            raise TypeError(f"occupations must be an Occupation object, got {type(occupations)}")
        
        self.occupations = occupations.copy()  # Keep as Occupation object
        self.time = time
        self.step = step
        
        # Mobile species tracking (initialized later by simulator)
        self.mobile_species_positions: Optional[np.ndarray] = None
        self.displacements: Optional[np.ndarray] = None
        self.hop_counts: Optional[np.ndarray] = None
        self.n_mobile_species: int = 0
    
    def apply_event(self, event: Event, dt: float) -> None:
        """
        Apply a KMC event to update state.
        
        This is a generic method that can handle any type of site-to-site transition:
        - Ion hopping
        - Vacancy diffusion  
        - Spin flips
        - Chemical reactions
        - etc.
        
        Args:
            event: Event object containing site indices to transition between
            dt: Time increment for this event
        """
        # Extract site indices from event
        from_site, to_site = event.mobile_ion_indices
        
        # Use Occupation object's KMC event method
        self.occupations.apply_event_inplace(from_site, to_site)
        
        # Update time and step
        self.time += dt
        self.step += 1
    
    def get_mobile_species_sites(self) -> List[int]:
        """Get indices of sites currently occupied by mobile species."""
        return self.occupations.get_occupied_indices()
    
    def get_vacant_sites(self) -> List[int]:
        """Get indices of vacant sites.""" 
        return self.occupations.get_vacant_indices()
    
    def count_mobile_species(self) -> int:
        """Count total mobile species in system."""
        return self.occupations.count_occupied()
    
    def initialize_tracking(self, structure_data: Dict[str, Any]) -> None:
        """
        Initialize mobile species tracking arrays.
        Called once by simulator after loading structure.
        
        Args:
            structure_data: Dictionary with lattice and coordinates info
        """
        n_mobile = self.count_mobile_species()
        
        self.n_mobile_species = n_mobile
        self.mobile_species_positions = np.zeros((n_mobile, 3))
        self.displacements = np.zeros((n_mobile, 3))
        self.hop_counts = np.zeros(n_mobile, dtype=int)
        
        # Initialize positions from current occupations
        mobile_sites = self.get_mobile_species_sites()
        if 'frac_coords' in structure_data:
            coords = structure_data['frac_coords']
            for i, site_idx in enumerate(mobile_sites):
                self.mobile_species_positions[i] = coords[site_idx]
    
    def update_tracking(self, species_index: int, new_position: np.ndarray) -> None:
        """
        Update tracking data for a mobile species.
        
        Args:
            species_index: Index of the mobile species
            new_position: New fractional coordinates
        """
        if self.mobile_species_positions is not None:
            # Calculate displacement
            old_pos = self.mobile_species_positions[species_index]
            displacement = new_position - old_pos
            
            # Handle periodic boundaries
            displacement = displacement - np.round(displacement)
            
            # Update arrays
            self.displacements[species_index] += displacement
            self.mobile_species_positions[species_index] = new_position
            self.hop_counts[species_index] += 1
    
    def copy(self) -> "SimulationState":
        """Create deep copy of this state."""
        new_state = SimulationState(
            occupations=self.occupations,  # Constructor will copy the Occupation object
            time=self.time,
            step=self.step
        )
        
        # Deep copy tracking data if it exists
        if self.mobile_species_positions is not None:
            new_state.mobile_species_positions = self.mobile_species_positions.copy()
            new_state.displacements = self.displacements.copy()
            new_state.hop_counts = self.hop_counts.copy()
            new_state.n_mobile_species = self.n_mobile_species
        
        return new_state

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save current state to checkpoint file.
        
        Args:
            filepath: Path to save checkpoint (.json or .h5)
        """
        data = {
            'occupations': self.occupations.values,  # Save as list for JSON compatibility
            'occupation_basis': self.occupations.basis,  # Save basis information
            'time': self.time,
            'step': self.step,
            'n_mobile_species': self.n_mobile_species
        }
        
        # Add tracking data if available
        if self.mobile_species_positions is not None:
            data.update({
                'mobile_species_positions': self.mobile_species_positions.tolist(),
                'displacements': self.displacements.tolist(), 
                'hop_counts': self.hop_counts.tolist()
            })
        
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif filepath.endswith('.h5'):
            try:
                import h5py
                with h5py.File(filepath, 'w') as f:
                    for key, value in data.items():
                        f.create_dataset(key, data=value)
            except ImportError:
                raise ImportError("h5py required for HDF5 checkpoints")
        else:
            raise ValueError("Checkpoint format must be .json or .h5")
    
    @classmethod
    def load_checkpoint(cls, filepath: str) -> "SimulationState":
        """
        Load state from checkpoint file.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Restored SimulationState
        """
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif filepath.endswith('.h5'):
            try:
                import h5py
                with h5py.File(filepath, 'r') as f:
                    data = {key: f[key][()] for key in f.keys()}
            except ImportError:
                raise ImportError("h5py required for HDF5 checkpoints")
        else:
            raise ValueError("Checkpoint format must be .json or .h5")
        
        # Create state object with proper Occupation object
        from ..structure.basis import Occupation
        
        # Load occupation data and basis
        occupation_basis = data.get('occupation_basis', 'chebyshev')  # Default for backward compatibility
        occupation_obj = Occupation(data['occupations'], basis=occupation_basis)
        
        state = cls(
            occupations=occupation_obj,
            time=data['time'],
            step=data['step']
        )
        
        # Restore tracking data if present
        if 'mobile_species_positions' in data:
            state.mobile_species_positions = np.array(data['mobile_species_positions'])
            state.displacements = np.array(data['displacements'])
            state.hop_counts = np.array(data['hop_counts'])
            state.n_mobile_species = data['n_mobile_species']
        # Backward compatibility for old checkpoint files
        elif 'mobile_ion_positions' in data:
            state.mobile_species_positions = np.array(data['mobile_ion_positions'])
            state.displacements = np.array(data['displacements'])
            state.hop_counts = np.array(data['hop_counts'])
            state.n_mobile_species = data.get('n_mobile_ions', data.get('n_mobile_species', 0))
        
        return state
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current simulation statistics."""
        stats = {
            'time': self.time,
            'steps': self.step,
            'mobile_species': self.count_mobile_species(),
            'vacant_sites': len(self.get_vacant_sites())
        }
        
        if self.hop_counts is not None:
            stats.update({
                'total_hops': int(np.sum(self.hop_counts)),
                'mean_hops_per_ion': float(np.mean(self.hop_counts)),
                'max_displacement': float(np.max(np.linalg.norm(self.displacements, axis=1)))
            })
        
        return stats
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"SimulationState(time={self.time:.2e}, step={self.step}, "
            f"mobile_species={self.count_mobile_species()})"
        )
