

class SimulationState:
    """
    Concrete class for managing mutable simulation state during KMC runs.
    
    This class handles core mutable state during simulation:
    - Current occupations (evolving from initial_occ)
    - Time and step tracking
    
    Note: Mobile ion tracking should be handled by the Tracker class.
    For checkpointing, save/load SimulationState using to()/from_file() methods.
    At restart, ignore SimulationConfig's initial_* fields and load from checkpoint.
    """
    
    def __init__(self, initial_occ: list, time: float = 0.0, step: int = 0):
        """
        Initialize simulation state.
        
        Args:
            initial_occ: Initial occupation list
            time: Initial simulation time
            step: Initial step count
        """
        from copy import copy
        
        # Core state only
        self.occupations = copy(initial_occ)
        self.time = time
        self.step = step
    
    def update_from_event(self, event, dt: float):
        """
        Update state from a KMC event.
        
        Args:
            event: KMC event object
            dt: Time increment
        """
        # Update time and step
        self.time += dt
        self.step += 1
        
        # Update occupations
        self.occupations[event.mobile_ion_indices[0]] *= -1
        self.occupations[event.mobile_ion_indices[1]] *= -1
    
    def get_current_occupations(self) -> list:
        """Get current occupation state."""
        return self.occupations.copy()
    
    def get_mobile_ion_info(self) -> dict:
        """Get mobile ion tracking information."""
        return {
            'locations': self.mobile_ion_locations.copy() if hasattr(self, 'mobile_ion_locations') else [],
            'displacement': self.displacement.copy() if hasattr(self, 'displacement') else [],
            'hop_counter': self.hop_counter.copy() if hasattr(self, 'hop_counter') else [],
            'n_mobile_ions': self.n_mobile_ion_specie if hasattr(self, 'n_mobile_ion_specie') else 0
        }
    
    def copy(self) -> "SimulationState":
        """Create a copy of this simulation state."""
        from copy import deepcopy
        
        # Create new state with same basic parameters
        new_state = SimulationState(
            initial_occ=self.occupations.copy(),
            time=self.time,
            step=self.step
        )
        
        # Copy mobile ion tracking data if available
        if hasattr(self, 'mobile_ion_locations'):
            new_state.mobile_ion_locations = deepcopy(self.mobile_ion_locations)
        if hasattr(self, 'displacement'):
            new_state.displacement = deepcopy(self.displacement)
        if hasattr(self, 'hop_counter'):
            new_state.hop_counter = deepcopy(self.hop_counter)
        if hasattr(self, 'r0'):
            new_state.r0 = deepcopy(self.r0)
        if hasattr(self, 'n_mobile_ion_specie'):
            new_state.n_mobile_ion_specie = self.n_mobile_ion_specie
        if hasattr(self, 'n_mobile_ion_specie_site'):
            new_state.n_mobile_ion_specie_site = self.n_mobile_ion_specie_site
        if hasattr(self, 'frac_coords'):
            new_state.frac_coords = deepcopy(self.frac_coords)
        if hasattr(self, 'latt'):
            new_state.latt = self.latt
        
        return new_state

    def to(self, filename: str) -> None:
        """
        Save the state to a file.
        
        :param filename: The name of the file to save the state.
        """
        import json
        import numpy as np
        
        # Prepare data for serialization
        state_data = {
            'occupations': self.occupations,
            'time': self.time,
            'step': self.step,
            'mobile_ion_specie': self.mobile_ion_specie
        }
        
        # Add mobile ion tracking data if available
        if hasattr(self, 'mobile_ion_locations'):
            state_data.update({
                'mobile_ion_locations': self.mobile_ion_locations.tolist(),
                'displacement': self.displacement.tolist(),
                'hop_counter': self.hop_counter.tolist(),
                'n_mobile_ion_specie': self.n_mobile_ion_specie
            })
        
        if filename.endswith('.json'):
            with open(filename, 'w') as f:
                json.dump(state_data, f, indent=4)
        elif filename.endswith('.h5'):
            try:
                import h5py
                with h5py.File(filename, 'w') as f:
                    for key, value in state_data.items():
                        f.create_dataset(key, data=value)
            except ImportError:
                raise ImportError("h5py is required for HDF5 file support. Install with: pip install h5py")
        else:
            raise ValueError("Unsupported file format. Use .json or .h5")
        
    @classmethod
    def from_file(cls, filename: str) -> "SimulationState":
        """
        Load the state from a file.
        
        :param filename: The name of the file to load the state from.
        :return: An instance of SimulationState.
        """
        import json
        import numpy as np
        
        if filename.endswith('.json'):
            with open(filename, 'r') as f:
                data = json.load(f)
        elif filename.endswith('.h5'):
            try:
                import h5py
                with h5py.File(filename, 'r') as f:
                    data = {key: f[key][()] for key in f.keys()}
            except ImportError:
                raise ImportError("h5py is required for HDF5 file support. Install with: pip install h5py")
        else:
            raise ValueError("Unsupported file format. Use .json or .h5")
        
        # Create instance with basic parameters
        instance = cls(
            initial_occ=data['occupations'],
            time=data.get('time', 0.0),
            step=data.get('step', 0),
            mobile_ion_specie=data.get('mobile_ion_specie', 'Li')
        )
        
        # Restore mobile ion tracking data if available
        if 'mobile_ion_locations' in data:
            instance.mobile_ion_locations = np.array(data['mobile_ion_locations'])
            instance.displacement = np.array(data['displacement'])
            instance.hop_counter = np.array(data['hop_counter'])
            instance.n_mobile_ion_specie = data['n_mobile_ion_specie']
        
        return instance
