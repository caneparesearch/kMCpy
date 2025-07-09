from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from kmcpy.model.model import BaseModel
import numpy as np

if TYPE_CHECKING:
    from kmcpy.io import InputSet
    from kmcpy.event import EventLib
    from kmcpy.external.structure import StructureKMCpy

@dataclass
class SimulationCondition:
    """
    Base class for KMC simulation conditions using dataclass for cleaner parameter management.
    """
    name: str = "DefaultSimulation"
    temperature: float = 300.0  # Kelvin
    attempt_frequency: float = 1e13  # Hz
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        if self.attempt_frequency <= 0:
            raise ValueError("Attempt frequency must be positive")
    
    def get_condition(self) -> str:
        """Return a string representation of the condition."""
        return f"{self.name}: T={self.temperature}K, f={self.attempt_frequency}Hz"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return {
            'name': self.name,
            'temperature': self.temperature,
            'v': self.attempt_frequency,  # Use 'v' to match existing KMC interface
            'random_seed': self.random_seed
        }
    
    def to_dataclass_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with dataclass field names."""
        from dataclasses import asdict
        return asdict(self)

@dataclass
class KMCSimulationCondition(SimulationCondition):
    """
    Extended simulation condition for KMC with runtime parameters.
    """
    equilibration_passes: int = 1000
    kmc_passes: int = 10000
    dimension: int = 3
    elementary_hop_distance: float = 1.0
    mobile_ion_charge: float = 1.0
    convert_to_primitive_cell: bool = False
    immutable_sites: list = field(default_factory=list)
    
    def __post_init__(self):
        """Validate KMC-specific parameters."""
        super().__post_init__()
        if self.equilibration_passes < 0:
            raise ValueError("Equilibration passes must be non-negative")
        if self.kmc_passes <= 0:
            raise ValueError("KMC passes must be positive")
        if self.dimension not in [1, 2, 3]:
            raise ValueError("Dimension must be 1, 2, or 3")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary compatible with InputSet."""
        base_dict = super().to_dict()
        base_dict.update({
            'equ_pass': self.equilibration_passes,
            'kmc_pass': self.kmc_passes,
            'dimension': self.dimension,
            'elem_hop_distance': self.elementary_hop_distance,
            'q': self.mobile_ion_charge,
            'convert_to_primitive_cell': self.convert_to_primitive_cell,
            'immutable_sites': self.immutable_sites
        })
        return base_dict

@dataclass
class SimulationConfig(KMCSimulationCondition):
    """
    Complete simulation configuration including models and state.
    """
    model: Optional[BaseModel] = None
    event_lib: Optional["EventLib"] = None
    simulation_state: Optional["SimulationState"] = None
    
    # File paths for model components
    fitting_results: Optional[str] = None
    fitting_results_site: Optional[str] = None
    lce_fname: Optional[str] = None
    lce_site_fname: Optional[str] = None
    template_structure_fname: Optional[str] = None
    event_fname: Optional[str] = None
    event_dependencies: Optional[str] = None
    
    # System configuration
    supercell_shape: list = field(default_factory=lambda: [1, 1, 1])
    initial_occ: list = field(default_factory=list)
    initial_state: Optional[str] = None  # Path to initial state JSON file
    mobile_ion_specie: str = "Li"
    
    def get_condition(self) -> str:
        """Extended condition description."""
        base_condition = super().get_condition()
        components = []
        if self.event_lib:
            components.append(f"Events: {len(self.event_lib.events) if hasattr(self.event_lib, 'events') else 'loaded'}")
        if self.simulation_state:
            components.append(f"State: {type(self.simulation_state).__name__}")
        
        if components:
            return f"{base_condition}, {', '.join(components)}"
        return base_condition
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for InputSet creation."""
        config_dict = super().to_dict()
        
        # Add file paths
        file_params = {
            'fitting_results': self.fitting_results,
            'fitting_results_site': self.fitting_results_site,
            'lce_fname': self.lce_fname,
            'lce_site_fname': self.lce_site_fname,
            'template_structure_fname': self.template_structure_fname,
            'event_fname': self.event_fname,
            'event_dependencies': self.event_dependencies,
            'event_kernel': self.event_dependencies,  # Backward compatibility
            'supercell_shape': self.supercell_shape,
            'initial_occ': self.initial_occ,
            'initial_state': self.initial_state,
            'mobile_ion_specie': self.mobile_ion_specie,
            'task': 'kmc'  # Set task for InputSet
        }
        
        # Only add non-None values
        for key, value in file_params.items():
            if value is not None:
                config_dict[key] = value
        
        return config_dict
    
    def to_inputset(self) -> "InputSet":
        """Convert to InputSet object for KMC simulation."""
        from kmcpy.io import InputSet
        import tempfile
        import json
        import os
        
        # Get the dictionary representation
        config_dict = self.to_dict()
        
        # Handle initial_state vs initial_occ precedence
        if self.initial_state:
            # If initial_state file is provided, use it directly
            config_dict['initial_state'] = self.initial_state
            config_dict.pop('initial_occ', None)
        elif self.initial_occ and isinstance(self.initial_occ, list):
            # Create temporary file for initial state
            temp_initial_state = tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False
            )
            
            initial_state_data = {
                "occupation": self.initial_occ,
                "created_by": "SimulationConfig",
                "description": f"Initial state for {self.name}"
            }
            
            json.dump(initial_state_data, temp_initial_state, indent=2)
            temp_initial_state.close()
            
            # Update config_dict to use the temporary file
            config_dict['initial_state'] = temp_initial_state.name
            
            # Store reference for cleanup
            self._temp_initial_state_file = temp_initial_state.name
            
            # Remove the direct initial_occ since we now have initial_state
            config_dict.pop('initial_occ', None)
        
        return InputSet.from_dict(config_dict)
    
    @classmethod
    def from_inputset(cls, inputset: "InputSet") -> "SimulationConfig":
        """Create SimulationConfig from InputSet."""
        params = inputset._parameters.copy()
        
        # Map InputSet parameters to SimulationConfig fields
        config_params = {
            'name': params.get('name', 'FromInputSet'),
            'temperature': params.get('temperature', 300.0),
            'attempt_frequency': params.get('v', 1e13),
            'random_seed': params.get('random_seed'),
            'equilibration_passes': params.get('equ_pass', 1000),
            'kmc_passes': params.get('kmc_pass', 10000),
            'dimension': params.get('dimension', 3),
            'elementary_hop_distance': params.get('elem_hop_distance', 1.0),
            'mobile_ion_charge': params.get('q', 1.0),
            'convert_to_primitive_cell': params.get('convert_to_primitive_cell', False),
            'immutable_sites': params.get('immutable_sites', []),
            'fitting_results': params.get('fitting_results'),
            'fitting_results_site': params.get('fitting_results_site'),
            'lce_fname': params.get('lce_fname'),
            'lce_site_fname': params.get('lce_site_fname'),
            'template_structure_fname': params.get('template_structure_fname'),
            'event_fname': params.get('event_fname'),
            'event_dependencies': params.get('event_dependencies') or params.get('event_kernel'),
            'supercell_shape': params.get('supercell_shape', [1, 1, 1]),
            'initial_occ': params.get('initial_occ', []),
            'mobile_ion_specie': params.get('mobile_ion_specie', 'Li'),
        }
        
        return cls(**{k: v for k, v in config_params.items() if v is not None})
    
    def validate(self) -> bool:
        """Validate that all required parameters are set."""
        required_files = [
            'fitting_results', 'fitting_results_site', 'lce_fname', 
            'lce_site_fname', 'template_structure_fname', 'event_fname'
        ]
        
        missing = [field for field in required_files if getattr(self, field) is None]
        if missing:
            raise ValueError(f"Missing required file parameters: {missing}")
        
        if not self.initial_occ and not self.initial_state:
            raise ValueError("Either initial_occ or initial_state must be provided")
        
        if not self.supercell_shape or len(self.supercell_shape) != 3:
            raise ValueError("Supercell shape must be a list of 3 integers")
        
        return True
    
    def copy_with_changes(self, **kwargs) -> "SimulationConfig":
        """
        Create a copy of this configuration with specified changes.
        
        Args:
            **kwargs: Parameters to change (uses dataclass field names)
            
        Returns:
            New SimulationConfig with modifications
        """
        config_dict = self.to_dataclass_dict()
        
        # Map common InputSet parameter names to dataclass field names
        param_mapping = {
            'v': 'attempt_frequency',
            'equ_pass': 'equilibration_passes',
            'kmc_pass': 'kmc_passes',
            'elem_hop_distance': 'elementary_hop_distance',
            'q': 'mobile_ion_charge'
        }
        
        # Apply mapping to kwargs
        mapped_kwargs = {}
        for key, value in kwargs.items():
            mapped_key = param_mapping.get(key, key)
            mapped_kwargs[mapped_key] = value
        
        config_dict.update(mapped_kwargs)
        return SimulationConfig(**config_dict)
    
    def run_simulation(self, label: str = None):
        """
        Run KMC simulation with this configuration.
        
        Args:
            label: Optional label for the simulation run
            
        Returns:
            Tracker object with simulation results
        """
        return run_simulation_from_config(self, label)
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during InputSet conversion."""
        if hasattr(self, '_temp_initial_state_file'):
            try:
                import os
                os.unlink(self._temp_initial_state_file)
                delattr(self, '_temp_initial_state_file')
            except (OSError, AttributeError):
                pass
    
    def __del__(self):
        """Cleanup when object is deleted."""
        self.cleanup_temp_files()

class SimulationState:
    """
    Concrete class for managing simulation state during KMC runs.
    
    This class handles all mutable state during simulation including:
    - Current occupations
    - Time and step tracking
    - Mobile ion positions and displacements
    - Hop counters
    """
    
    def __init__(self, initial_occ: list, structure: Optional["StructureKMCpy"] = None, 
                 mobile_ion_specie: str = "Li", time: float = 0.0, step: int = 0):
        """
        Initialize simulation state.
        
        Args:
            initial_occ: Initial occupation list
            structure: Structure object for mobile ion analysis
            mobile_ion_specie: Mobile ion species identifier
            time: Initial simulation time
            step: Initial step count
        """
        from copy import copy
        import numpy as np
        
        # Core state
        self.occupations = copy(initial_occ)
        self.time = time
        self.step = step
        self.mobile_ion_specie = mobile_ion_specie
        
        # Mobile ion tracking (moved from Tracker)
        if structure is not None:
            self._initialize_mobile_ion_tracking(structure, initial_occ)
        else:
            # Minimal initialization without structure
            self.mobile_ion_locations = []
            self.n_mobile_ion_specie = 0
            self.displacement = np.array([])
            self.hop_counter = np.array([])
            self.r0 = np.array([])
    
    def _initialize_mobile_ion_tracking(self, structure: "StructureKMCpy", initial_occ: list):
        """Initialize mobile ion tracking arrays."""
        import numpy as np
        
        self.structure = structure
        self.frac_coords = structure.frac_coords
        self.latt = structure.lattice
        
        # Find mobile ion sites
        self.n_mobile_ion_specie_site = len(
            [el.symbol for el in structure.species if self.mobile_ion_specie in el.symbol]
        )
        self.mobile_ion_locations = np.where(
            np.array(initial_occ[0:self.n_mobile_ion_specie_site]) == -1
        )[0]
        self.n_mobile_ion_specie = len(self.mobile_ion_locations)
        
        # Initialize tracking arrays
        self.displacement = np.zeros((self.n_mobile_ion_specie, 3))
        self.hop_counter = np.zeros(self.n_mobile_ion_specie, dtype=np.int64)
        self.r0 = self.frac_coords[self.mobile_ion_locations] @ self.latt.matrix
    
    def update_from_event(self, event, dt: float):
        """
        Update state from a KMC event.
        
        Args:
            event: KMC event object
            dt: Time increment
        """
        import numpy as np
        
        # Update time and step
        self.time += dt
        self.step += 1
        
        # Update mobile ion positions if structure is available
        if hasattr(self, 'structure') and self.structure is not None:
            self._update_mobile_ion_positions(event)
    
    def _update_mobile_ion_positions(self, event):
        """Update mobile ion positions and displacements from event."""
        import numpy as np
        from copy import copy
        
        # Get event coordinates
        mobile_ion_specie_1_coord = copy(self.frac_coords[event.mobile_ion_specie_1_index])
        mobile_ion_specie_2_coord = copy(self.frac_coords[event.mobile_ion_specie_2_index])
        
        # Get current occupations
        mobile_ion_specie_1_occ = self.occupations[event.mobile_ion_specie_1_index]
        mobile_ion_specie_2_occ = self.occupations[event.mobile_ion_specie_2_index]
        
        # Determine direction and update positions
        if mobile_ion_specie_1_occ == -1 and mobile_ion_specie_2_occ == 1:
            # Ion moves from site 1 to site 2
            direction = 1
            mobile_ion_array_index = np.where(self.mobile_ion_locations == event.mobile_ion_specie_1_index)[0][0]
            self.mobile_ion_locations[mobile_ion_array_index] = event.mobile_ion_specie_2_index
            
            # Update displacement
            displacement_vector = (mobile_ion_specie_2_coord - mobile_ion_specie_1_coord)
            displacement_vector = displacement_vector - np.round(displacement_vector)
            self.displacement[mobile_ion_array_index] += displacement_vector @ self.latt.matrix
            self.hop_counter[mobile_ion_array_index] += 1
            
        elif mobile_ion_specie_1_occ == 1 and mobile_ion_specie_2_occ == -1:
            # Ion moves from site 2 to site 1
            direction = -1
            mobile_ion_array_index = np.where(self.mobile_ion_locations == event.mobile_ion_specie_2_index)[0][0]
            self.mobile_ion_locations[mobile_ion_array_index] = event.mobile_ion_specie_1_index
            
            # Update displacement
            displacement_vector = (mobile_ion_specie_1_coord - mobile_ion_specie_2_coord)
            displacement_vector = displacement_vector - np.round(displacement_vector)
            self.displacement[mobile_ion_array_index] += displacement_vector @ self.latt.matrix
            self.hop_counter[mobile_ion_array_index] += 1
        
        # Update occupations
        self.occupations[event.mobile_ion_specie_1_index] *= -1
        self.occupations[event.mobile_ion_specie_2_index] *= -1
    
    def advance_time(self, dt: float):
        """Advance simulation time."""
        self.time += dt
        self.step += 1
    
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

class LCESimulationState(SimulationState):
    def __init__(self, occupations:list=[], probabilities:list=[], barriers:list=[],  ekra:list=[], 
                 esite_diff:list=[], time=0.0, step=0):
        super().__init__(occupations, probabilities, barriers, time, step)
        self.ekra = ekra
        self.esite_diff = esite_diff

    def update(self, update_message:dict, indices:list):
        """
        Update the occupations and barriers based on the update message.
        
        :param update_message: Dictionary containing 'ekra', 'esite_diff', 'direction', 'barrier', 'probability'.
        :param indices: List of indices to update in the occupations.
        """
        ekra = update_message.get('ekra')
        esite_diff = update_message.get('esite_diff')
        barrier = update_message.get('barrier')
        probability = update_message.get('probability')

        for idx in indices:
            self.occupations[idx] *= -1
            self.barriers[idx] = barrier
            self.probabilities[idx] = probability
            self.ekra[idx] = ekra
            self.esite_diff[idx] = esite_diff

# Convenience functions for common use cases

def create_nasicon_config(
    name: str = "NASICON_Simulation",
    temperature: float = 573.0,
    supercell_shape: list = None,
    initial_occ: list = None,
    data_dir: str = "example",
    **kwargs
) -> SimulationConfig:
    """
    Create a standard NASICON simulation configuration.
    
    Args:
        name: Simulation name
        temperature: Temperature in Kelvin
        supercell_shape: Supercell dimensions [nx, ny, nz]
        initial_occ: Initial occupation list
        data_dir: Directory containing input files
        **kwargs: Additional parameters to override defaults
    
    Returns:
        SimulationConfig: Configured simulation setup
    """
    if supercell_shape is None:
        supercell_shape = [2, 2, 2]
    if initial_occ is None:
        initial_occ = [1, -1, 1, -1, 1, -1, 1, -1]
    
    default_config = {
        'name': name,
        'temperature': temperature,
        'attempt_frequency': 1e13,
        'equilibration_passes': 1000,
        'kmc_passes': 10000,
        'dimension': 3,
        'elementary_hop_distance': 2.5,
        'mobile_ion_charge': 1.0,
        'mobile_ion_specie': 'Na',
        'supercell_shape': supercell_shape,
        'initial_occ': initial_occ,
        'fitting_results': f"{data_dir}/fitting_results.json",
        'fitting_results_site': f"{data_dir}/fitting_results_site.json",
        'lce_fname': f"{data_dir}/lce.json",
        'lce_site_fname': f"{data_dir}/lce_site.json",
        'template_structure_fname': f"{data_dir}/0th_reference_local_env.cif",
        'event_fname': f"{data_dir}/events.json",
        'event_dependencies': f"{data_dir}/event_kernal.csv"
    }
    
    # Override with user-provided kwargs
    default_config.update(kwargs)
    
    return SimulationConfig(**default_config)

def create_temperature_series(
    base_config: SimulationConfig,
    temperatures: list,
    name_template: str = "{base_name}_T_{temp}K"
) -> list[SimulationConfig]:
    """
    Create a series of configurations with different temperatures.
    
    Args:
        base_config: Base configuration to modify
        temperatures: List of temperatures to test
        name_template: Template for naming each configuration
    
    Returns:
        List of SimulationConfig objects
    """
    configs = []
    
    for temp in temperatures:
        new_name = name_template.format(
            base_name=base_config.name, 
            temp=temp
        )
        config = base_config.copy_with_changes(temperature=temp, name=new_name)
        configs.append(config)
    
    return configs

def run_simulation_from_config(config: SimulationConfig, label: str = None):
    """
    Convenience function to run a KMC simulation from a configuration.
    
    Args:
        config: SimulationConfig object
        label: Optional label for the simulation run
    
    Returns:
        Tracker object with simulation results
    """
    from kmcpy.kmc import KMC
    
    # Validate configuration
    config.validate()
    
    # Convert to InputSet and create KMC
    inputset = config.to_inputset()
    kmc = KMC.from_inputset(inputset)
    
    # Run simulation
    if label is None:
        label = config.name
    
    return kmc.run(inputset, label=label)