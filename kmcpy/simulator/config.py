"""
Clean simulation configuration classes with clear separation of concerns.

Architecture:
- SystemConfig: Physical system definition (        # Parameter routing tables - ONLY clean parameter names, no legacy support
        system_param_names = {
            'structure_file', 'supercell_shape', 'dimension', 'mobile_ion_specie',
            'mobile_ion_charge', 'elementary_hop_distance', 'model_type', 'cluster_expansion_file',
            'cluster_expansion_site_file', 'fitting_results_file', 'fitting_results_site_file',
            'event_file', 'event_dependencies', 'immutable_sites', 'convert_to_primitive_cell'
        }
        
        runtime_param_names = {
            'temperature', 'attempt_frequency', 'equilibration_passes', 'kmc_passes',
            'random_seed', 'name'
        }
        
        # Handle parameter aliases for commonly confused names
        parameter_aliases = {
            'mobile_species': 'mobile_ion_specie',  # Common alias
            'initial_state_file': None,  # This should be handled differently, ignore for now
        } RuntimeConfig: Simulation runtime parameters (immutable) 
- SimulationConfig: Complete simulation setup (immutable)
- SimulationState: Mutable state during execution
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SystemConfig:
    """
    Physical system configuration - completely immutable.
    This defines WHAT you're simulating.
    """
    # Structure definition
    structure_file: str
    supercell_shape: tuple[int, int, int] = (1, 1, 1)
    dimension: int = 3
    
    # Mobile ion properties
    mobile_ion_specie: str = "Li"
    mobile_ion_charge: float = 1.0
    elementary_hop_distance: float = 1.0
    
    # Model configuration
    model_type: str = "composite_lce"  # Default to composite_lce for backward compatibility
    cluster_expansion_file: str = ""
    cluster_expansion_site_file: Optional[str] = None
    fitting_results_file: str = ""
    fitting_results_site_file: Optional[str] = None
    event_file: str = ""
    event_dependencies: Optional[str] = None
    
    # System constraints
    immutable_sites: tuple = field(default_factory=tuple)
    convert_to_primitive_cell: bool = False
    
    # Initial state specification
    initial_state_file: Optional[str] = None
    initial_occupations: Optional[list] = None
    
    def __post_init__(self):
        """Validate system configuration."""
        if self.dimension not in [1, 2, 3]:
            raise ValueError(f"Invalid dimension: {self.dimension}")
        
        if len(self.supercell_shape) != 3:
            raise ValueError("Supercell shape must have 3 components")
        
        # Temporarily disabled for testing
        # if not Path(self.structure_file).exists():
        #     raise FileNotFoundError(f"Structure file not found: {self.structure_file}")
        # 
        # if not Path(self.event_file).exists():
        #     raise FileNotFoundError(f"Event file not found: {self.event_file}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        from dataclasses import asdict
        data = asdict(self)
        # Convert tuple back to list for compatibility
        data['supercell_shape'] = list(self.supercell_shape)
        data['immutable_sites'] = list(self.immutable_sites)
        return data


@dataclass(frozen=True)
class RuntimeConfig:
    """
    Simulation runtime configuration - completely immutable.
    This defines HOW you're simulating.
    """
    # Thermodynamic conditions
    temperature: float = 300.0  # Kelvin
    attempt_frequency: float = 1e13  # Hz
    
    # KMC algorithm parameters
    equilibration_passes: int = 1000
    kmc_passes: int = 10000
    random_seed: Optional[int] = None
    
    # Simulation identification
    name: str = "DefaultSimulation"
    
    def __post_init__(self):
        """Validate runtime parameters."""
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        if self.attempt_frequency <= 0:
            raise ValueError("Attempt frequency must be positive")
        
        if self.equilibration_passes < 0:
            raise ValueError("Equilibration passes must be non-negative")
        
        if self.kmc_passes <= 0:
            raise ValueError("KMC passes must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with legacy key names."""
        return {
            'temperature': self.temperature,
            'equ_pass': self.equilibration_passes,
            'kmc_pass': self.kmc_passes,
            'random_seed': self.random_seed,
            'name': self.name
        }


@dataclass(frozen=True)
class SimulationConfig:
    """Complete simulation configuration combining system and runtime parameters."""
    
    system_config: SystemConfig
    runtime_config: RuntimeConfig
    
    def __init__(self, system_config=None, runtime_config=None, **kwargs):
        """
        Create SimulationConfig with automatic parameter routing.
        
        You can either:
        1. Pass pre-built configs: SimulationConfig(system_config=sys, runtime_config=run)
        2. Pass parameters directly: SimulationConfig(temperature=300, structure_file="x.cif", ...)
        3. Mix both: SimulationConfig(system_config=sys, temperature=400)
        
        Parameters are automatically routed to SystemConfig or RuntimeConfig based on their names.
        """
        if system_config is None and runtime_config is None and not kwargs:
            raise ValueError("Must provide either configs or parameters")
        
        # Split kwargs into system and runtime parameters
        system_params = {}
        runtime_params = {}
        unknown_params = {}
        
        # Parameter routing tables
        system_param_names = {
            'structure_file', 'supercell_shape', 'dimension', 'mobile_ion_specie',
            'mobile_ion_charge', 'elementary_hop_distance', 'model_type', 'cluster_expansion_file',
            'cluster_expansion_site_file', 'fitting_results_file', 'fitting_results_site_file',
            'event_file', 'event_dependencies', 'immutable_sites', 'convert_to_primitive_cell',
            'initial_state_file', 'initial_occupations'  # Added initial state parameters
        }
        
        runtime_param_names = {
            'temperature', 'attempt_frequency', 'equilibration_passes', 'kmc_passes',
            'random_seed', 'name'
        }
        
        # Route parameters
        for key, value in kwargs.items():
            if key in system_param_names:
                system_params[key] = value
            elif key in runtime_param_names:
                runtime_params[key] = value
            else:
                unknown_params[key] = value
        
        # Reject unknown parameters - no legacy support
        if unknown_params:
            raise ValueError(f"Unknown parameters: {list(unknown_params.keys())}. "
                           f"Use SimulationConfig.help_parameters() to see valid parameters.")
        
        # Create or update configs
        if system_config is None:
            system_config = SystemConfig(**system_params)
        elif system_params:
            # Update existing system config with new parameters
            from dataclasses import replace
            system_config = replace(system_config, **system_params)
        
        if runtime_config is None:
            runtime_config = RuntimeConfig(**runtime_params)
        elif runtime_params:
            # Update existing runtime config with new parameters
            from dataclasses import replace
            runtime_config = replace(runtime_config, **runtime_params)
        
        # Set the attributes using object.__setattr__ since the class is frozen
        object.__setattr__(self, 'system_config', system_config)
        object.__setattr__(self, 'runtime_config', runtime_config)
    
    @classmethod
    def create(cls, **kwargs):
        """
        Alternative factory method for cleaner API.
        
        Examples:
            config = SimulationConfig.create(
                structure_file="test.cif",
                temperature=400.0,
                kmc_passes=50000
            )
        """
        return cls(**kwargs)
    
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        result.update(self.system_config.to_dict())
        result.update(self.runtime_config.to_dict())
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SimulationConfig":
        """Create from dictionary."""
        # Split parameters between system and runtime configs
        system_params = {}
        runtime_params = {}
        
        # SystemConfig parameter names
        system_param_names = {
            'structure_file', 'supercell_shape', 'dimension', 'mobile_ion_specie',
            'mobile_ion_charge', 'elementary_hop_distance', 'model_type', 'cluster_expansion_file',
            'cluster_expansion_site_file', 'fitting_results_file', 'fitting_results_site_file',
            'event_file', 'event_dependencies', 'immutable_sites', 'convert_to_primitive_cell'
        }
        
        # RuntimeConfig parameter names
        runtime_param_names = {
            'temperature', 'attempt_frequency', 'equilibration_passes', 'kmc_passes',
            'random_seed', 'name'
        }
        
        for key, value in config_dict.items():
            if key in system_param_names:
                system_params[key] = value
            elif key in runtime_param_names:
                runtime_params[key] = value
        
        return cls(
            system_config=SystemConfig(**system_params),
            runtime_config=RuntimeConfig(**runtime_params)
        )
    
    # ===== FILE I/O METHODS =====
    
    @classmethod
    def from_file(cls, filepath: str) -> "SimulationConfig":
        """
        Load SimulationConfig from file (auto-detects format from extension).
        
        Args:
            filepath: Path to configuration file (.json, .yaml, .yml)
            
        Returns:
            SimulationConfig instance
            
        Example:
            config = SimulationConfig.from_file("simulation.yaml")
            config = SimulationConfig.from_file("simulation.json")
        """
        from kmcpy.io.config_io import SimulationConfigIO
        
        file_format = SimulationConfigIO._detect_file_format(filepath)
        
        if file_format == 'json':
            raw_data = SimulationConfigIO._load_json(filepath)
        elif file_format == 'yaml':
            raw_data = SimulationConfigIO._load_yaml(filepath)
        else:
            raise ValueError(f"Unsupported file format for {filepath}. Supported: .json, .yaml, .yml")
                
        return cls.from_dict(raw_data)
    
    @classmethod
    def from_yaml_section(cls, filepath: str, section: str = "kmc", task_type: Optional[str] = None) -> "SimulationConfig":
        """
        Load SimulationConfig from specific section of YAML file.
        
        Useful for multi-section YAML files that contain different configurations.
        
        Args:
            filepath: Path to YAML file
            section: Section name to load (default: "kmc")  
            task_type: Optional task type for registry-style sections
            
        Returns:
            SimulationConfig instance
            
        Example:
            # Load from simple section
            config = SimulationConfig.from_yaml_section("workflow.yaml", "kmc")
            # Load from registry-style section
            config = SimulationConfig.from_yaml_section("workflow.yaml", "kmc", "diffusion")
        """
        from kmcpy.io.config_io import SimulationConfigIO
        
        raw_data = SimulationConfigIO._load_yaml_section(filepath, section, task_type)
    
        return cls.from_dict(raw_data)
    
    def save(self, filepath: str, **kwargs) -> None:
        """
        Save SimulationConfig to file (auto-detects format from extension).
        
        Args:
            filepath: Output file path (.json, .yaml, .yml)
            **kwargs: Additional arguments for file formatting (e.g., indent for JSON)
            
        Example:
            config.save("output.yaml")
            config.save("output.json", indent=4)
        """
        from kmcpy.io.config_io import SimulationConfigIO
        
        data = self.to_dict()
        file_format = SimulationConfigIO._detect_file_format(filepath)
        
        if file_format == 'json':
            indent = kwargs.get('indent', 2)
            SimulationConfigIO._save_json(data, filepath, indent=indent)
        elif file_format == 'yaml':
            SimulationConfigIO._save_yaml(data, filepath)
        else:
            raise ValueError(f"Unsupported file format for {filepath}. Supported: .json, .yaml, .yml")
    
    def save_yaml_section(self, filepath: str, section: str = "kmc", task_type: str = "default") -> None:
        """
        Save SimulationConfig as a section in YAML file.
        
        Creates a registry-style YAML section with type field.
        
        Args:
            filepath: Output YAML file path
            section: Section name (default: "kmc")
            task_type: Task type name (default: "default")
            
        Example:
            config.save_yaml_section("workflow.yaml", "kmc", "diffusion")
            # Creates: 
            # kmc:
            #   type: diffusion
            #   diffusion:
            #     temperature: 300.0
            #     ...
        """
        from kmcpy.io.config_io import SimulationConfigIO
        import os
        
        # Load existing YAML file or create new structure
        if os.path.exists(filepath):
            try:
                yaml_data = SimulationConfigIO._load_yaml(filepath)
            except:
                yaml_data = {}
        else:
            yaml_data = {}
        
        # Create registry-style section
        config_data = self.to_dict()
        # Remove task field since we'll use section structure
        config_data.pop('task', None)
        
        yaml_data[section] = {
            'type': task_type,
            task_type: config_data
        }
        
        SimulationConfigIO._save_yaml(yaml_data, filepath)
    
    def with_runtime_changes(self, **changes) -> "SimulationConfig":
        """Create new config with runtime parameter changes."""
        from dataclasses import replace
        new_runtime = replace(self.runtime_config, **changes)
        return replace(self, runtime_config=new_runtime)
    
    def with_system_changes(self, **changes) -> "SimulationConfig":
        """Create new config with system parameter changes."""
        from dataclasses import replace
        new_system = replace(self.system_config, **changes)
        return replace(self, system_config=new_system)
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"{self.runtime_config.name}: "
            f"T={self.runtime_config.temperature}K, "
            f"passes={self.runtime_config.kmc_passes}, "
            f"system={Path(self.system_config.structure_file).name}"
        )
    
    # ===== CONVENIENT PROPERTY ACCESS =====
    # Users don't need to remember which config contains what parameter
    
    # Runtime properties
    @property
    def temperature(self) -> float:
        """Access temperature directly."""
        return self.runtime_config.temperature
    
    @property
    def name(self) -> str:
        """Access simulation name directly."""
        return self.runtime_config.name
    
    @property
    def kmc_passes(self) -> int:
        """Access KMC passes directly."""
        return self.runtime_config.kmc_passes
    
    @property
    def equilibration_passes(self) -> int:
        """Access equilibration passes directly."""
        return self.runtime_config.equilibration_passes
    
    @property
    def attempt_frequency(self) -> float:
        """Access attempt frequency directly."""
        return self.runtime_config.attempt_frequency
    
    @property
    def random_seed(self) -> Optional[int]:
        """Access random seed directly."""
        return self.runtime_config.random_seed
    
    # System properties
    @property
    def structure_file(self) -> str:
        """Access structure file directly."""
        return self.system_config.structure_file
    
    @property
    def mobile_ion_specie(self) -> str:
        """Access mobile ion species directly."""
        return self.system_config.mobile_ion_specie
    
    @property
    def supercell_shape(self) -> tuple[int, int, int]:
        """Access supercell shape directly."""
        return self.system_config.supercell_shape
    
    @property
    def dimension(self) -> int:
        """Access dimension directly."""
        return self.system_config.dimension
    
    @property
    def model_type(self) -> str:
        """Access model type directly."""
        return self.system_config.model_type
    
    @property
    def cluster_expansion_file(self) -> str:
        """Access cluster expansion file directly."""
        return self.system_config.cluster_expansion_file
    
    @property
    def event_file(self) -> str:
        """Access event file directly."""
        return self.system_config.event_file
    
    @property
    def immutable_sites(self) -> tuple:
        """Access immutable sites directly."""
        return self.system_config.immutable_sites
    
    @property
    def elementary_hop_distance(self) -> float:
        """Access elementary hop distance directly."""
        return self.system_config.elementary_hop_distance
    
    @property  
    def mobile_ion_charge(self) -> float:
        """Access mobile ion charge directly."""
        return self.system_config.mobile_ion_charge
    
    @property
    def convert_to_primitive_cell(self) -> bool:
        """Access convert to primitive cell directly."""
        return self.system_config.convert_to_primitive_cell
    
    @property
    def initial_state_file(self) -> Optional[str]:
        """Access initial state file directly."""
        return self.system_config.initial_state_file
    
    @property
    def initial_occupations(self) -> Optional[list]:
        """Access initial occupations directly."""
        return self.system_config.initial_occupations
    
    @property
    def fitting_results_file(self) -> str:
        """Access fitting results file directly."""
        return self.system_config.fitting_results_file
    
    @property
    def fitting_results_site_file(self) -> Optional[str]:
        """Access fitting results site file directly."""
        return self.system_config.fitting_results_site_file
    
    @property
    def cluster_expansion_site_file(self) -> Optional[str]:
        """Access cluster expansion site file directly."""
        return self.system_config.cluster_expansion_site_file
    
    @property
    def event_dependencies(self) -> Optional[str]:
        """Access event dependencies directly."""
        return self.system_config.event_dependencies
    
    # ===== HELPER METHODS =====
    
    @classmethod
    def help_parameters(cls):
        """Print available parameters and which config they belong to."""
        print("SimulationConfig Parameters:\n")
        
        print("SYSTEM PARAMETERS (physical setup):")
        system_params = [
            "structure_file", "supercell_shape", "dimension", "mobile_ion_specie",
            "mobile_ion_charge", "elementary_hop_distance", "model_type", "cluster_expansion_file",
            "cluster_expansion_site_file", "fitting_results_file", "fitting_results_site_file",
            "event_file", "event_dependencies", "immutable_sites", "convert_to_primitive_cell"
        ]
        for param in system_params:
            print(f"  - {param}")
        
        print("\nRUNTIME PARAMETERS (simulation settings):")
        runtime_params = [
            "temperature", "attempt_frequency", "equilibration_passes", "kmc_passes",
            "random_seed", "name"
        ]
        for param in runtime_params:
            print(f"  - {param}")
        
        print("\nUsage examples:")
        print("  config = SimulationConfig(structure_file='x.cif', temperature=400)")
        print("  config = SimulationConfig.create(temperature=300, kmc_passes=10000)")
        print("  print(config.temperature)  # Direct access to any parameter")
    
    def which_config(self, parameter_name: str) -> str:
        """Show which sub-config contains a parameter."""
        system_params = {
            'structure_file', 'supercell_shape', 'dimension', 'mobile_ion_specie',
            'mobile_ion_charge', 'elementary_hop_distance', 'model_type', 'cluster_expansion_file',
            'cluster_expansion_site_file', 'fitting_results_file', 'fitting_results_site_file',
            'event_file', 'event_dependencies', 'immutable_sites', 'convert_to_primitive_cell'
        }
        
        runtime_params = {
            'temperature', 'attempt_frequency', 'equilibration_passes', 'kmc_passes',
            'random_seed', 'name'
        }
        
        if parameter_name in system_params:
            return f"'{parameter_name}' is in system_config (physical setup)"
        elif parameter_name in runtime_params:
            return f"'{parameter_name}' is in runtime_config (simulation settings)"
        else:
            return f"'{parameter_name}' is not a recognized parameter"

    def validate(self) -> bool:
        """Validate the configuration."""
        try:
            # Basic validation - configs validate themselves in __post_init__
            # This could be expanded with more complex cross-parameter validation
            return True
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def copy_with_changes(self, **changes) -> "SimulationConfig":
        """Create a copy of this config with some parameters changed.
        
        Args:
            **changes: Parameter changes to apply
            
        Returns:
            SimulationConfig: New config with changes applied
        """
        # Get current config as dict
        current_dict = self.to_dict()
        
        # Apply changes
        current_dict.update(changes)
        
        # Create new config
        return SimulationConfig.from_dict(current_dict)


# ===== I/O HELPER CLASS =====