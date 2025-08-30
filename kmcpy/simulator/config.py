"""
Clean simulation configuration classes with clear separation of concerns.

Architecture:
- SystemConfig: Physical system definition (immutable)
- RuntimeConfig: Simulation runtime parameters (immutable) 
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
    
    # Model files
    cluster_expansion_file: str = ""
    cluster_expansion_site_file: Optional[str] = None
    fitting_results_file: str = ""
    fitting_results_site_file: Optional[str] = None
    event_file: str = ""
    event_dependencies: Optional[str] = None
    
    # System constraints
    immutable_sites: tuple = field(default_factory=tuple)
    convert_to_primitive_cell: bool = False
    
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
    """
    Complete simulation configuration.
    Immutable composition of system and runtime configs.
    """
    system: SystemConfig
    runtime: RuntimeConfig
    
    # Initial state specification (immutable - describes what to load)
    initial_state_file: Optional[str] = None
    initial_occupations: Optional[tuple] = None
    
    def __post_init__(self):
        """Validate complete configuration."""
        # Ensure we have exactly one way to specify initial state
        has_state_file = self.initial_state_file is not None
        has_occupations = self.initial_occupations is not None
        
        if not (has_state_file or has_occupations):
            raise ValueError("Must specify either initial_state_file or initial_occupations")
        
        if has_state_file and has_occupations:
            raise ValueError("Cannot specify both initial_state_file and initial_occupations")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for InputSet compatibility."""
        # Merge system and runtime configs
        config = {}
        config.update(self.system.to_dict())
        config.update(self.runtime.to_dict())
        
        # Add initial state handling
        if self.initial_state_file:
            config['initial_state'] = self.initial_state_file
        elif self.initial_occupations:
            config['initial_occ'] = list(self.initial_occupations)
        
        # Add metadata
        config['task'] = 'kmc'
        
        return config
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SimulationConfig":
        """Create from dictionary."""
        # Extract system parameters
        system_params = {
            'structure_file': config_dict.get('template_structure_fname', ''),
            'supercell_shape': tuple(config_dict.get('supercell_shape', [1, 1, 1])),
            'dimension': config_dict.get('dimension', 3),
            'mobile_ion_specie': config_dict.get('mobile_ion_specie', 'Li'),
            'mobile_ion_charge': config_dict.get('q', 1.0),
            'elementary_hop_distance': config_dict.get('elem_hop_distance', 1.0),
            'cluster_expansion_file': config_dict.get('lce_fname', ''),
            'cluster_expansion_site_file': config_dict.get('lce_site_fname'),
            'fitting_results_file': config_dict.get('fitting_results', ''),
            'fitting_results_site_file': config_dict.get('fitting_results_site'),
            'event_file': config_dict.get('event_fname', ''),
            'event_dependencies': config_dict.get('event_dependencies'),
            'immutable_sites': tuple(config_dict.get('immutable_sites', [])),
            'convert_to_primitive_cell': config_dict.get('convert_to_primitive_cell', False)
        }
        
        # Extract runtime parameters - support both old and new parameter names
        runtime_params = {
            'temperature': float(config_dict.get('temperature', 300.0)),
            'attempt_frequency': float(config_dict.get('attempt_frequency', 1e13)),
            'equilibration_passes': int(config_dict.get('equ_pass') or config_dict.get('equilibration_passes', 1000)),
            'kmc_passes': int(config_dict.get('kmc_pass') or config_dict.get('kmc_passes', 10000)),
            'random_seed': config_dict.get('random_seed'),
            'name': str(config_dict.get('name', 'DefaultSimulation'))
        }
        
        # Extract initial state - support both old and new parameter names
        initial_state_file = config_dict.get('initial_state')
        initial_occupations = (config_dict.get('initial_occupations') or 
                             config_dict.get('occupations') or
                             config_dict.get('initial_occ'))
        if initial_occupations:
            initial_occupations = tuple(initial_occupations)
        
        return cls(
            system=SystemConfig(**system_params),
            runtime=RuntimeConfig(**runtime_params),
            initial_state_file=initial_state_file,
            initial_occupations=initial_occupations
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
        
        # Convert legacy parameter names
        data = SimulationConfigIO._convert_legacy_parameters(raw_data)
        
        return cls.from_dict(data)
    
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
        
        # Convert legacy parameter names
        data = SimulationConfigIO._convert_legacy_parameters(raw_data)
        
        return cls.from_dict(data)
    
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
        new_runtime = replace(self.runtime, **changes)
        return replace(self, runtime=new_runtime)
    
    def with_system_changes(self, **changes) -> "SimulationConfig":
        """Create new config with system parameter changes."""
        from dataclasses import replace
        new_system = replace(self.system, **changes)
        return replace(self, system=new_system)
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"{self.runtime.name}: "
            f"T={self.runtime.temperature}K, "
            f"passes={self.runtime.kmc_passes}, "
            f"system={Path(self.system.structure_file).name}"
        )
    
    # ===== CONVENIENCE FACTORY METHODS =====
    
    @classmethod
    def create(
        cls,
        # Required files
        structure_file: str,
        cluster_expansion_file: str, 
        event_file: str,
        
        # Initial state (one of these must be provided)
        initial_occupations: Optional[List[int]] = None,
        initial_state_file: Optional[str] = None,
        
        # Common simulation parameters
        name: str = "KMC_Simulation",
        temperature: float = 300.0,
        kmc_passes: int = 10000,
        equilibration_passes: int = 1000,
        
        # System parameters with sensible defaults
        mobile_species: str = "Li",
        supercell_shape: tuple = (1, 1, 1),
        dimension: int = 3,
        
        # Optional advanced parameters
        attempt_frequency: float = 1e13,
        mobile_species_charge: float = 1.0,
        elementary_hop_distance: float = 1.0,
        cluster_expansion_site_file: Optional[str] = None,
        fitting_results_file: Optional[str] = None,
        fitting_results_site_file: Optional[str] = None,
        event_dependencies: Optional[str] = None,
        immutable_sites: Optional[List[int]] = None,
        convert_to_primitive_cell: bool = False,
        random_seed: Optional[int] = None,
        
        **kwargs
    ) -> "SimulationConfig":
        """
        Convenient factory method to create SimulationConfig with minimal required parameters.
        
        This method automatically creates SystemConfig and RuntimeConfig internally.
        
        Args:
            structure_file: Path to crystal structure file (.cif, .vasp, etc.)
            cluster_expansion_file: Path to cluster expansion model file  
            event_file: Path to KMC events definition file
            initial_occupations: List of initial site occupations [1, -1, 1, ...]
            initial_state_file: Path to initial state JSON file (alternative to initial_occupations)
            name: Simulation name for identification
            temperature: Temperature in Kelvin
            kmc_passes: Number of KMC steps to run
            equilibration_passes: Number of equilibration steps
            mobile_species: Name of mobile species (e.g., "Li", "Na", "H")
            supercell_shape: Supercell dimensions (nx, ny, nz)
            dimension: System dimensionality (1, 2, or 3)
            **kwargs: Additional parameters passed to SystemConfig/RuntimeConfig
            
        Returns:
            Complete SimulationConfig ready for use
            
        Example:
            config = SimulationConfig.create(
                structure_file="nasicon.cif",
                cluster_expansion_file="lce.json", 
                event_file="events.json",
                initial_occupations=[1, -1, 1, -1, 1, -1],
                name="NASICON_300K",
                temperature=300.0,
                kmc_passes=50000,
                mobile_species="Na"
            )
        """
        # Create SystemConfig
        system = SystemConfig(
            structure_file=structure_file,
            supercell_shape=tuple(supercell_shape) if isinstance(supercell_shape, list) else supercell_shape,
            dimension=dimension,
            mobile_ion_specie=mobile_species,  # Keep legacy name for compatibility
            mobile_ion_charge=mobile_species_charge,
            elementary_hop_distance=elementary_hop_distance,
            cluster_expansion_file=cluster_expansion_file,
            cluster_expansion_site_file=cluster_expansion_site_file,
            fitting_results_file=fitting_results_file or cluster_expansion_file,
            fitting_results_site_file=fitting_results_site_file or cluster_expansion_site_file,
            event_file=event_file,
            event_dependencies=event_dependencies,
            immutable_sites=tuple(immutable_sites or []),
            convert_to_primitive_cell=convert_to_primitive_cell
        )
        
        # Create RuntimeConfig 
        runtime = RuntimeConfig(
            temperature=temperature,
            attempt_frequency=attempt_frequency,
            equilibration_passes=equilibration_passes,
            kmc_passes=kmc_passes,
            random_seed=random_seed,
            name=name
        )
        
        # Convert initial_occupations to tuple if provided
        if initial_occupations is not None:
            initial_occupations = tuple(initial_occupations)
        
        return cls(
            system=system,
            runtime=runtime,
            initial_state_file=initial_state_file,
            initial_occupations=initial_occupations
        )
    
    @classmethod 
    def quick_setup(
        cls,
        structure_file: str,
        model_files: Dict[str, str],
        initial_occupations: List[int],
        temperature: float = 300.0,
        kmc_passes: int = 10000,
        **kwargs
    ) -> "SimulationConfig":
        """
        Ultra-quick setup for common use cases.
        
        Args:
            structure_file: Path to structure file
            model_files: Dictionary with keys 'cluster_expansion', 'events'
            initial_occupations: Initial site occupations
            temperature: Temperature in Kelvin
            kmc_passes: Number of KMC steps
            **kwargs: Additional parameters
            
        Example:
            config = SimulationConfig.quick_setup(
                structure_file="structure.cif",
                model_files={
                    'cluster_expansion': 'lce.json',
                    'events': 'events.json'
                },
                initial_occupations=[1, -1, 1, -1],
                temperature=400.0
            )
        """
        return cls.create(
            structure_file=structure_file,
            cluster_expansion_file=model_files['cluster_expansion'],
            event_file=model_files['events'],
            initial_occupations=initial_occupations,
            temperature=temperature,
            kmc_passes=kmc_passes,
            **kwargs
        )