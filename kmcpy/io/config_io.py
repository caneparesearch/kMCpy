"""
Internal I/O utilities for SimulationConfig.

This module provides pure I/O operations for loading/saving simulation configurations.
These classes are internal implementation details and should not be used directly by users.
All user interactions should go through SimulationConfig methods.
"""

from typing import Dict, Any, Optional
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SimulationConfigIO:
    """
    INTERNAL ONLY: Pure I/O utility for SimulationConfig.
    
    This class handles file operations without any business logic.
    Users should never interact with this class directly - use SimulationConfig methods instead.
    """
    
    @staticmethod
    def _load_json(filepath: str) -> Dict[str, Any]:
        """
        Internal: Load raw dictionary from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Dictionary with raw configuration data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            JSONDecodeError: If file is not valid JSON
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            logger.debug(f"Loaded JSON configuration from {filepath}")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filepath}: {e}")
    
    @staticmethod  
    def _load_yaml(filepath: str) -> Dict[str, Any]:
        """
        Internal: Load raw dictionary from YAML file.
        
        Args:
            filepath: Path to YAML file
            
        Returns:
            Dictionary with raw configuration data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ImportError: If PyYAML not installed
            YAMLError: If file is not valid YAML
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for YAML support. Install with: pip install PyYAML")
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            logger.debug(f"Loaded YAML configuration from {filepath}")
            return data
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {filepath}: {e}")
    
    @staticmethod
    def _load_yaml_section(filepath: str, section: str, task_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Internal: Load specific section from YAML file.
        
        Supports both flat sections and registry-style sections with type field.
        
        Args:
            filepath: Path to YAML file
            section: Section name to load ('kmc', 'model', 'generate_event')
            task_type: Optional task type for registry-style sections
            
        Returns:
            Dictionary with section configuration data
            
        Raises:
            ValueError: If section or task_type not found
        """
        yaml_data = SimulationConfigIO._load_yaml(filepath)
        
        if section not in yaml_data:
            available = list(yaml_data.keys())
            raise ValueError(f"Section '{section}' not found in {filepath}. Available: {available}")
        
        section_data = yaml_data[section]
        
        # Handle registry-style sections with type field
        if isinstance(section_data, dict) and 'type' in section_data:
            if task_type is None:
                task_type = section_data['type']
            
            if task_type not in section_data:
                available_types = [k for k in section_data.keys() if k != 'type']
                raise ValueError(f"Task type '{task_type}' not found in section '{section}'. Available: {available_types}")
            
            # Extract parameters from the specific task type
            parameters = section_data[task_type].copy()
            logger.debug(f"Loaded section '{section}' with task type '{task_type}' from {filepath}")
        else:
            # Handle flat sections (like simple kmc section)
            parameters = section_data.copy()
            logger.debug(f"Loaded flat section '{section}' from {filepath}")
        
        # Add task field based on section for compatibility
        if section == "model":
            parameters["task"] = "lce"  # Default model task
        else:
            parameters["task"] = section
        
        return parameters
    
    @staticmethod
    def _save_json(data: Dict[str, Any], filepath: str, indent: int = 2) -> None:
        """
        Internal: Save dictionary to JSON file.
        
        Args:
            data: Configuration dictionary to save
            filepath: Output file path
            indent: JSON indentation (default: 2)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=indent, default=SimulationConfigIO._json_serializer)
            logger.debug(f"Saved JSON configuration to {filepath}")
        except TypeError as e:
            raise ValueError(f"Cannot serialize configuration to JSON: {e}")
    
    @staticmethod
    def _save_yaml(data: Dict[str, Any], filepath: str) -> None:
        """
        Internal: Save dictionary to YAML file.
        
        Args:
            data: Configuration dictionary to save  
            filepath: Output file path
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for YAML support. Install with: pip install PyYAML")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w') as f:
                yaml.safe_dump(data, f, default_flow_style=False, indent=2)
            logger.debug(f"Saved YAML configuration to {filepath}")
        except yaml.YAMLError as e:
            raise ValueError(f"Cannot serialize configuration to YAML: {e}")
    
    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for numpy types and other objects."""
        import numpy as np
        
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    @staticmethod
    def _detect_file_format(filepath: str) -> str:
        """
        Internal: Auto-detect file format from extension.
        
        Args:
            filepath: File path to analyze
            
        Returns:
            Format string ('json', 'yaml', or 'unknown')
        """
        suffix = Path(filepath).suffix.lower()
        
        if suffix == '.json':
            return 'json'
        elif suffix in ['.yaml', '.yml']:
            return 'yaml'
        else:
            return 'unknown'

    @staticmethod
    def load_occupation_data(initial_state_file: str, supercell_shape: list, select_sites: list) -> list:
        """
        Load occupation data from file, replacing deprecated InputSet.load_occ functionality.
        
        Args:
            initial_state_file: Path to initial state file (JSON format)
            supercell_shape: Supercell dimensions [x, y, z]
            select_sites: Indices of sites to include in KMC (excluding immutable sites)
            
        Returns:
            List of occupation values in Chebyshev basis (1 and -1)
            
        Raises:
            FileNotFoundError: If initial state file doesn't exist
            ValueError: If occupation data is incompatible with supercell shape
        """
        import numpy as np
        import json
        
        with open(initial_state_file, "r") as f:
            # Read the occupation from json
            occupation_raw_data = np.array(json.load(f)["occupation"])

            # Check if the occupation is compatible with the shape
            if len(occupation_raw_data) % (supercell_shape[0] * supercell_shape[1] * supercell_shape[2]) != 0:
                logger.error(
                    f"The length of occupation data {len(occupation_raw_data)} is incompatible with the supercell shape"
                )
                raise ValueError(
                    f"The length of occupation data {len(occupation_raw_data)} is incompatible with the supercell shape"
                )

            # Calculate total sites
            site_nums = int(len(occupation_raw_data) / (supercell_shape[0] * supercell_shape[1] * supercell_shape[2]))

            # Reshape and select sites
            convert_to_dimension = (site_nums, supercell_shape[0], supercell_shape[1], supercell_shape[2])
            occupation = occupation_raw_data.reshape(convert_to_dimension)[
                select_sites
            ].flatten("C")

            # Convert to Chebyshev basis (replace 0 with -1)
            occupation_chebyshev = np.where(occupation == 0, -1, occupation)

            logger.debug(f"Selected sites are {select_sites}")
            logger.debug(f"Converting the occupation raw data to dimension: {convert_to_dimension}")
            logger.debug(f"Occupation (Chebyshev basis after removing immutable sites): {occupation_chebyshev}")

            return occupation_chebyshev.tolist()

    @staticmethod
    def load_events_from_file(event_file: str) -> 'EventLib':
        """
        Load events from file, centralizing event loading logic.
        
        Args:
            event_file: Path to event file (JSON format)
            
        Returns:
            EventLib instance with loaded events
            
        Raises:
            FileNotFoundError: If event file doesn't exist
            ValueError: If event file format is invalid
        """
        from kmcpy.event import EventLib, Event
        import json
        
        event_lib = EventLib()
        with open(event_file, "rb") as fhandle:
            events_dict = json.load(fhandle)
        
        for event_dict in events_dict:
            event = Event.from_dict(event_dict)
            event_lib.add_event(event)
        
        event_lib.generate_event_dependencies()
        return event_lib

    @staticmethod
    def load_simulation_components(config: 'SimulationConfig') -> tuple:
        """
        Load all simulation components from config, centralizing all manual loading logic.
        
        This method replaces all manual loading logic in KMC.from_config() and provides
        a single entry point for loading structure, model, events, and simulation state.
        
        Args:
            config: SimulationConfig object with all parameters
            
        Returns:
            tuple: (structure, model, event_lib, simulation_state)
            
        Raises:
            FileNotFoundError: If required files don't exist
            ValueError: If configuration is invalid
        """
        import numpy as np
        from kmcpy.external.structure import StructureKMCpy
        from kmcpy.models.composite_lce_model import CompositeLCEModel
        from kmcpy.simulator.state import SimulationState
        
        # Load structure directly from config
        structure = StructureKMCpy.from_cif(
            config.structure_file, 
            primitive=config.convert_to_primitive_cell
        )
        
        # Calculate select_sites based on original structure BEFORE removing immutable sites
        select_sites_for_occupation = []
        if config.initial_state_file:
            immutable_sites = config.immutable_sites or []
            for index, site in enumerate(structure.sites):
                if site.specie.symbol not in immutable_sites:
                    select_sites_for_occupation.append(index)
            logger.debug(f"Select sites for occupation loading: {select_sites_for_occupation}")
        
        # Apply transformations AFTER calculating select_sites
        if config.immutable_sites:
            structure.remove_species(config.immutable_sites)
        
        if config.supercell_shape:
            supercell_shape_matrix = np.diag(config.supercell_shape)
            structure.make_supercell(supercell_shape_matrix)
        
        # Load composite model using its centralized from_json method
        model = CompositeLCEModel.from_json(
            lce_fname=config.cluster_expansion_file,
            fitting_results=config.fitting_results_file,
            lce_site_fname=config.cluster_expansion_site_file,
            fitting_results_site=config.fitting_results_site_file
        )
        
        # Load events using centralized method
        event_lib = SimulationConfigIO.load_events_from_file(config.event_file)
        
        # Handle initial occupation from config using centralized loading
        initial_occ = None
        if config.initial_occupations:
            initial_occ = list(config.initial_occupations)
        elif config.initial_state_file:
            # Use centralized occupation loading method
            initial_occ = SimulationConfigIO.load_occupation_data(
                initial_state_file=config.initial_state_file,
                supercell_shape=list(config.supercell_shape),
                select_sites=select_sites_for_occupation
            )
        
        # Always create a SimulationState (even if we have to use empty occupations)
        if initial_occ is not None:
            simulation_state = SimulationState(occupations=initial_occ)
        else:
            # Create with empty occupations - this will be populated during structure loading
            simulation_state = SimulationState(occupations=[])
        
        return structure, model, event_lib, simulation_state
