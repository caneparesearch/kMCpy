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
    def _convert_legacy_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal: Convert legacy parameter names to current names.
        
        Args:
            params: Raw parameter dictionary
            
        Returns:
            Dictionary with updated parameter names
        """
        # Create a copy to avoid modifying the original
        converted = params.copy()
        
        # Legacy parameter name mappings
        legacy_mappings = {
            'event_kernel': 'event_dependencies',  # Old name -> new name
            # Add more mappings as needed
        }
        
        # Apply conversions
        for old_name, new_name in legacy_mappings.items():
            if old_name in converted and new_name not in converted:
                converted[new_name] = converted[old_name]
                logger.warning(f"Converted legacy parameter '{old_name}' to '{new_name}'")
                # Keep old name for now to avoid breaking things
        
        return converted
    
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
