from typing import Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from copy import copy

if TYPE_CHECKING:
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
            'attempt_frequency': self.attempt_frequency,
            'random_seed': self.random_seed
        }
    
    def to_dataclass_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with dataclass field names."""
        from dataclasses import asdict
        return asdict(self)

# SimulationConfig has been moved to config.py
# This import ensures backward compatibility
from .config import SimulationConfig

