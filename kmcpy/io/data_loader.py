#!/usr/bin/env python
"""
This module provides data loading and management functionality for NEB (Nudged Elastic Band) 
calculations and structure databases used in Local Cluster Expansion model fitting.
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass
import logging
from kmcpy.models.local_env import LocalEnvironment

if TYPE_CHECKING:
    from kmcpy.models import LocalClusterExpansion

logger = logging.getLogger(__name__)
logging.getLogger('numba').setLevel(logging.WARNING)

@dataclass
class NEBEntry:
    """
    Data class to hold structure information for fitting.
    
    Attributes:
        local_env (LocalEnvironment): Structure model of NEB calculation
        property_value (float): Property value for this structure (e.g. E_KRA or E_site)
        occupation (Optional[List[int]]): Occupation vector for the structure, defaults to None
        correlation (Optional[List[float]]): Correlation vector for the structure, defaults to None
        metadata (Dict): Additional metadata for the structure, defaults to None
    """
    local_env: LocalEnvironment
    property_value: float
    occupation: Optional[List[int]] = None
    correlation: Optional[List[float]] = None
    metadata: Optional[Dict] = None

    def compute_occ_corr(self, model: 'LocalClusterExpansion') -> None:
        """
        Compute occupation and correlation vectors for the structure.
        Args:
            model (LocalClusterExpansion): Local Cluster Expansion model instance
        """
        from kmcpy.models.local_cluster_expansion import _calc_corr

        self.occupation = model.get_occ_from_structure(self.structure)
        logger.warning(f"{self.structure}: {self.occupation}")
        self.correlation = model.get_corr_from_structure(self.structure)
        logger.info(f"{self.structure}: {self.correlation}")

class DataLoader:
    """
    Base class for data loaders.
    
    This class can be extended to implement specific data loading functionality.
    """
    def __init__(self):
        pass
    
    def add(self):
        """Add data from the source."""
        raise NotImplementedError("Subclasses should implement this method.")
    
    def __len__(self) -> int:
        """Return the number of loaded entries."""
        return 0

class NEBDataLoader(DataLoader):
    """
    Data loader for handling databases of structures from NEB calculations.
    
    This class manages loading structures, energies, and computing correlation
    matrices for Local Cluster Expansion model fitting.
    """
    
    def __init__(self):
        """
        Initialize the NEBDataLoader.
        """
        self.neb_entries: List[NEBEntry] = []

    def add(self, neb_entry: NEBEntry, model: 'LocalClusterExpansion') -> None:
        """
        Add a NEBEntry to the loader.
        
        Args:
            neb_entry (NEBEntry): NEBEntry object to add
            model (LocalClusterExpansion): Local Cluster Expansion model instance
        """
        neb_entry.compute_occ_corr(model)
        self.neb_entries.append(neb_entry)
        self.model_name = model.name
        logger.debug(f"Added NEB entry: {neb_entry.structure} with property: {neb_entry.property_value}")
    
    def get_correlation_matrix(self) -> np.ndarray:
        """
        Get the correlation matrix for all structures.
        
        Returns:
            np.ndarray: Correlation matrix [n_structures, n_correlations]
        """
        return np.array([neb_entry.correlation for neb_entry in self.neb_entries])
    
    def get_occupation_matrix(self) -> np.ndarray:
        """
        Get the occupation matrix for all structures.
        
        Returns:
            np.ndarray: Occupation matrix [n_structures, n_sites]
        """
        return np.array([neb_entry.occupation for neb_entry in self.neb_entries])
    
    def get_properties(self) -> np.ndarray:
        """
        Get the properties for all structures.
        
        Returns:
            np.ndarray: Property vector [n_structures]
        """
        return np.array([neb_entry.property_value for neb_entry in self.neb_entries])
    
    def to_json(self, 
                output_dir: str = ".", 
                prefix: str = "ekra") -> None:
        """
        Save training data to JSON file.
        
        Args:
            output_dir (str): Output directory
            prefix (str): Prefix for output files, defaults to "ekra"
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data to save
        data = {
            "correlation_matrix": self.get_correlation_matrix().tolist(),
            "occupation_matrix": self.get_occupation_matrix().tolist(),
            "properties": self.get_properties().tolist(),
            "metadata": [
            neb_entry.metadata if neb_entry.metadata is not None else {}
            for neb_entry in self.neb_entries
            ]
        }
        output_file = os.path.join(output_dir, f"{prefix}.json")
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved all data to {output_file}")
    
    
    def __str__(self) -> str:
        """
        String summary of loaded data.
        """
        if not self.neb_entries:
            return "NEBDataLoader: n_structures=0"
        
        properties = self.get_properties()
        n_structures = len(self.neb_entries)
        property_min = min(properties)
        property_max = max(properties)
        property_mean = np.mean(properties)
        property_std = np.std(properties)
        has_occupations = any(entry.occupation is not None for entry in self.neb_entries)
        has_correlations = any(entry.correlation is not None for entry in self.neb_entries)

        summary = (
            f"NEBDataLoader:\n"
            f"  n_structures: {n_structures}\n"
            f"  property_min: {property_min:.4f}\n"
            f"  property_max: {property_max:.4f}\n"
            f"  property_mean: {property_mean:.4f}\n"
            f"  property_std: {property_std:.4f}\n"
            f"  has_occupations: {has_occupations}\n"
            f"  has_correlations: {has_correlations}"
        )
        return summary
            
    def __len__(self) -> int:
        """Return number of loaded structures."""
        return len(self.neb_entries)
    
    def __repr__(self) -> str:
        """String representation of the data loader."""
        return f"NEBDataLoader(n_structures={len(self.neb_entries)})"
