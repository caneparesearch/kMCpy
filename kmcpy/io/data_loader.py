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
from kmcpy.structure.local_lattice_structure import LocalLatticeStructure
from kmcpy.structure.basis import Occupation
if TYPE_CHECKING:
    from kmcpy.models import LocalClusterExpansion

logger = logging.getLogger(__name__)
logging.getLogger('numba').setLevel(logging.WARNING)

@dataclass
class NEBEntry:
    """
    Data class to hold structure information for fitting.
    
    Attributes:
        local_lattice_structure (LocalLatticeStructure): Structure model of NEB calculation
        property_value (float): Property value for this structure (e.g. E_KRA or E_site)
        occupation (Occupation): Occupation vector for the structure, defaults to None
        correlation (Optional[List[float]]): Correlation vector for the structure, defaults to None
        metadata (Dict): Additional metadata for the structure, defaults to None
    """
    local_lattice_structure: LocalLatticeStructure
    property_value: float
    occupation: Occupation = None
    correlation: Optional[List[float]] = None
    metadata: Optional[Dict] = None

    def compute_occ_corr(self, model: 'LocalClusterExpansion') -> None:
        """
        Compute occupation and correlation vectors for the structure.
        
        Args:
            model (LocalClusterExpansion): Local Cluster Expansion model instance
        """
        try:
            # Get occupation vector from the LocalLatticeStructure
            if hasattr(self.local_lattice_structure, 'template_structure'):
                structure_for_occ = self.local_lattice_structure.template_structure
            else:
                structure_for_occ = getattr(self.local_lattice_structure, 'structure', None)
                if structure_for_occ is None:
                    raise ValueError("Cannot find structure data in LocalLatticeStructure")
            
            self.occupation = self.local_lattice_structure.get_occ_from_structure(structure_for_occ)
            self.correlation = model.get_corr_from_structure(structure_for_occ)
            logger.debug(f"Computed vectors: occ_len={len(self.occupation)}, corr_len={len(self.correlation)}")
            
        except Exception as e:
            logger.error(f"Failed to compute occupation and correlation vectors: {e}")
            raise

class DataLoader:
    """
    Base class for data loaders.
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
        """Initialize the NEBDataLoader."""
        self.neb_entries: List[NEBEntry] = []

    def _validate_structure_consistency(self, new_entry: NEBEntry, rtol: float = 1e-3, atol: float = 1e-3) -> bool:
        """
        Validate that a new LocalLatticeStructure is consistent with existing entries.
        
        Args:
            new_entry: New entry to validate
            rtol: Relative tolerance for distance matrix comparison
            atol: Absolute tolerance for distance matrix comparison
            
        Returns:
            bool: True if consistent, False otherwise
        """
        if not self.neb_entries or new_entry.local_lattice_structure is None:
            return True
            
        try:
            ref_structure = self.neb_entries[0].local_lattice_structure
            new_structure = new_entry.local_lattice_structure
            
            # Check if both have structure attribute (Molecule)
            if not hasattr(ref_structure, 'structure') or not hasattr(new_structure, 'structure'):
                return True  # Skip validation if structure not available
                
            ref_molecule = ref_structure.structure
            new_molecule = new_structure.structure
            
            # Check site count
            if len(ref_molecule) != len(new_molecule):
                logger.warning(f"Site count mismatch: {len(ref_molecule)} vs {len(new_molecule)}")
                return False
            
            # Check species
            ref_species = sorted([str(site.specie) for site in ref_molecule])
            new_species = sorted([str(site.specie) for site in new_molecule])
            if ref_species != new_species:
                logger.warning(f"Species mismatch: {ref_species} vs {new_species}")
                return False
            
            # Check distance matrices
            ref_dist_matrix = ref_molecule.distance_matrix
            new_dist_matrix = new_molecule.distance_matrix
            if not np.allclose(ref_dist_matrix, new_dist_matrix, rtol=rtol, atol=atol):
                logger.warning("Distance matrices do not match within tolerance")
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"Structure validation failed: {e}")
            return True  # Don't fail if validation itself fails

    def add(self, neb_entry: NEBEntry, model: 'LocalClusterExpansion', validate_structure: bool = True) -> None:
        """
        Add a NEBEntry to the loader with validation.
        
        Args:
            neb_entry: NEBEntry object to add
            model: Local Cluster Expansion model instance
            validate_structure: Whether to validate LocalLatticeStructure consistency
        """
        if not isinstance(neb_entry, NEBEntry):
            raise ValueError("Entry must be a NEBEntry instance")
            
        if neb_entry.local_lattice_structure is None:
            raise ValueError("NEBEntry must have a valid LocalLatticeStructure")
            
        # Validate structure consistency
        if validate_structure and not self._validate_structure_consistency(neb_entry):
            raise ValueError("LocalLatticeStructure is inconsistent with existing entries")
            
        # Compute vectors
        neb_entry.compute_occ_corr(model)
        
        # Check vector consistency
        if self.neb_entries:
            if len(neb_entry.occupation) != len(self.neb_entries[0].occupation):
                raise ValueError("Occupation vector length mismatch")
            if len(neb_entry.correlation) != len(self.neb_entries[0].correlation):
                raise ValueError("Correlation vector length mismatch")
        
        self.neb_entries.append(neb_entry)
        self.model_name = model.name
        logger.info(f"Added NEB entry with property value: {neb_entry.property_value:.6f}")
    
    def get_correlation_matrix(self) -> np.ndarray:
        """Get the correlation matrix for all structures."""
        if not self.neb_entries:
            raise ValueError("No entries available")
        return np.array([entry.correlation for entry in self.neb_entries])
    
    def get_occupation_matrix(self) -> np.ndarray:
        """Get the occupation matrix for all structures."""
        if not self.neb_entries:
            raise ValueError("No entries available")
        return np.array([entry.occupation for entry in self.neb_entries])
    
    def get_properties(self) -> np.ndarray:
        """Get the properties for all structures."""
        if not self.neb_entries:
            raise ValueError("No entries available")
        return np.array([entry.property_value for entry in self.neb_entries])
        
    def to_json(self, output_dir: str = ".", prefix: str = "ekra") -> str:
        """
        Save training data to JSON file.
        
        Args:
            output_dir: Output directory
            prefix: Prefix for output files
            
        Returns:
            str: Path to saved file
        """
        if not self.neb_entries:
            raise ValueError("No entries to save")
            
        os.makedirs(output_dir, exist_ok=True)
        
        data = {
            "correlation_matrix": self.get_correlation_matrix().tolist(),
            "occupation_matrix": self.get_occupation_matrix().tolist(),
            "properties": self.get_properties().tolist(),
            "metadata": [entry.metadata or {} for entry in self.neb_entries],
            "n_structures": len(self.neb_entries),
            "model_name": getattr(self, 'model_name', 'unknown')
        }
        
        output_file = os.path.join(output_dir, f"{prefix}.json")
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved {len(self.neb_entries)} entries to {output_file}")
        return output_file
    
    def __str__(self) -> str:
        """String summary of loaded data."""
        if not self.neb_entries:
            return "NEBDataLoader: 0 structures"
        
        properties = self.get_properties()
        n_structures = len(self.neb_entries)
        
        return (
            f"NEBDataLoader: {n_structures} structures\n"
            f"  Property range: [{np.min(properties):.4f}, {np.max(properties):.4f}]\n"
            f"  Correlation matrix: {self.get_correlation_matrix().shape}\n"
            f"  Occupation matrix: {self.get_occupation_matrix().shape}"
        )
            
    def __len__(self) -> int:
        """Return number of loaded structures."""
        return len(self.neb_entries)
    
    def __repr__(self) -> str:
        """String representation of the data loader."""
        return f"NEBDataLoader(n_structures={len(self.neb_entries)})"
