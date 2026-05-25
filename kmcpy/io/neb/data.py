#!/usr/bin/env python
"""
This module provides data loading and management functionality for NEB (Nudged Elastic Band) 
calculations and structure databases used in Local Cluster Expansion model fitting.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import numpy as np
from typing import Any, List, Dict, Optional, Sequence, TYPE_CHECKING
from dataclasses import dataclass
import logging
from pymatgen.core import Structure
from kmcpy.structure.local_lattice_structure import LocalLatticeStructure
from kmcpy.structure.basis import Occupation
if TYPE_CHECKING:
    from kmcpy.models import LocalClusterExpansion

logger = logging.getLogger(__name__)
logging.getLogger('numba').setLevel(logging.WARNING)

StructureSource = Structure | str | os.PathLike[str]


def _load_structure_source(
    structure: StructureSource,
    metadata: Optional[Dict[str, Any]] = None,
    **from_file_kwargs,
) -> tuple[Structure, Dict[str, Any]]:
    entry_metadata = dict(metadata or {})
    if isinstance(structure, Structure):
        if from_file_kwargs:
            raise ValueError(
                "Structure.from_file keyword arguments are only valid when "
                "structure is a file path."
            )
        return structure, entry_metadata

    if isinstance(structure, (str, os.PathLike)):
        path = Path(structure)
        loaded_structure = Structure.from_file(str(path), **from_file_kwargs)
        entry_metadata.setdefault("structure_file", str(path))
        return loaded_structure, entry_metadata

    raise TypeError(
        "structure must be a pymatgen Structure or a path readable by "
        "pymatgen.core.Structure.from_file"
    )


@dataclass
class NEBEntry:
    """
    Represents an entry for Nudged Elastic Band (NEB) calculations, storing structural and property data.
    Attributes:
        structure (Structure): Structure associated with this entry.
        property_value (float): The property value (e.g., energy) associated with this entry.
        occupation (Occupation, optional): The occupation vector for the structure. Defaults to None.
        correlation (List[float], optional): The correlation vector for the structure. Defaults to None.
        metadata (Dict, optional): Additional metadata for the entry. Defaults to None.
    """
    structure: Structure
    property_value: float
    occupation: Occupation = None
    correlation: Optional[List[float]] = None
    metadata: Optional[Dict] = None

    @classmethod
    def from_structure(
        cls,
        structure: StructureSource,
        property_value: float,
        metadata: Optional[Dict[str, Any]] = None,
        **from_file_kwargs,
    ) -> "NEBEntry":
        """
        Create an entry from a structure object or structure file.

        Args:
            structure: Structure compatible with the reference local lattice, or
                a path readable by ``pymatgen.core.Structure.from_file``.
            property_value: Target value, typically an NEB barrier.
            metadata: Optional metadata stored with the entry.
            **from_file_kwargs: Additional keyword arguments passed to pymatgen
                when ``structure`` is a file path.

        Returns:
            NEBEntry: Entry ready to be added to a loader.
        """
        structure_obj, entry_metadata = _load_structure_source(
            structure,
            metadata=metadata,
            **from_file_kwargs,
        )
        return cls(
            structure=structure_obj,
            property_value=float(property_value),
            metadata=entry_metadata,
        )

    def compute_occ_corr(
        self,
        model: 'LocalClusterExpansion',
        reference_local_lattice_structure: Optional[LocalLatticeStructure] = None,
        exclude_species: Optional[Sequence[str]] = None,
        tol: float = 1e-2,
        angle_tol: float = 5,
    ) -> None:
        """
        Compute occupation and correlation vectors for the structure.
        
        Args:
            model (LocalClusterExpansion): Local Cluster Expansion model instance
            reference_local_lattice_structure: Reference local lattice used to
                map structures into occupation vectors. If omitted, the model
                must carry ``local_lattice_structure`` from ``build``.
            exclude_species: Removed legacy argument; use site_mapping fixed sites. If omitted, the reference local lattice's
                exclusion list is used when available.
            tol: Structure matching tolerance.
            angle_tol: Structure matching angle tolerance.
        """
        try:
            reference = (
                reference_local_lattice_structure
                or getattr(model, "local_lattice_structure", None)
            )
            self.occupation, self.correlation = model.get_occ_corr_from_structure(
                self.structure,
                reference_local_lattice_structure=reference,
                exclude_species=exclude_species,
                tol=tol,
                angle_tol=angle_tol,
            )
            logger.debug(f"Computed vectors: occ_len={len(self.occupation)}, corr_len={len(self.correlation)}")
            
        except Exception as e:
            logger.error(f"Failed to compute occupation and correlation vectors: {e}")
            raise


class NEBDataLoader:
    """
    A data loader class for managing and validating databases of structures from NEB (Nudged Elastic Band) calculations,
    intended for Local Cluster Expansion model fitting. This class provides methods to add NEBEntry objects with
    consistency checks, compute occupation and correlation matrices, retrieve property values, and serialize the dataset
    to JSON format.
    Attributes:
        neb_entries (List[NEBEntry]): List of NEBEntry objects loaded into the data loader.
        model_name (str): Name of the Local Cluster Expansion model associated with the data.
    """
    
    def __init__(
        self,
        model: Optional['LocalClusterExpansion'] = None,
        reference_local_lattice_structure: Optional[LocalLatticeStructure] = None,
        exclude_species: Optional[Sequence[str]] = None,
    ):
        """Initialize the NEBDataLoader."""
        if exclude_species is not None:
            raise ValueError(
                "exclude_species is no longer supported; encode fixed sites in "
                "site_mapping with a single allowed species."
            )
        self.neb_entries: List[NEBEntry] = []
        self.model = model
        self.reference_local_lattice_structure = reference_local_lattice_structure
        self.exclude_species = None
        if model is not None:
            self.model_name = getattr(model, "name", "unknown")

    def _resolve_model(
        self,
        model: Optional['LocalClusterExpansion'] = None,
    ) -> 'LocalClusterExpansion':
        resolved_model = model or self.model
        if resolved_model is None:
            raise ValueError(
                "A LocalClusterExpansion model is required. Pass model=... to "
                "NEBDataLoader(...) or to add()."
            )
        return resolved_model

    def add(
        self,
        neb_entry: NEBEntry,
        model: Optional['LocalClusterExpansion'] = None,
        reference_local_lattice_structure: Optional[LocalLatticeStructure] = None,
        exclude_species: Optional[Sequence[str]] = None,
        tol: float = 1e-2,
        angle_tol: float = 5,
    ) -> None:
        """
        Add a NEBEntry to the loader with validation.
        
        Args:
            neb_entry: NEBEntry object to add
            model: Local Cluster Expansion model instance
            reference_local_lattice_structure: Reference local lattice used to
                compute occupation and correlation vectors.
            exclude_species: Removed legacy argument; use site_mapping fixed sites.
            tol: Structure matching tolerance.
            angle_tol: Structure matching angle tolerance.
        """
        if exclude_species is not None:
            raise ValueError(
                "exclude_species is no longer supported; encode fixed sites in "
                "site_mapping with a single allowed species."
            )
        if not isinstance(neb_entry, NEBEntry):
            raise ValueError("Entry must be a NEBEntry instance")

        resolved_model = self._resolve_model(model)
        reference = (
            reference_local_lattice_structure
            or self.reference_local_lattice_structure
        )

        # Compute vectors
        neb_entry.compute_occ_corr(
            resolved_model,
            reference_local_lattice_structure=reference,
            exclude_species=(
                exclude_species
                if exclude_species is not None
                else self.exclude_species
            ),
            tol=tol,
            angle_tol=angle_tol,
        )
        
        # Check vector consistency
        if self.neb_entries:
            if len(neb_entry.occupation) != len(self.neb_entries[0].occupation):
                raise ValueError("Occupation vector length mismatch")
            if len(neb_entry.correlation) != len(self.neb_entries[0].correlation):
                raise ValueError("Correlation vector length mismatch")
        
        self.neb_entries.append(neb_entry)
        self.model = resolved_model
        self.model_name = getattr(resolved_model, "name", "unknown")
        logger.info(f"Added NEB entry with property value: {neb_entry.property_value:.6f}")

    def add_structure(
        self,
        structure: StructureSource,
        property_value: float,
        model: Optional['LocalClusterExpansion'] = None,
        metadata: Optional[Dict[str, Any]] = None,
        reference_local_lattice_structure: Optional[LocalLatticeStructure] = None,
        exclude_species: Optional[Sequence[str]] = None,
        tol: float = 1e-2,
        angle_tol: float = 5,
        **from_file_kwargs,
    ) -> NEBEntry:
        """
        Add a structure object or structure file and target value directly.

        Args:
            structure: Structure compatible with the reference local lattice, or
                a path readable by ``pymatgen.core.Structure.from_file``.
            property_value: Target value, typically an NEB barrier.
            model: Local Cluster Expansion model instance.
            metadata: Optional metadata stored with the entry.
            reference_local_lattice_structure: Reference local lattice used to
                compute occupation and correlation vectors.
            exclude_species: Removed legacy argument; use site_mapping fixed sites.
            tol: Structure matching tolerance.
            angle_tol: Structure matching angle tolerance.
            **from_file_kwargs: Additional keyword arguments passed to pymatgen
                when ``structure`` is a file path.

        Returns:
            NEBEntry: Added entry.
        """
        entry = NEBEntry.from_structure(
            structure=structure,
            property_value=property_value,
            metadata=metadata,
            **from_file_kwargs,
        )
        self.add(
            entry,
            model=model,
            reference_local_lattice_structure=reference_local_lattice_structure,
            exclude_species=exclude_species,
            tol=tol,
            angle_tol=angle_tol,
        )
        return entry

    def add_structures(
        self,
        structures: Sequence[StructureSource],
        property_values: Sequence[float],
        model: Optional['LocalClusterExpansion'] = None,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
        reference_local_lattice_structure: Optional[LocalLatticeStructure] = None,
        exclude_species: Optional[Sequence[str]] = None,
        tol: float = 1e-2,
        angle_tol: float = 5,
        **from_file_kwargs,
    ) -> None:
        """
        Add multiple structure objects or structure files and target values.

        Args:
            structures: Structure objects or files readable by pymatgen.
            property_values: Target values matching ``structures``.
            model: Local Cluster Expansion model instance.
            metadata: Optional metadata entries matching ``structures``.
            reference_local_lattice_structure: Reference local lattice used to
                compute occupation and correlation vectors.
            exclude_species: Removed legacy argument; use site_mapping fixed sites.
            tol: Structure matching tolerance.
            angle_tol: Structure matching angle tolerance.
            **from_file_kwargs: Additional keyword arguments passed to pymatgen
                for path inputs.
        """
        if len(structures) != len(property_values):
            raise ValueError("structures and property_values must have the same length")
        if metadata is not None and len(metadata) != len(structures):
            raise ValueError("metadata must have the same length as structures")

        for index, (structure, property_value) in enumerate(
            zip(structures, property_values)
        ):
            entry_metadata = metadata[index] if metadata is not None else None
            self.add_structure(
                structure=structure,
                property_value=property_value,
                model=model,
                metadata=entry_metadata,
                reference_local_lattice_structure=reference_local_lattice_structure,
                exclude_species=exclude_species,
                tol=tol,
                angle_tol=angle_tol,
                **from_file_kwargs,
            )

    @classmethod
    def from_structures(
        cls,
        structures: Sequence[StructureSource],
        property_values: Sequence[float],
        model: 'LocalClusterExpansion',
        reference_local_lattice_structure: Optional[LocalLatticeStructure] = None,
        exclude_species: Optional[Sequence[str]] = None,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
        tol: float = 1e-2,
        angle_tol: float = 5,
        **from_file_kwargs,
    ) -> "NEBDataLoader":
        """
        Build a loader from structure objects or structure files.

        Args:
            structures: Structure objects or files readable by pymatgen.
            property_values: Target values matching ``structures``.
            model: Local Cluster Expansion model instance.
            reference_local_lattice_structure: Reference local lattice used to
                compute occupation and correlation vectors.
            exclude_species: Removed legacy argument; use site_mapping fixed sites.
            metadata: Optional metadata entries matching ``structures``.
            tol: Structure matching tolerance.
            angle_tol: Structure matching angle tolerance.
            **from_file_kwargs: Additional keyword arguments passed to pymatgen
                for path inputs.

        Returns:
            NEBDataLoader: Loader containing computed fitting data.
        """
        loader = cls(
            model=model,
            reference_local_lattice_structure=reference_local_lattice_structure,
            exclude_species=exclude_species,
        )
        loader.add_structures(
            structures=structures,
            property_values=property_values,
            metadata=metadata,
            exclude_species=exclude_species,
            tol=tol,
            angle_tol=angle_tol,
            **from_file_kwargs,
        )
        return loader
    
    def get_correlation_matrix(self) -> np.ndarray:
        """Get the correlation matrix for all structures."""
        if not self.neb_entries:
            raise ValueError("No entries available")
        return np.array([entry.correlation for entry in self.neb_entries])
    
    def get_occupation_matrix(self) -> np.ndarray:
        """Get the occupation matrix for all structures."""
        if not self.neb_entries:
            raise ValueError("No entries available")
        return np.array([entry.occupation.array for entry in self.neb_entries])
    
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

    @staticmethod
    def _as_weight_array(weight: float | Sequence[float], n_entries: int) -> np.ndarray:
        if np.isscalar(weight):
            return np.full(n_entries, float(weight))
        weight_array = np.asarray(weight, dtype=float)
        if weight_array.shape != (n_entries,):
            raise ValueError(
                f"weight must be a scalar or an array with shape ({n_entries},)"
            )
        return weight_array

    def write_fitting_inputs(
        self,
        output_dir: str | os.PathLike[str] = ".",
        weight: float | Sequence[float] = 1.0,
        corr_fname: str = "correlation_matrix.txt",
        ekra_fname: str = "e_kra.txt",
        weight_fname: str = "weight.txt",
    ) -> Dict[str, str]:
        """
        Write files consumed by ``LocalClusterExpansion.fit``.

        Args:
            output_dir: Directory for fitting input files.
            weight: Scalar sample weight or one weight per entry.
            corr_fname: Correlation matrix file name.
            ekra_fname: Target property file name.
            weight_fname: Sample weight file name.

        Returns:
            dict: Keyword arguments that can be passed to
                ``LocalClusterExpansion.fit``.
        """
        if not self.neb_entries:
            raise ValueError("No entries to save")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        corr_path = output_path / corr_fname
        ekra_path = output_path / ekra_fname
        weight_path = output_path / weight_fname

        np.savetxt(corr_path, self.get_correlation_matrix(), fmt="%.18e")
        np.savetxt(ekra_path, self.get_properties(), fmt="%.18e")
        np.savetxt(
            weight_path,
            self._as_weight_array(weight, len(self.neb_entries)),
            fmt="%.18e",
        )

        logger.info("Saved fitting inputs to %s", output_path)
        return {
            "corr_fname": str(corr_path),
            "ekra_fname": str(ekra_path),
            "weight_fname": str(weight_path),
        }
    
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
