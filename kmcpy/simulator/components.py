"""Load simulator components from a validated configuration."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

from kmcpy.structure.active_site_index_map import ActiveSiteIndexMap

from kmcpy.io.registry import MODEL_CLASS_REGISTRY

if TYPE_CHECKING:
    from kmcpy.simulator.config import Configuration

logger = logging.getLogger(__name__)

MODEL_REGISTRY = MODEL_CLASS_REGISTRY


def create_model_from_config(config: "Configuration"):
    """Create a model from configuration using the model registry."""
    model_type = getattr(config, "model_type", "composite_lce")

    if model_type not in MODEL_REGISTRY:
        available_types = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model type '{model_type}'. Available types: {available_types}")

    model_class_path = MODEL_REGISTRY[model_type]
    module_path, class_name = model_class_path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Cannot import model class '{model_class_path}': {e}")

    if hasattr(model_class, "from_config"):
        return model_class.from_config(config)

    raise ValueError(f"Model class '{class_name}' does not have a 'from_config' method")


def load_simulation_components(config: "Configuration") -> tuple:
    """Load structure, model, events, and compact active-site state."""
    from kmcpy.event import EventLib
    from kmcpy.external.structure import StructureKMCpy
    from kmcpy.simulator.state import State

    if config.site_mapping is None:
        raise ValueError(
            "site_mapping is required so kMC state, events, and model "
            "data use the same active-site index space."
        )

    full_structure = StructureKMCpy.from_cif(
        config.structure_file,
        primitive=config.convert_to_primitive_cell,
    )
    active_site_index_map = ActiveSiteIndexMap.from_structure_and_mapping(
        full_structure,
        config.site_mapping,
        supercell_shape=config.supercell_shape,
    )
    structure = active_site_index_map.active_structure()

    model = create_model_from_config(config)
    event_lib = EventLib.from_json(config.event_file)
    event_lib.validate_index_metadata(active_site_index_map)

    if config.initial_occupations is not None:
        simulation_state = State.from_occupations(
            config.initial_occupations,
            active_site_index_map=active_site_index_map,
        )
    elif config.initial_state_file:
        simulation_state = State.from_file(
            config.initial_state_file,
            supercell_shape=config.supercell_shape,
            active_site_index_map=active_site_index_map,
        )
    else:
        raise ValueError("Initial occupations could not be determined.")
    return structure, model, event_lib, simulation_state
