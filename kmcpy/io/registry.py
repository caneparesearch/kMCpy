"""Shared registry definitions for I/O modules."""

# Maps legacy model names to legacy task names.
MODEL_TASK_REGISTRY = {
    "lce": "lce",
    "composite_lce": "lce",
}

# Maps model names to fully-qualified model class paths.
MODEL_CLASS_REGISTRY = {
    "composite_lce": "kmcpy.models.composite_lce_model.CompositeLCEModel",
    "lce": "kmcpy.models.local_cluster_expansion.LocalClusterExpansion",
    "local_cluster_expansion": "kmcpy.models.local_cluster_expansion.LocalClusterExpansion",
}
