"""Shared registry definitions for I/O modules."""

# Maps legacy model names to legacy task names.
MODEL_TASK_REGISTRY = {
    "lce": "lce",
    "composite_lce": "lce",
    "local_barrier": "lce",
    "external_site_energy": "lce",
    "mapped_site_energy": "lce",
    "zero_site_energy": "lce",
}

# Maps model names to fully-qualified model class paths.
MODEL_CLASS_REGISTRY = {
    "composite_lce": "kmcpy.models.composite_lce_model.CompositeLCEModel",
    "lce": "kmcpy.models.local_cluster_expansion.LocalClusterExpansion",
    "local_cluster_expansion": "kmcpy.models.local_cluster_expansion.LocalClusterExpansion",
    "local_barrier": "kmcpy.models.local_barrier_model.LocalBarrierModel",
    "external_site_energy": "kmcpy.models.site_energy.ExternalSiteEnergyModel",
    "mapped_site_energy": "kmcpy.models.site_energy.MappedSiteEnergyModel",
    "zero_site_energy": "kmcpy.models.site_energy.ZeroSiteEnergyModel",
}
