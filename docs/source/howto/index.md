# How-To Guides

Use these pages when you already know what you want to do and need the concrete
workflow.

## Simulation Workflows

- [Attach custom properties](attach_properties.md): sample user-defined
  quantities during a KMC run.
- [Use external site-energy-difference models](external_site_energy.md): connect
  smol, CLEASE, ASE, or project-specific energy-difference code without
  converting the full occupation every KMC step.

## Model And Fitting Workflows

- [Use local barrier models](local_barrier_model.md): define constant,
  rule-based, pattern-based, or exact-catalog migration barriers.
- [Prepare NEB fitting inputs](neb_fitting.md): turn NEB structures and
  barriers into fitting files for a local cluster expansion.
- [Control local site ordering](local_ordering.md): keep LCE feature ordering
  compatible with fitted coefficients.

## Structure Workflows

- [Enumerate local environments](local_environment_enumeration.md): generate
  ordered local structures and NEB endpoint pairs.

```{toctree}
:maxdepth: 1
:hidden:

attach_properties
external_site_energy
local_barrier_model
neb_fitting
local_ordering
local_environment_enumeration
```
