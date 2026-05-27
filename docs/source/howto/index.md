# Advanced Tutorials

Use these pages after you understand the basic kMCpy workflow. They focus on
customization, external interfaces, and order-sensitive model details.

For a beginner path, start with the [tutorial workflow](../tutorial/index.md).

## Simulation Customization

- [Attach custom properties](attach_properties.md): sample user-defined
  quantities during a kMC run.
- [Write a custom model](custom_model.md): implement the small model surface
  kMCpy needs without adding extra framework layers.

## Model Customization

- [Use advanced local barrier rules](local_barrier_model.md): define constant,
  count-based, species-count, wildcard, and exact local-environment barriers.
- [Choose basis functions](basis_functions.md): understand occupation and
  Chebyshev basis functions for binary and multicomponent sites.
- [Prepare NEB fitting inputs](neb_fitting.md): turn NEB structures and
  barriers into fitting files for a local cluster expansion.

## Site Order And External Codes

- [Control local site order](local_site_order.md): keep LCE feature order
  compatible with fitted coefficients.
- [Use site-energy-difference models](external_site_energy.md): connect direct
  callables or mapped external codes without converting the full occupation
  every kMC step.
- [Enumerate local environments](local_environment_enumeration.md): generate
  ordered local structures and NEB endpoint pairs.

```{toctree}
:maxdepth: 1
:hidden:

attach_properties
custom_model
local_barrier_model
basis_functions
neb_fitting
local_site_order
external_site_energy
local_environment_enumeration
```
