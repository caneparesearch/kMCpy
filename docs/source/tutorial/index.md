# Workflow Overview

This tutorial follows the order of a kMCpy study. It teaches how the scientific
inputs are prepared before showing how those inputs are loaded into a run.

Every kMCpy simulation needs a small set of inputs:

- a crystal structure containing every possible mobile-ion site,
- a site mapping that defines mobile-ion and vacancy states,
- an event library describing allowed hops and event dependencies,
- a rate model,
- an initial occupation vector,
- runtime settings,
- a tracker for output and analysis.

The common workflow is:

```text
structure or CIF
      |
      v
site_mapping and active-site order
      |
      +--> event library
      |
      +--> rate model branch
              |
              +--> LocalBarrierModel rules
              |
              +--> LocalClusterExpansion local environments -> NEB/calculated data -> fit
                    |
                    v
                 model file
      |
      v
Configuration + initial occupations
      |
      v
KMC run -> Tracker -> results and analysis
```

There are two common model branches:

- **Local barrier branch**: write explicit barriers with
  [`LocalBarrierModel`](../modules/local_barrier_model.rst). You do not need
  NEB enumeration or fitting unless you want exact local-environment rules from
  calculated barriers.
- **LCE branch**: build a
  [`LocalClusterExpansion`](../modules/local_cluster_expansion.rst), enumerate
  local environments, collect NEB or other target data, fit coefficients, then
  use the fitted model in kMC.

Read [Structure And Active Sites](structure.md), [Choose And Build A
Model](models.md), and [Build The Event Library](events.md) first. If you use
the LCE branch, continue through [Local Environments And NEB Data](local_environments_neb.md)
and [Fit A Local Cluster Expansion](lce_fitting.md) before running kMC. If you
use a local barrier model, you can skip directly to [Prepare Input And Run
kMC](run_kmc.md).

Advanced pages cover custom properties, custom models, advanced local barrier
rules, basis functions, local site order, and external-code adapters.

```{toctree}
:maxdepth: 1

structure
models
events
local_environments_neb
lce_fitting
run_kmc
tracker_outputs
analysis
units_interoperability
```
