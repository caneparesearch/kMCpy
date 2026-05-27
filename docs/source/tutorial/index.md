# Beginner Tutorial

This tutorial teaches the workflow, not the internals. A kMCpy simulation needs
a small set of scientific inputs:

- a crystal structure containing every possible mobile-ion site,
- a site mapping that defines mobile-ion and vacancy states,
- an event library describing allowed hops and event dependencies,
- a model that gives each event a rate,
- an initial occupation vector,
- runtime settings,
- a tracker for output and analysis.

The workflow is:

```text
structure or CIF
      |
      v
site_mapping and active-site order
      |
      +--> event library and dependencies
      |
      +--> local environments, NEB data, and model fitting
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

Read the pages in order the first time. Later, you can jump to the advanced
tutorials for custom properties, custom models, local barrier rules, basis
functions, and external-code adapters.

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
