# Use External Site-Energy Models

This guide shows how to use a site-energy model from another code, such as
smol, CLEASE, ASE, or a project-specific cluster expansion, inside a kMCpy KMC
simulation.

The main rule is simple:

```text
delta_e_site = E_after_hop - E_before_hop
```

Your external model must return that signed energy difference for the proposed
event. kMCpy combines it with the KRA barrier as:

```text
effective_barrier = e_kra + delta_e_site / 2
```

In kMCpy, barriers and site-energy differences are consumed in `meV`. If the
external code returns `eV`, set `units="eV"` and kMCpy will convert it.

## Choose The Interface

Use one of these three options:

| Use case | Interface |
| --- | --- |
| No site-energy term | `site_model=None` or `ZeroSiteEnergyModel()` |
| A simple function already returns `E_after - E_before` | `ExternalSiteEnergyModel` |
| A live smol/CLEASE/ASE object must stay synchronized during KMC | `MappedSiteEnergyModel` |

For production smol or CLEASE runs, use `MappedSiteEnergyModel`. It avoids
rebuilding or converting the full occupation vector on every KMC step.

## What kMCpy Stores

kMCpy stores the simulation occupation as a compact active-site list:

```python
simulation_state.occupations
```

For a binary mobile-ion/vacancy model this often looks like:

```text
0 = mobile ion
1 = vacancy
```

For multicomponent models, states may be `0`, `1`, `2`, etc. These integers are
kMCpy state labels. External codes often use a different site order and
different occupation labels, so you must provide mappings.

## Required Mappings

`MappedSiteEnergyModel` needs two mappings.

`site_mapping` maps from kMCpy active-site index to the external code's site
index:

```python
site_mapping = {
    0: 12,  # kMCpy active site 0 is external site 12
    1: 18,
    2: 19,
}
```

`state_mapping` maps from kMCpy state index to the external occupation value:

```python
state_mapping = {
    0: 0,  # kMCpy mobile-ion state -> smol occupation code
    1: 1,  # kMCpy vacancy state -> smol occupation code
    2: 2,  # another species
}
```

For CLEASE or ASE, the external value may be a symbol:

```python
state_mapping = {
    0: "Li",
    1: "X",
}
```

If the same kMCpy state number means different external values on different
sublattices, use `state_mapping_by_site`:

```python
state_mapping_by_site = {
    0: {0: 0, 1: 1},
    1: {0: 3, 1: 4},
}
```

kMCpy validates the site mapping during initialization. If an `event_lib` is
available, it also checks that the endpoint state mappings needed by the events
are present before the KMC loop starts.

## Runtime Lifecycle

`MappedSiteEnergyModel` does three things:

1. At setup, it builds one external occupation array from the initial kMCpy
   occupation.
2. For a proposed event, it maps only the two endpoint changes and calls your
   `delta_fn`.
3. After an accepted event, it optionally calls your `apply_fn`, then updates
   only the two changed external occupation entries.

So the full occupation conversion happens once, not every event.

## Smol-Style Example

For smol, keep the smol processor as the runtime object. The delta function
turns kMCpy's mapped endpoint changes into smol-style flips:

```python
import numpy as np

from kmcpy.models import CompositeLCEModel, MappedSiteEnergyModel


def smol_delta(runtime, external_occupation, changes, coefficients):
    flips = [change.as_flip() for change in changes]
    delta_features = runtime.compute_feature_vector_change(
        external_occupation,
        flips,
    )
    return float(np.dot(delta_features, coefficients))  # eV


site_model = MappedSiteEnergyModel(
    runtime=smol_processor,
    delta_fn=smol_delta,
    delta_kwargs={"coefficients": smol_coefficients},
    site_mapping=kmcpy_to_smol_site,
    state_mapping=kmcpy_state_to_smol_code,
    external_dtype="int32",
    units="eV",
)

model = CompositeLCEModel(
    kra_model=kra_lce_model,
    site_model=site_model,
)
```

`compute_delta(...)` does not mutate `external_occupation`. kMCpy updates the
cached external occupation only after the event is accepted.

## CLEASE-Style Example

For CLEASE or ASE, keep the evaluator as the runtime object. The delta function
evaluates the proposed local changes; the apply function commits accepted
changes to the external evaluator.

```python
from kmcpy.models import CompositeLCEModel, MappedSiteEnergyModel


def clease_delta(runtime, external_occupation, changes):
    system_changes = [
        make_system_change(
            change.external_site,
            change.old_value,
            change.new_value,
        )
        for change in changes
    ]
    return runtime.get_energy_given_change(system_changes) - runtime.get_energy()


def clease_apply(runtime, external_occupation, changes):
    system_changes = [
        make_system_change(
            change.external_site,
            change.old_value,
            change.new_value,
        )
        for change in changes
    ]
    runtime.apply_system_changes(system_changes, keep=True)


site_model = MappedSiteEnergyModel(
    runtime=clease_evaluator,
    delta_fn=clease_delta,
    apply_fn=clease_apply,
    site_mapping=kmcpy_to_clease_site,
    state_mapping={0: "Li", 1: "X"},
    units="eV",
)

model = CompositeLCEModel(
    kra_model=kra_lce_model,
    site_model=site_model,
)
```

The `external_occupation` argument is still updated by kMCpy after accepted
events. Use `apply_fn` only for external runtime state that kMCpy cannot update
itself, such as a live evaluator or calculator.

## Model Files

Live Python objects cannot be serialized into model files. For file-based
workflows, provide factory references instead:

```python
site_model = MappedSiteEnergyModel(
    runtime_ref="my_project.smol_adapter:build_processor",
    runtime_kwargs={"model_file": "smol_ce.json"},
    delta_ref="my_project.smol_adapter:smol_delta",
    delta_kwargs={"coefficients_file": "eci.npy"},
    site_mapping=kmcpy_to_smol_site,
    state_mapping=kmcpy_state_to_smol_code,
    external_dtype="int32",
    units="eV",
)

site_model.to("site_energy.json")
```

At runtime, kMCpy resolves `runtime_ref`, `delta_ref`, and `apply_ref` from
their import paths.

## Simple Callable Adapter

Use `ExternalSiteEnergyModel` only when your callable already handles its own
state and returns the signed event energy difference:

```python
from kmcpy.models import ExternalSiteEnergyModel

site_model = ExternalSiteEnergyModel(
    callable_ref="my_project.site_energy:compute_delta",
    units="eV",
    kwargs={"model_file": "site_ce.json"},
)
```

The callable receives:

```python
def compute_delta(event, simulation_state, model_file):
    return 0.04  # E_after - E_before, in eV
```

Do not use this pattern if the callable rebuilds a full smol or CLEASE
occupation object every time it is called.

## No Site-Energy Term

If the KRA model already contains everything you need, omit the site model:

```python
model = CompositeLCEModel(
    kra_model=kra_lce_model,
    site_model=None,
)
```

or use an explicit zero model:

```python
from kmcpy.models import ZeroSiteEnergyModel

model = CompositeLCEModel(
    kra_model=kra_lce_model,
    site_model=ZeroSiteEnergyModel(),
)
```

## Checklist

Before running a simulation, check:

- `delta_fn` returns `E_after_hop - E_before_hop`, not an absolute energy.
- `units` matches the external model output.
- `site_mapping` covers every kMCpy active site.
- `state_mapping` or `state_mapping_by_site` covers every state needed by the
  events.
- `delta_fn` does not mutate the external occupation for proposed events.
- `apply_fn` mutates only accepted events.
- Full occupation conversion happens in `initialize_state(...)`, not in
  `compute_delta(...)`.
