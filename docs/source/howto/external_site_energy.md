# Use Site-Energy-Difference Models

This guide shows how to use site-energy-difference models inside a kMCpy KMC
simulation, including direct Python callables and external codes such as smol,
CLEASE, ASE, or a project-specific cluster expansion.

The main rule is simple:

```text
delta_e_site = E_after_hop - E_before_hop
```

The site model must return that signed energy difference for the proposed event.
kMCpy combines it with the KRA barrier as:

```text
effective_barrier = e_kra + delta_e_site / 2
```

In kMCpy, barriers and site-energy differences are consumed in `meV`. If a
callable or external code returns `eV`, set `units="eV"` and kMCpy will convert
it.

## Choose The Interface

Use one of these options:

| Use case | Interface |
| --- | --- |
| No site-energy-difference term | `site_model=None` |
| A kMCpy `LocalClusterExpansion` supplies the site-energy difference | `LocalClusterExpansion` as `site_model` |
| A Python function or external runtime supplies the site-energy difference | `SiteEnergyModel` |

`SiteEnergyModel` handles both simple kMCpy-native callables and mapped external
codes. If no mapping is supplied, it uses kMCpy's active-site order and
occupation labels directly. If an external code has its own site order or state
labels, provide `site_mapping` and `state_mapping`; kMCpy builds the mapping once
before the KMC loop and then updates only event endpoints.

When the site contribution is another kMCpy `LocalClusterExpansion`, it still
uses the regular `compute(simulation_state=..., event=...)` method. Its role in
`CompositeLCEModel` determines the meaning: as `kra_model` it returns
`E_KRA`, and as `site_model` it returns the site-energy-difference
contribution. `SiteEnergyModel` uses the same public `compute(...)` method and
returns `E_after_hop - E_before_hop` directly.

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

## Optional Mappings

Mappings are only needed when an external code uses a different site order or
different occupation labels.

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

During initialization, readable dictionaries/lists are converted to runtime
lookup arrays:

```text
site_lookup[kmcpy_active_site] -> external_site
state_lookup[kmcpy_state] -> external_occupation_value
state_lookup_by_site[kmcpy_site][kmcpy_state] -> external_occupation_value
```

The KMC loop then uses array indexing for event endpoints. It does not walk the
mapping dictionaries for every proposed hop.

## Build The External Site Mapping

Build `site_mapping` after the external code has finalized its site order. The
dictionary direction is always:

```python
site_mapping[kmcpy_active_site_index] = external_site_index
```

If the external object is built directly from kMCpy's active-site structure and
keeps the same order, no site mapping is needed:

```python
active_structure = active_site_index_map.active_structure()
external_runtime = build_external_runtime(active_structure)

site_model = SiteEnergyModel(
    runtime=external_runtime,
    compute_fn=external_delta,
    state_mapping=kmcpy_state_to_external_value,
    units="eV",
)
```

If the external structure preserves site properties, use the
`_kmcpy_active_site_index` property. This works for either active-only
structures or full structures with fixed sites:

```python
from kmcpy.structure.active_site_index_map import ACTIVE_SITE_PROPERTY


def site_mapping_from_property(external_structure, active_site_index_map):
    site_mapping = {}

    for external_index, site in enumerate(external_structure):
        active_index = site.properties.get(ACTIVE_SITE_PROPERTY)
        if active_index is None or int(active_index) < 0:
            continue
        site_mapping[int(active_index)] = int(external_index)

    if len(site_mapping) != active_site_index_map.active_site_count:
        raise ValueError("External structure does not contain every active site.")

    return site_mapping
```

If the external code drops site properties, match coordinates once before the
KMC run:

```python
import numpy as np


def site_mapping_from_coordinates(
    external_structure,
    active_site_index_map,
    tol=1e-3,
):
    active_structure = active_site_index_map.active_structure()
    site_mapping = {}
    used_external_sites = set()

    for active_index, active_site in enumerate(active_structure):
        distances = external_structure.lattice.get_all_distances(
            [active_site.frac_coords],
            external_structure.frac_coords,
        )[0]
        external_index = int(np.argmin(distances))

        if float(distances[external_index]) > tol:
            raise ValueError(f"No external site matches active site {active_index}.")
        if external_index in used_external_sites:
            raise ValueError("Two active sites map to the same external site.")

        site_mapping[active_index] = external_index
        used_external_sites.add(external_index)

    return site_mapping
```

Coordinate matching is a setup step, not a per-event operation. Store the
resulting dictionary in the `SiteEnergyModel`; kMCpy converts it to an array in
`initialize_state(...)`.

## Site-Order Traceability

kMCpy's own compact active-site order is defined by `ActiveSiteIndexMap`, the
same object used when structures, events, and occupations are loaded. When a
`SiteEnergyModel` is initialized during a normal KMC run, kMCpy passes
that map to the model. The model records:

- `kmcpy_site_order_hash`: the `ActiveSiteIndexMap.fingerprint` for the kMCpy
  active-site order.
- `external_site_order_hash`: a hash of the active-site to external-site
  mapping and external occupation size.

These hashes are serialized with the model so model files can be traced
back to the exact active-site ordering and external mapping they were built
against. If you construct the model manually, pass the index map explicitly:

```python
site_model = SiteEnergyModel(
    runtime=external_evaluator,
    compute_fn=external_delta,
    site_mapping=kmcpy_to_external_site,
    state_mapping=kmcpy_state_to_external_value,
    active_site_index_map=active_site_index_map,
    units="eV",
)
```

## Runtime Lifecycle

When mappings or an external runtime are used, `SiteEnergyModel` does three
things:

1. At setup, it builds site/state lookup arrays and one external occupation
   array from the initial kMCpy occupation.
2. For a proposed event, it maps only the two endpoint changes and calls your
   `compute_fn`.
3. After an accepted event, it optionally calls your `apply_fn`, then updates
   only the two changed external occupation entries.

So the full occupation conversion happens once, not every event.

## Smol-Style Example

For smol, keep the smol processor as the runtime object. The delta function
turns kMCpy's mapped endpoint changes into smol-style flips:

```python
import numpy as np

from kmcpy.models import CompositeLCEModel, SiteEnergyModel


def smol_delta(runtime, external_occupation, changes, coefficients):
    flips = [change.as_flip() for change in changes]
    delta_features = runtime.compute_feature_vector_change(
        external_occupation,
        flips,
    )
    return float(np.dot(delta_features, coefficients))  # eV


site_model = SiteEnergyModel(
    runtime=smol_processor,
    compute_fn=smol_delta,
    compute_kwargs={"coefficients": smol_coefficients},
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

`compute(...)` does not mutate `external_occupation`. kMCpy updates the
cached external occupation only after the event is accepted.

## CLEASE-Style Example

For CLEASE or ASE, keep the evaluator as the runtime object. The delta function
evaluates the proposed local changes; the apply function commits accepted
changes to the external evaluator.

```python
from kmcpy.models import CompositeLCEModel, SiteEnergyModel


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


site_model = SiteEnergyModel(
    runtime=clease_evaluator,
    compute_fn=clease_delta,
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
site_model = SiteEnergyModel(
    runtime_ref="my_project.smol_site_energy:build_processor",
    runtime_kwargs={"model_file": "smol_ce.json"},
    compute_ref="my_project.smol_site_energy:smol_delta",
    compute_kwargs={"coefficients_file": "eci.npy"},
    site_mapping=kmcpy_to_smol_site,
    state_mapping=kmcpy_state_to_smol_code,
    external_dtype="int32",
    units="eV",
)

site_model.to("site_energy.json")
```

At runtime, kMCpy resolves `runtime_ref`, `compute_ref`, and `apply_ref` from
their import paths.

## Direct Callable Example

For a kMCpy-native callable, omit `site_mapping` and `state_mapping`. The
callable can accept only the arguments it needs.

```python
from kmcpy.models import SiteEnergyModel

site_model = SiteEnergyModel(
    compute_ref="my_project.site_energy:site_energy_difference",
    units="eV",
    compute_kwargs={"model_file": "site_ce.json"},
)
```

The callable receives:

```python
def site_energy_difference(event, simulation_state, model_file):
    return 0.04  # E_after - E_before, in eV
```

Do not use this pattern if the callable rebuilds a full smol or CLEASE
occupation object every time it is called. Keep the external runtime in
`SiteEnergyModel` and provide mappings instead.

## No Site-Energy Term

If the KRA model already contains everything you need, omit the site model:

```python
model = CompositeLCEModel(
    kra_model=kra_lce_model,
    site_model=None,
)
```

## Checklist

Before running a simulation, check:

- `compute_fn` returns `E_after_hop - E_before_hop`, not an absolute energy.
- `units` matches the model output.
- `site_mapping` covers every kMCpy active site.
- `state_mapping` or `state_mapping_by_site` covers every state needed by the
  events.
- `compute_fn` does not mutate the external occupation for proposed events.
- `apply_fn` mutates only accepted events.
- Full occupation and mapping conversion happens in `initialize_state(...)`,
  not in `compute(...)`.
