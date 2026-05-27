# External Site-Energy Models

`CompositeLCEModel` uses two energy terms for an event:

```text
effective_barrier = e_kra + delta_e_site / 2
```

`e_kra` is still computed by a kMCpy `LocalClusterExpansion`. The
`delta_e_site` term can now come from either:

- a kMCpy `LocalClusterExpansion`, using the historical kMCpy directional
  convention, or
- an external model exposing `compute_delta(event=..., simulation_state=...)`.

External site-energy models must return:

```text
delta_e_site = E_after_hop - E_before_hop
```

in meV. If the external code works in eV, convert to meV in the adapter.

## Callable Adapter

`ExternalSiteEnergyModel` wraps a Python callable so smol, CLEASE, ASE, or
project-specific site-energy code can be used without becoming a hard kMCpy
dependency.

The callable is resolved from a string:

```python
from kmcpy.models import ExternalSiteEnergyModel

site_model = ExternalSiteEnergyModel(
    callable_ref="my_project.site_energy:compute_delta",
    units="eV",
    kwargs={"model_file": "site_ce.json"},
)
```

The callable must accept keyword arguments:

```python
def compute_delta(event, simulation_state, model_file):
    # Build or load any external evaluator needed by the project.
    # Return E_after_hop - E_before_hop.
    return 0.04  # eV
```

Because the adapter above uses `units="eV"`, kMCpy converts `0.04 eV` to
`40 meV` before constructing the event rate.

This callable adapter is best for simple or cached evaluators. Do not rebuild a
full smol/CLEASE occupancy object inside this callable for every event.

## Stateful Adapters

For production smol/CLEASE use, prefer a stateful adapter object. The adapter
should build its external occupancy once, then update only the accepted hop
endpoints after each KMC event:

```python
class MySmolSiteEnergy:
    def initialize_state(self, *, simulation_state, event_lib=None, structure=None, config=None):
        self.external_occ = build_external_occupancy(simulation_state.occupations)

    def compute_delta(self, event, simulation_state):
        changes = map_event_to_external_changes(event, self.external_occ)
        return compute_external_delta(changes)  # meV

    def apply_event(self, *, event, simulation_state):
        update_external_occupancy_in_place(event, self.external_occ)
```

kMCpy calls `initialize_state(...)` once during `KMC` setup and
`apply_event(...)` after every accepted event, after kMCpy has updated its own
`State`. Existing kMCpy LCE models ignore these hooks.

## Use With CompositeLCEModel

```python
from kmcpy.models import CompositeLCEModel, ExternalSiteEnergyModel

site_model = ExternalSiteEnergyModel(
    callable_ref="my_project.site_energy:compute_delta",
    units="eV",
)

model = CompositeLCEModel(
    kra_model=kra_lce_model,
    site_model=site_model,
)
model.to("model.json")
```

The resulting model file stores the external site model under `site`:

```json
{
  "filetype": "kmcpy.model_file",
  "model_type": "composite_lce",
  "kra": {
    "lce": {},
    "parameters": {}
  },
  "site": {
    "model_type": "external_site_energy",
    "delta_convention": "after_minus_before",
    "units": "meV",
    "model": {
      "@module": "kmcpy.models.site_energy",
      "@class": "ExternalSiteEnergyModel",
      "callable_ref": "my_project.site_energy:compute_delta",
      "units": "eV",
      "kwargs": {}
    }
  }
}
```

The `kra` section is abbreviated here; real model files include the serialized
KRA LCE and fitted parameters.

## Mapped Runtime Adapter

`MappedSiteEnergyModel` is the built-in stateful bridge for live external
runtime objects. It does not know how to build smol or CLEASE models. Instead,
it precomputes the compatible site/state mapping once, keeps one external
occupation array in memory, and passes only the two changed endpoints to your
delta function.

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
    runtime=my_smol_processor,
    delta_fn=smol_delta,
    delta_kwargs={"coefficients": smol_coefficients},
    site_mapping={0: 12, 1: 18},
    state_mapping={0: 0, 1: 1, 2: 2},
    external_dtype="int32",
    units="eV",
)

model = CompositeLCEModel(
    kra_model=kra_lce_model,
    site_model=site_model,
)
```

Use `state_mapping_by_site` when the same kMCpy state index maps to different
external species codes on different sublattices:

```python
site_model = MappedSiteEnergyModel(
    runtime=my_smol_processor,
    delta_fn=smol_delta,
    delta_kwargs={"coefficients": smol_coefficients},
    site_mapping={0: 12, 1: 18},
    state_mapping_by_site={
        0: {0: 0, 1: 1},
        1: {0: 3, 1: 4},
    },
    units="eV",
)
```

For CLEASE-like evaluators, keep the evaluator as the runtime object and provide
both a delta function and an apply function:

```python
def clease_delta(runtime, external_occupation, changes):
    system_changes = [
        make_system_change(change.external_site, change.old_value, change.new_value)
        for change in changes
    ]
    return runtime.get_energy_given_change(system_changes) - runtime.get_energy()


def clease_apply(runtime, external_occupation, changes):
    system_changes = [
        make_system_change(change.external_site, change.old_value, change.new_value)
        for change in changes
    ]
    runtime.apply_system_changes(system_changes, keep=True)


site_model = MappedSiteEnergyModel(
    runtime=my_clease_evaluator,
    delta_fn=clease_delta,
    apply_fn=clease_apply,
    site_mapping={0: 12, 1: 18},
    state_mapping={0: "Li", 1: "X"},
    units="eV",
)
```

In both cases, `initialize_state(...)` builds the full external occupation once.
During the KMC loop, `compute_delta(...)` maps only the proposed event
endpoints. After an accepted event, `apply_event(...)` updates only those two
external occupation entries.

For model files, use references instead of live Python objects:

```python
site_model = MappedSiteEnergyModel(
    runtime_ref="my_project.smol_adapter:build_processor",
    runtime_kwargs={"model_file": "smol_ce.json"},
    delta_ref="my_project.smol_adapter:smol_delta",
    delta_kwargs={"coefficients_file": "eci.npy"},
    site_mapping={0: 12, 1: 18},
    state_mapping={0: 0, 1: 1},
    units="eV",
)
site_model.to("mapped_site_energy.json")
```

The mapped adapter is responsible for:

- mapping kMCpy active-site indices to the external code's site ordering,
- mapping kMCpy state indices to external species symbols or occupancy codes,
- checking that the supplied mappings cover the kMCpy occupation vector,
- avoiding full occupation conversion inside the KMC hot loop,
- returning a signed energy difference,
- converting external energy units to `meV`.

## Zero Site-Energy Term

If you only want a KRA barrier model, use `site_model=None` or
`ZeroSiteEnergyModel()`:

```python
from kmcpy.models import CompositeLCEModel, ZeroSiteEnergyModel

model = CompositeLCEModel(
    kra_model=kra_lce_model,
    site_model=ZeroSiteEnergyModel(),
)
```
