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

## Smol Or CLEASE Adapters

For smol, write a callable that maps the kMCpy `simulation_state.occupations`
and event endpoint swap to the smol occupancy string, evaluates the energy
change, and returns that change. For performance, a stateful adapter should
keep the smol occupancy string in memory and use smol's local
feature-vector-change APIs for the two changed sites.

For CLEASE, write a callable that maps the event to CLEASE/ASE system changes,
uses the CLEASE evaluator/calculator to evaluate the changed energy, and returns
`E_after_hop - E_before_hop`. For performance, keep the CLEASE/ASE object alive
inside a stateful adapter and commit accepted changes in `apply_event(...)`.

Keep the adapter responsible for:

- mapping kMCpy active-site indices to the external code's site ordering,
- mapping kMCpy state indices to external species symbols or occupancy codes,
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
