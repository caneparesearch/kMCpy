# Build The Event Library

The event library contains the allowed hops and the dependency information
needed to update affected rates after a hop. Events are stored in the same
active-site order as the simulation occupations. Build it after the
[`site_mapping`](structure.md) is fixed and before writing the final
[`Configuration`](run_kmc.md).

## Generate Events

```python
from kmcpy.event import EventGenerator

site_mapping = {
    "Na": ["Na", "X"],
    "Zr": "Zr",
    "Si": ["Si", "P"],
    "O": "O",
}

generator = EventGenerator()
generator.generate_events(
    structure_file="nasicon.cif",
    site_mapping=site_mapping,
    supercell_shape=[2, 1, 1],
    mobile_species=["Na"],
    mobile_ion_identifiers=("Na1", "Na2"),
    local_env_cutoff=4.0,
    event_file="events.json",
)
```

`event_file` is a bundled event library. It contains both events and event
dependencies, so a separate dependency CSV is not needed for new workflows.

The important [`EventGenerator.generate_events(...)`](../modules/generators.rst)
arguments are:

- `structure_file`: CIF or structure file containing all possible mobile-ion
  sites.
- `site_mapping`: same active-site convention used by the structure, model, and
  simulation configuration.
- `supercell_shape`: event-library supercell relative to the input structure.
- `mobile_species`: species that can hop.
- `mobile_ion_identifiers`: optional pair of labels or species identifiers used
  to restrict hop endpoints.
- `local_env_cutoff`: local-environment cutoff in angstrom used for event
  dependencies and local model updates.
- `event_file`: output event-library JSON.

## Species And Label Selection

`site_mapping` is the primary convention for active sites. `mobile_species`
defines which species are mobile. `mobile_ion_identifiers` is only needed when
you want to restrict the hop endpoints further.

Use species identifiers when all sites of a species are valid hop endpoints:

```python
generator.generate_events(
    structure_file="toy.cif",
    site_mapping={"Li": ["Li", "X"], "O": "O"},
    mobile_species=["Li"],
    local_env_cutoff=4.0,
    event_file="events.json",
)
```

Use labels when only particular crystallographic sites should be connected:

```python
generator.generate_events(
    structure_file="nasicon.cif",
    site_mapping=site_mapping,
    mobile_species=["Na"],
    mobile_ion_identifiers=("Na1", "Na2"),
    event_file="events.json",
)
```

## What Dependencies Mean

Two events are dependent if accepting one event can change the availability or
rate of the other. For local barrier and LCE models, this usually means the
events share a mobile-ion endpoint or local-environment site.

kMCpy stores these dependencies with the event library so the simulator can
update rates locally instead of recalculating every event after every hop.

## Sanity Checks

After generation, inspect:

- the number of generated events,
- whether endpoint labels match the intended hop network,
- whether the supercell shape is large enough for the desired kMC cell,
- whether the event library was generated with the same `site_mapping` used by
  the simulation.

Next:

- local barrier branch: [Prepare Input And Run kMC](run_kmc.md);
- LCE branch: [Local Environments And NEB Data](local_environments_neb.md).
