# Prepare Input And Run kMC

After the structure, event library, model, and initial occupations are ready,
combine them into a run. In Python, plug them into a
[`Simulation`](../modules/simulation.rst); for input files and the CLI, use a
[`Configuration`](../modules/config.rst).

## Assemble A Simulation In Python

```python
import kmcpy

lattice = kmcpy.LatticeStructure.from_cif(
    "nasicon.cif",
    site_mapping={"Na": ["Na", "X"], "Si": ["Si", "P"]},
    primitive=True,
)
lattice.make_supercell((2, 1, 1))  # in place, like pymatgen

simulation = kmcpy.Simulation(
    lattice,
    events="events.json",
    model="model.json",
    state="initial_state.json",
    temperature=298.0,
    attempt_frequency=5e12,
    equilibration_passes=1000,
    kmc_passes=10000,
    random_seed=12345,
    name="NASICON_298K",
)
tracker = simulation.run(output_dir="results")
```

The mobile ion (`Na`), its charge (+1, from the structure's oxidation states),
and the hop length used for the correlation factor (from the events) are
derived; pass `mobile_ion_specie`, `mobile_ion_charge`, or
`elementary_hop_distance` only to override them.

The [`LatticeStructure`](../modules/lattice_structure.rst) is the part every
component shares. It is the disordered structure: every site that can be
occupied and the species it may hold. Partial occupancies in the CIF define
these (`Na: 0.75` means Na or vacancy); `site_mapping` lists what may vary for
fully occupied sites, and unlisted species are fixed. `make_supercell(...)` turns
it into the simulated supercell, and from it comes the order of the active sites. Event
files, models, and states are checked against it, so they cannot silently refer
to a different cell.

A run takes and returns ordered structures as well: `state=` accepts a pymatgen
`Structure` with one configuration (each active site matched by position, empty
sites read as vacancies), and `lattice.structure_from_occupations(
tracker.state.occupations)` turns the occupations back into a structure.

Each slot accepts files or objects:

| Slot | Accepts |
|---|---|
| `events` | an event-file path, an `EventLib`, or a generator such as `kmcpy.HopEvents(cutoff=4.0)` |
| `model` | a model-file path of any registered type, or a model object such as `LocalBarrierModel` or `CompositeLCEModel` |
| `state` | an initial-state file path, a `State`, a list of active-site occupations, or a builder such as `kmcpy.RandomOccupation({"Na": 0.75})` |

Other keyword arguments are run settings with the same names as the
`Configuration` fields below. `simulation.attach(func, interval=...)` records a
custom property during the run (see [Attach properties](../howto/attach_properties.md)),
and `simulation.build()` returns the underlying `KMC` object.

Your own model class plugs in the same way. Register it to load model files by
type name:

```python
@kmcpy.register_model("my_model")
class MyModel(kmcpy.BaseModel):
    ...
```

## Create A Configuration (Input Files)

```python
from kmcpy import Configuration, run

config = Configuration(
    structure_file="nasicon.cif",
    model_file="model.json",
    event_file="events.json",
    initial_state_file="initial_state.json",
    supercell_shape=(2, 1, 1),
    mobile_ion_specie="Na",
    elementary_hop_distance=3.47782,
    site_mapping={"Na": ["Na", "X"], "Zr": "Zr", "Si": ["Si", "P"], "O": "O"},
    convert_to_primitive_cell=True,
    temperature=298.0,
    attempt_frequency=5e12,
    equilibration_passes=1000,
    kmc_passes=10000,
    random_seed=12345,
    name="NASICON_298K",
)

tracker = run(config)
```

[`run(config)`](../modules/high_level_api.rst) loads the files, runs the
simulation, writes standard outputs, and returns the
[`Tracker`](../modules/tracker.rst).

The important `Configuration` fields are:

- `structure_file`, `model_file`, `event_file`, `initial_state_file` (or
  `initial_occupations`): loader paths used to start the run.
- `supercell_shape`, `site_mapping`, `convert_to_primitive_cell`: must match
  the event library and model.
- `temperature`, `attempt_frequency`: rate-model runtime conditions.
- `equilibration_passes`, `kmc_passes`, `random_seed`: simulation controls.
- `mobile_ion_specie`, `mobile_ion_charge`, `elementary_hop_distance`,
  `dimension`: transport-output metadata.

## Write A Reloadable Input File

Loader-only paths such as `structure_file`, `model_file`, and `event_file` are
needed to start a run, but they are not intrinsic recorded metadata after the
objects are loaded. Use `include_loader_paths=True` when writing a file that
should be used as an input later:

```python
config.to("input.yaml", include_loader_paths=True)
```

Load and run it:

```python
config = Configuration.from_file("input.yaml")
tracker = run(config)
```

## Run From The CLI

Create a commented template:

```shell
kmcpy init --output input_template.yaml
```

Edit the fields, then run:

```shell
kmcpy run --input input_template.yaml
```

The standalone `run_kmc --input input_template.yaml` command is also supported.

For concrete starter files, generate a small local-barrier sample set:

```shell
kmcpy sample all --output-dir kmcpy_sample
```

This writes `input.yaml`, `model.json`, and `initial_state.json`. Replace the
placeholder `structure_file` and `event_file` values with files prepared for
your system.

## Change Runtime Conditions

Use `with_runtime_changes(...)` for temperature sweeps without rebuilding the
system setup:

```python
for temperature in [300.0, 400.0, 500.0]:
    sweep_config = config.with_runtime_changes(
        temperature=temperature,
        name=f"NASICON_{temperature:.0f}K",
    )
    run(sweep_config)
```

Keep the event library and model fixed unless the physical system or active-site
order changes.

Next: [Track Outputs](tracker_outputs.md).
