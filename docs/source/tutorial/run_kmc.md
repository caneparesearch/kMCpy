# Prepare Input And Run kMC

After the structure, event library, model, and initial occupations are ready,
put them into a [`Configuration`](../modules/config.rst).

## Create A Configuration In Python

```python
from kmcpy import Configuration, run

config = Configuration(
    structure_file="nasicon.cif",
    model_file="model.json",
    event_file="events.json",
    initial_occupations=initial_occupations,
    supercell_shape=(2, 1, 1),
    dimension=3,
    mobile_ion_specie="Na",
    mobile_ion_charge=1.0,
    elementary_hop_distance=3.47782,
    site_mapping={"Na": ["Na", "X"], "Zr": "Zr", "Si": ["Si", "P"], "O": "O"},
    convert_to_primitive_cell=False,
    temperature=298.0,
    attempt_frequency=5e12,
    equilibration_passes=1000,
    kmc_passes=10000,
    random_seed=12345,
    name="NASICON_298K",
)

tracker = run(config)
```

[`run(config)`](../modules/high_level_api.rst) creates a `KMC` object, loads the
model and event library, runs the simulation, writes standard outputs, and
returns the [`Tracker`](../modules/tracker.rst).

The important `Configuration` fields are:

- `structure_file`, `model_file`, `event_file`: loader paths used to start the
  run.
- `initial_occupations`: active-site occupation vector from
  [Prepare Structures And Occupations](structure.md).
- `supercell_shape`, `site_mapping`: must match the event library and model.
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
