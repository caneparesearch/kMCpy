# Prepare Input And Run kMC

After the structure, event library, model, and initial occupations are ready,
put them into a `Configuration`.

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

`run(config)` creates a `KMC` object, loads the model and event library, runs the
simulation, writes standard outputs, and returns the `Tracker`.

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

Create a template:

```shell
kmcpy init --output input_template.yaml
```

Edit the fields, then run:

```shell
run_kmc --input input_template.yaml
```

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
