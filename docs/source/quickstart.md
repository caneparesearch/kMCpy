# Quickstart

This page runs one small kMC simulation after kMCpy is installed. For
installation options, see [Install](install.md).

## Run The Bundled Example From Source

```shell
uv run python example/minimal_example.py
```

It writes results under `example/output/minimal/`. Use a source checkout,
because the example data files live in the repository.

## Run From Python

A simulation is assembled from one shared **lattice structure** and one
component per **slot**: events, rate model, and initial state. Any component that fits a
slot can be plugged in.

```python
import kmcpy

# Every site and what it may hold: Na or vacancy ("X"), Si or P; the other
# species are fixed. Partial occupancies in the CIF work the same way.
lattice = kmcpy.LatticeStructure.from_cif(
    "nasicon.cif",
    site_mapping={"Na": ["Na", "X"], "Si": ["Si", "P"]},
    primitive=True,
)
lattice.make_supercell((2, 1, 1))  # in place, like pymatgen

simulation = kmcpy.Simulation(
    lattice,
    events=kmcpy.HopEvents(cutoff=4.0),                     # or an events.json path
    model=kmcpy.LocalBarrierModel.constant_barrier(300.0),  # or a model.json path
    state=kmcpy.RandomOccupation({"Na": 0.75}, seed=1),     # or an initial-state path
    temperature=298.0,     # K
    kmc_passes=1000,
    random_seed=1,
)
tracker = simulation.run(output_dir="results")
```

Swapping a component changes one line; the rest of the setup stays the same.
`example/tutorial_nasicon.py` runs the same system with a fitted cluster
expansion or a constant barrier, from a stored or a random initial state.

A run needs only `temperature` (K) and `kmc_passes`, plus `random_seed` for a
reproducible run. `attempt_frequency` (Hz), `equilibration_passes`, and `name`
are optional. The mobile ion, its charge, and the hop length used for the
transport results are taken from `site_mapping`, the structure, and the
events.

## Run From An Input File

The same setup as a YAML file has one section per part:

```yaml
lattice_structure:
  structure: nasicon.cif
  site_mapping: {Na: [Na, X], Si: [Si, P]}
  supercell_shape: [2, 1, 1]
  primitive: true
events: {type: hop, cutoff: 4.0}                       # or: events.json
model: {type: local_barrier, default_barrier: 300.0}   # or: model.json
state: {type: random, fractions: {Na: 0.75}, seed: 1}  # or: initial_state.json
run: {temperature: 298.0, kmc_passes: 1000, random_seed: 1, output_dir: results}
```

Relative paths are resolved from the folder of the YAML file.

```python
simulation = kmcpy.Simulation.from_file("input.yaml")
tracker = simulation.run()
```

`tracker` contains the final state and sampled transport/property records.
Flat `Configuration` files (`Configuration.from_file(...)` with `kmcpy.run`)
keep working; `kmcpy init --format configuration` writes that format.

## Run From The Command Line

Create a commented template input file:

```shell
uv run kmcpy init --output input_template.yaml
```

Edit the file paths and runtime fields, then run:

```shell
uv run kmcpy run --input input_template.yaml
```

The standalone `run_kmc --input input_template.yaml` command is also supported.

The template uses the sectioned format above. `kmcpy run` also accepts flat
`Configuration` files.

To write concrete starter files instead of a commented template, use:

```shell
uv run kmcpy sample all --output-dir kmcpy_sample
```

This writes:

- `kmcpy_sample/input.yaml`,
- `kmcpy_sample/model.json`,
- `kmcpy_sample/initial_state.json`.

The sample model is a constant-barrier `LocalBarrierModel`. You still need to
provide a real structure file and event library (or `events: {type: hop, ...}`)
before running a physical simulation.

See [Command Line Interface](cli.md) for all scaffold and run commands.

## What The Input Must Provide

A kMC run needs:

- a structure containing all possible mobile-ion sites,
- a `site_mapping` that says which sites are mutable and which species or
  vacancy states are allowed,
- an event library,
- a model file that can assign rates to those events,
- initial occupations,
- runtime settings such as temperature, attempt frequency, and number of passes.

The workflow tutorial explains how to prepare each part.

## Inspect Configuration Fields

If you are unsure which fields belong in the input file, ask kMCpy:

```shell
uv run python -c "from kmcpy.simulator.config import Configuration; Configuration.help_fields()"
```

The output separates physical system inputs from runtime controls.

## Common Problems

### Unknown Field Error

If you see `Unknown configuration fields: [...]`, the input file contains a
misspelled or legacy field. Compare it against:

```python
from kmcpy.simulator.config import Configuration

Configuration.help_fields()
```

### Missing File Error

Relative file paths are resolved from the current working directory. Run from
the repository root or use absolute paths when debugging.

### No Results Written

Check that the run actually reached production steps and that the output
directory is writable. Attached custom properties are written separately from
built-in transport properties.
