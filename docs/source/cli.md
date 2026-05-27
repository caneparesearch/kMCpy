# Command Line Interface

kMCpy provides two command-line entry points:

- `kmcpy`: scaffolding, sample generation, and a `run` alias.
- `run_kmc`: standalone simulation runner kept for existing workflows.

The recommended command-line workflow is:

```shell
kmcpy sample all --output-dir kmcpy_sample
# edit kmcpy_sample/input.yaml
kmcpy run --input kmcpy_sample/input.yaml
```

For real simulations, replace the placeholder structure and event paths in the
generated input with files prepared for your system.

## Create Input Files

Use `kmcpy init` when you want a commented template:

```shell
kmcpy init --output input_template.yaml
```

The template is intended for reading and editing. It shows the available
`Configuration` fields, units, and property-sampling options.

Use `kmcpy sample` when you want concrete starter files:

```shell
kmcpy sample all --output-dir kmcpy_sample
```

This writes:

- `kmcpy_sample/input.yaml`,
- `kmcpy_sample/model.json`,
- `kmcpy_sample/initial_state.json`.

The generated model is a constant-barrier
[`LocalBarrierModel`](modules/local_barrier_model.rst). The generated state is a
small active-site occupation vector. These files are useful for learning the file
format, but they are not a physical system until `structure_file` and
`event_file` are replaced with real files.

## Generate Individual Sample Files

Write only a sample input file:

```shell
kmcpy sample config --output input.yaml
```

Write a constant-barrier model:

```shell
kmcpy sample model --output model.json --barrier 300
```

Barrier values are in meV.

Write an initial state:

```shell
kmcpy sample state --output initial_state.json --occupations 0,1
```

Occupation values are active-site state indices. For a mapping such as
`{"Li": ["Li", "X"]}`, state `0` means `Li` and state `1` means vacancy `X`.

## Run A Simulation

Run from a YAML or JSON input:

```shell
kmcpy run --input input.yaml
```

This is the clearest interface for research workflows because the input file
records the structure path, event library, model, initial occupations, runtime
settings, and property-sampling controls together.

The standalone command is equivalent:

```shell
run_kmc --input input.yaml
```

Both `kmcpy run` and `run_kmc` also accept a small set of direct flags:

```shell
kmcpy run \
  --structure_file structure.cif \
  --event_file events.json \
  --model_file model.json \
  --site_mapping '{"Li": ["Li", "X"]}' \
  --supercell_shape '[1, 1, 1]'
```

Direct flags are useful for quick checks, but input files are preferred for
reproducible simulations.

## Inspect Help

Use `--help` on any command:

```shell
kmcpy --help
kmcpy init --help
kmcpy sample --help
kmcpy sample all --help
kmcpy run --help
run_kmc --help
```

To inspect valid configuration fields from Python:

```shell
python -c "from kmcpy.simulator.config import Configuration; Configuration.help_fields()"
```

## Related Pages

- [Quickstart](quickstart.md)
- [Prepare Input And Run kMC](tutorial/run_kmc.md)
- [Configuration API](modules/config.rst)
- [High-level Python API](modules/high_level_api.rst)
