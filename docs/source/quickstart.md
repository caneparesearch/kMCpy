# Quickstart

This page runs a small kMC simulation with the example files included in the
repository. Use it to check that your installation, input files, model loading,
and result writing are working.

## Install

From the repository root:

```shell
uv sync
```

For tests and documentation:

```shell
uv sync --extra dev
uv sync --extra doc
```

## Run The Bundled Example

```shell
uv run python example/minimal_example.py
```

The example loads:

- a structure file,
- an event file,
- a model file,
- an optional initial-state file,
- runtime settings such as temperature, number of passes, and random seed.

It writes results under:

```text
example/output/minimal/
```

## Run From Python

Use `Configuration.from_file(...)` when you already have a YAML or JSON input
file:

```python
from kmcpy import Configuration, run

config = Configuration.from_file("input.yaml")
tracker = run(config)
```

`tracker` contains the final state and sampled transport/property records.

## Run From The Command Line

Create a template input file:

```shell
uv run kmcpy init --output input_template.yaml
```

Edit the file paths and runtime fields, then run:

```shell
uv run run_kmc --input input_template.yaml
```

The generated template uses the modern `Configuration` format. Legacy
`InputSet` style inputs are no longer supported.

## Find Valid Configuration Fields

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
