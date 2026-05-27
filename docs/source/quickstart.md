# Quickstart

This page runs one small kMC simulation after kMCpy is installed. For
installation options, see [Install](install.md).

## Run The Bundled Example From Source

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

Use this command when you have a source checkout, because the example data files
live in the repository.

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

## What The Input Must Provide

A kMC run needs:

- a structure containing all possible mobile-ion sites,
- a `site_mapping` that says which sites are mutable and which species or
  vacancy states are allowed,
- an event library,
- a model file that can assign rates to those events,
- initial occupations,
- runtime settings such as temperature, attempt frequency, and number of passes.

The beginner tutorial explains how to prepare each part.

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
