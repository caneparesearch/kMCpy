# Quickstart

This quickstart runs a minimal end-to-end kMC simulation using bundled example data.

## 1. Install

```shell
uv sync
```

## 2. Discover Field Names

Use `Configuration.help_fields()` to list valid keywords and see the system/runtime split:

```shell
uv run python -c "from kmcpy.simulator.config import Configuration; Configuration.help_fields()"
```

## 3. Run the minimal script

```shell
uv run python example/minimal_example.py
```

This script uses:

1. Loader inputs: structure, event, model, and optional state files.
2. Runtime fields: temperature, pass counts, and random seed.

It writes output to `example/output/minimal/`.

## 4. Run with one API call

```python
from kmcpy import Configuration, run

config = Configuration.from_file("input.yaml")
tracker = run(config)
```

## 5. Scaffold a template YAML for CLI/API

```shell
kmcpy init --output input_template.yaml
```

Then edit the required file paths and run:

```shell
run_kmc --input input_template.yaml
```

API users can load the same generated file with:

```python
from kmcpy.simulator.config import Configuration

config = Configuration.from_file("input_template.yaml")
```

## Troubleshooting

### Unknown Field Error

If you see `Unknown configuration fields: [...]`, check spelling and compare against:

```python
from kmcpy.simulator.config import Configuration

Configuration.help_fields()
```

### Missing required files

If file paths are invalid, use absolute paths or run from repository root so relative paths resolve correctly.

### YAML input for CLI

`run_kmc` expects modern `Configuration` style content. Legacy `InputSet` style files are no longer supported.
