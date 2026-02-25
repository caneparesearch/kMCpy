# Quickstart

This quickstart runs a minimal end-to-end kMC simulation using bundled example data.

## 1. Install

```shell
uv sync
```

## 2. Discover parameter names

Use `Configuration.help_parameters()` to list valid keywords and see the system/runtime split:

```shell
uv run python -c "from kmcpy.simulator.config import Configuration; Configuration.help_parameters()"
```

## 3. Run the minimal script

```shell
uv run python example/minimal_example.py
```

This script uses:

1. System parameters: structure, event, and model files.
2. Runtime parameters: temperature, pass counts, and random seed.

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

config = Configuration.from_yaml_section("input_template.yaml", "kmc", "default")
```

## Troubleshooting

### Unknown parameter error

If you see `Unknown parameters: [...]`, check spelling and compare against:

```python
from kmcpy.simulator.config import Configuration

Configuration.help_parameters()
```

### Missing required files

If file paths are invalid, use absolute paths or run from repository root so relative paths resolve correctly.

### YAML input for CLI

`run_kmc` expects modern `Configuration` style content. Legacy `InputSet` style files are no longer supported.
