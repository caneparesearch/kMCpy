# Quickstart

This quickstart runs a minimal end-to-end kMC simulation using bundled example data.

## 1. Install

```shell
uv sync
```

## 2. Discover parameter names

Use `SimulationConfig.help_parameters()` to list valid keywords and see the system/runtime split:

```shell
uv run python -c "from kmcpy.simulator.config import SimulationConfig; SimulationConfig.help_parameters()"
```

## 3. Run the minimal script

```shell
uv run python example/minimal_example.py
```

This script uses:

1. System parameters: structure, event, and model files.
2. Runtime parameters: temperature, pass counts, and random seed.

It writes output to `example/output/minimal/`.

## Troubleshooting

### Unknown parameter error

If you see `Unknown parameters: [...]`, check spelling and compare against:

```python
from kmcpy.simulator.config import SimulationConfig

SimulationConfig.help_parameters()
```

### Missing required files

If file paths are invalid, use absolute paths or run from repository root so relative paths resolve correctly.

### YAML input for CLI

`run_kmc` expects modern `SimulationConfig` style content. Legacy `InputSet` style files are no longer supported.
