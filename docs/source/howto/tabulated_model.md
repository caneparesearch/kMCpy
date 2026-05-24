# Tabulated Model Workflow

`TabulatedModel` is designed for sparse datasets where you want exact lookup instead of fitted extrapolation.

This guide shows how to:

1. define tabulated entries,
2. build `model.json` via API or CLI,
3. run KMC with `model_type: "tabulated"`,
4. append entries programmatically.

## 1. Entry schema

Each entry is keyed by:

- `mobile_ion_indices`
- `local_env_indices`
- `occupations` on canonical sites (stable dedup of `mobile_ion_indices + local_env_indices`)

and stores one or more numeric properties (for example `barrier`).

```json
[
  {
    "mobile_ion_indices": [0, 1],
    "local_env_indices": [1, 2, 3],
    "occupations": [1, -1, 1, -1],
    "properties": {
      "barrier": 250.0,
      "energy": 1.2
    }
  }
]
```

Notes:

- Occupations must use Chebyshev values `-1` and `1`.
- Barrier values are expected in meV.
- Missing entries at runtime raise `KeyError` (no extrapolation fallback).

## 2. Build model file with API

Use `build_tabulated_model_file_from_entries_file(...)` to produce a validated model file:

```python
from kmcpy.io.model_file import (
    build_tabulated_model_file_from_entries_file,
    save_model_file,
)

model_data = build_tabulated_model_file_from_entries_file(
    entries_file="tabulated_entries.json"
)
save_model_file(model_data, "model.json")
```

You can also build directly from in-memory entries:

```python
from kmcpy.io.model_file import (
    build_tabulated_model_file,
    save_model_file,
)

model_data = build_tabulated_model_file(
    entries=[
        {
            "mobile_ion_indices": [0, 1],
            "local_env_indices": [1, 2, 3],
            "occupations": [1, -1, 1, -1],
            "properties": {"barrier": 250.0},
        }
    ],
    default_property="barrier",
    probability_property="barrier",
)
save_model_file(model_data, "model.json")
```

## 3. Build model file with CLI

```shell
kmcpy pack-tabulated-model \
  --entries-file tabulated_entries.json \
  --output model.json
```

Optional flags:

- `--name`
- `--default-property`
- `--probability-property`

## 4. Use in simulation configuration

Set `model_type` to `tabulated` and point `model_file` to the generated model file:

```yaml
kmc:
  type: default
  default:
    model_type: "tabulated"
    model_file: "model.json"
    # ... other required config fields
```

## 5. Programmatic model construction and incremental updates

`TabulatedModel` supports both constructor styles:

- `from_entries(...)`: construct from a full list
- `add_entry(...)`: append one validated entry

```python
from kmcpy.models import TabulatedEntry, TabulatedModel

model = TabulatedModel.from_entries(
    entries=[
        {
            "mobile_ion_indices": [0, 1],
            "local_env_indices": [1, 2, 3],
            "occupations": [1, -1, 1, -1],
            "properties": {"barrier": 250.0},
        }
    ]
)

model.add_entry(
    TabulatedEntry.from_dict(
        {
            "mobile_ion_indices": [0, 1],
            "local_env_indices": [1, 2, 3],
            "occupations": [-1, 1, 1, -1],
            "properties": {"barrier": 300.0},
        }
    )
)
```

## 6. Probability model

`TabulatedModel.compute_probability(...)` uses Arrhenius form with runtime values:

- `attempt_frequency` from config
- `temperature` from config
- tabulated `barrier` property

`rate = abs(direction) * v * exp(-barrier / (k_B * T))`

