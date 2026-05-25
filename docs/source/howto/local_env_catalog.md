# Local Environment Catalog Workflow

`LocalEnvCatalog` is designed for sparse datasets where you want exact lookup instead of fitted extrapolation.

This guide shows how to:

1. build a raw `local_env_catalog_entries.json` file,
2. pack it into a validated `model.json` via API or CLI,
3. run KMC by pointing the simulation config at `model.json`,
4. append entries programmatically.

## 1. Build the entries JSON

The input to `kmcpy pack-local-env-catalog` is a raw entries JSON file. The pack command validates and wraps these rows; it does not infer hop sites, local environments, occupations, or barriers from structures. Those values should come from the event/environment workflow used to produce the data.

Each entry needs these fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `mobile_ion_indices` | yes | Two active-site indices for the hop, normally copied from `Event.mobile_ion_indices`. |
| `local_env_indices` | yes | Active-site indices included in the local environment lookup. These may overlap with `mobile_ion_indices`. |
| `occupations` | yes | Chebyshev occupation values for the canonical site order. The length must match the number of unique sites in `mobile_ion_indices + local_env_indices`. |
| `properties` | yes | Numeric values attached to this exact hop/environment, such as `barrier` in meV. |

The canonical site order is formed by taking `mobile_ion_indices` first and then appending `local_env_indices`, skipping duplicates. For example, `mobile_ion_indices = [0, 1]` and `local_env_indices = [1, 2, 3]` gives canonical sites `[0, 1, 2, 3]`, so `occupations` must contain four values in that order.

Use active-site indices, not raw full-structure site indices. In normal workflows these indices come from generated `Event` objects and the same active-site index map used by the KMC state.

The raw file can be either a JSON list of entries or an object with an `entries` key.

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

The object form also supports metadata used when packing the model file:

| Field | Required | Default | Meaning |
| --- | --- | --- | --- |
| `entries` | yes | none | List of catalog entries. |
| `name` | no | `LocalEnvCatalog` | Name stored in the packed model file. |
| `default_property` | no | `barrier` | Property returned by `compute(...)` when no property name is passed. |
| `probability_property` | no | `barrier` | Property used by `compute_probability(...)`. |

```json
{
  "name": "nzsp_neb_catalog",
  "default_property": "barrier",
  "probability_property": "barrier",
  "entries": [
    {
      "mobile_ion_indices": [0, 1],
      "local_env_indices": [1, 2, 3],
      "occupations": [1, -1, 1, -1],
      "properties": {"barrier": 250.0}
    }
  ]
}
```

You can generate this file from Python with Monty serialization:

```python
from monty.serialization import dumpfn
from kmcpy.models import LocalEnvCatalogEntry

entries = [
    LocalEnvCatalogEntry.from_dict(
        {
            "mobile_ion_indices": [0, 1],
            "local_env_indices": [1, 2, 3],
            "occupations": [1, -1, 1, -1],
            "properties": {"barrier": 250.0, "energy": 1.2},
        }
    ).as_dict()
]

dumpfn(
    {
        "name": "nzsp_neb_catalog",
        "default_property": "barrier",
        "probability_property": "barrier",
        "entries": entries,
    },
    "local_env_catalog_entries.json",
    indent=2,
)
```

Notes:

- Occupations must use Chebyshev values `-1` and `1`, matching the simulation `State` convention.
- Barrier values are expected in meV when used with the built-in Arrhenius probability calculation.
- Every runtime event/state pattern must have a matching entry. Missing entries raise `KeyError` because there is no extrapolation fallback.

## 2. Pack a model file with API

Use `LocalEnvCatalog.from_file(...)` to read a raw entries file and write a validated model file:

```python
from kmcpy.models import LocalEnvCatalog

model = LocalEnvCatalog.from_file("local_env_catalog_entries.json")
model.to("model.json")
```

You can also build directly from in-memory entries:

```python
from kmcpy.models import LocalEnvCatalog

model = LocalEnvCatalog.from_entries(
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
model.to("model.json")
```

## 3. Pack a model file with CLI

```shell
kmcpy pack-local-env-catalog \
  --entries-file local_env_catalog_entries.json \
  --output model.json
```

Optional flags:

- `--name`
- `--default-property`
- `--probability-property`

The generated `model.json` is a model-file envelope:

```json
{
  "filetype": "kmcpy.model_file",
  "model_type": "local_env_catalog",
  "local_env_catalog": {
    "@module": "kmcpy.models.local_env_catalog",
    "@class": "LocalEnvCatalog",
    "name": "nzsp_neb_catalog",
    "default_property": "barrier",
    "probability_property": "barrier",
    "entries": [
      {
        "mobile_ion_indices": [0, 1],
        "local_env_indices": [1, 2, 3],
        "occupations": [1, -1, 1, -1],
        "properties": {"barrier": 250.0}
      }
    ]
  }
}
```

`filetype` identifies the storage envelope. `model_type` identifies the model inside the file and is inferred when the simulator loads `model_file`.

## 4. Use in simulation configuration

Point `model_file` to the generated model file. The rest of the simulation still needs the usual KMC inputs: `structure_file`, `event_file`, `site_mapping`, `initial_occupations` or `initial_state_file`, and runtime settings such as `temperature` and `attempt_frequency`.

```yaml
kmc:
  type: default
  default:
    model_file: "model.json"
    # ... other required config fields
```

## 5. Programmatic model construction and incremental updates

`LocalEnvCatalog` supports both constructor styles:

- `from_entries(...)`: construct from a full list
- `add_entry(...)`: append one validated entry

```python
from kmcpy.models import LocalEnvCatalogEntry, LocalEnvCatalog

model = LocalEnvCatalog.from_entries(
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
    LocalEnvCatalogEntry.from_dict(
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

`LocalEnvCatalog.compute_probability(...)` uses Arrhenius form with runtime values:

- `attempt_frequency` from config
- `temperature` from config
- catalog `barrier` property

`rate = abs(direction) * v * exp(-barrier / (k_B * T))`

