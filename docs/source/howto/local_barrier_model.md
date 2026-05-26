# Local Barrier Models

`LocalBarrierModel` chooses a migration barrier from simple ordered local
environment rules. Use it when a full local cluster expansion is unnecessary or
when the barrier logic is known directly.

The model checks rules in order. The first matching rule supplies the requested
property, usually `barrier` in meV. If no rule matches, `default_properties` are
used when present; otherwise the lookup raises an error.

## Constant Barrier

Use a constant model when every hop has the same barrier:

```python
from kmcpy.models import LocalBarrierModel

model = LocalBarrierModel.constant_barrier(300.0)
model.to("model.json")
```

Equivalent YAML-style payload:

```yaml
model_type: local_barrier
local_barrier:
  default_barrier: 300.0
```

## Count Occupied Or Vacant Sites

Use `state_count` when the barrier depends on how many local sites are occupied
or vacant in the current KMC occupation vector.

```python
model = LocalBarrierModel(default_barrier=300.0)
model.add_state_count_rule(
    name="crowded",
    sites="local_env",
    state="occupied",
    min_count=3,
    barrier=450.0,
)
```

```yaml
model_type: local_barrier
local_barrier:
  default_barrier: 300.0
  rules:
    - name: crowded
      type: state_count
      sites: local_env
      state: occupied
      min_count: 3
      barrier: 450.0
```

Supported count fields:

- `count`: exactly this many sites
- `min_count`: at least this many sites
- `max_count`: at most this many sites

`occupied` maps to Chebyshev occupation `-1`; `vacant` maps to `+1`.

## Count Species

Use `species_count` when a rule depends on chemical identity, for example
"more than 3 Si in the local environment". The model needs `site_species` to
translate each site occupation into a species label.

```python
model = LocalBarrierModel(
    default_barrier=300.0,
    site_species={
        1: {-1: "P", 1: "Si"},
        2: {-1: "Si", 1: "P"},
        3: {-1: "Si", 1: "P"},
        4: {-1: "Al", 1: "Si"},
    },
)
model.add_species_count_rule(
    name="si_rich",
    sites="local_env",
    species="Si",
    min_count=4,
    barrier=420.0,
)
```

```yaml
model_type: local_barrier
local_barrier:
  default_barrier: 300.0
  site_species:
    "1": {"-1": P, "1": Si}
    "2": {"-1": Si, "1": P}
    "3": {"-1": Si, "1": P}
    "4": {"-1": Al, "1": Si}
  rules:
    - name: si_rich
      type: species_count
      sites: local_env
      species: Si
      min_count: 4
      barrier: 420.0
```

## Pattern Rules

Use `pattern` for a small number of explicit occupation patterns. `*` is a
wildcard. For many exact patterns, use exact rules instead.

```python
model.add_pattern_rule(
    name="pattern_1",
    sites="canonical",
    pattern=["occupied", "vacant", "*", "occupied"],
    barrier=250.0,
)
```

`canonical` means `mobile_ion_indices` first, followed by
`local_env_indices`, skipping duplicate sites.

## Exact Catalog Entries

Use `exact` rules when a specific event and exact local occupation pattern has
a known barrier.

```python
from kmcpy.models import LocalBarrierModel

model = LocalBarrierModel.from_exact_entries(
    [
        {
            "mobile_ion_indices": [0, 1],
            "local_env_indices": [1, 2, 3],
            "occupations": [1, -1, 1, -1],
            "properties": {"barrier": 250.0},
        }
    ]
)
```

You can also capture an exact rule from an event and state snapshot:

```python
rule = LocalBarrierModel.entry_from_event_state(
    event=event,
    state=state,
    properties={"barrier": 250.0},
)
model = LocalBarrierModel(rules=[rule])
```

## Site Selectors

Rules can select sites with:

- `local_env`: `event.local_env_indices`
- `mobile_ion`: `event.mobile_ion_indices`
- `canonical`: mobile-ion sites followed by local-environment sites, skipping duplicates
- `from`: the first mobile-ion site
- `to`: the second mobile-ion site
- `all`: every site in the compact KMC state
- an explicit list of active-site indices

## Simulation Config

Save the model and point `model_file` to it:

```python
model.to("model.json")
```

```yaml
kmc:
  type: default
  default:
    model_file: model.json
```

The model file uses:

```json
{
  "filetype": "kmcpy.model_file",
  "model_type": "local_barrier",
  "local_barrier": {
    "default_properties": {"barrier": 300.0},
    "rules": []
  }
}
```
