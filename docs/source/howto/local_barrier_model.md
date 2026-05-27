# Local Barrier Models

Use `LocalBarrierModel` when the migration barrier can be chosen from explicit
local rules instead of a fitted local cluster expansion.

Good examples are:

- every hop has the same barrier,
- the barrier changes when the local environment is crowded,
- the barrier depends on how many Si/P/Mg sites are nearby,
- a few exact local environments are known from NEB calculations,
- you want a transparent rule-based model before fitting an LCE.

`LocalBarrierModel` is not a fitting tool. It is a direct rule engine: for each
candidate event, it checks the rules in order and uses the first match.

## Choose A Rule Type

Start with the simplest rule that describes the physics:

| Question | Rule type |
| --- | --- |
| Is every hop the same? | constant barrier |
| Does the barrier depend on how many sites have a state index? | `state_count` |
| Does the barrier depend on how many sites are a species, such as Si? | `species_count` |
| Does the barrier depend on a local pattern with wildcards? | `pattern` |
| Do you have a catalog of exact local environments? | `exact` |

Put specific rules before broad rules. The first matching rule wins.

## Units

All barrier-like numeric values in `LocalBarrierModel` are in meV. This includes
`default_barrier`, rule-level `barrier`, and `properties: {"barrier": ...}`.
`compute_probability(...)` returns an event rate in Hz because it multiplies the
Arrhenius factor by `attempt_frequency`, which is also in Hz. Temperature is in
K.

## Basic Setup

The usual workflow is:

1. Build a `LocalBarrierModel` in Python.
2. Add rules from most specific to most general.
3. Add a fallback with `default_barrier` or `default_properties`.
4. Save the model with `model.to("model.json")`.
5. Point the simulation config `model_file` to that file.

```python
from kmcpy.models import LocalBarrierModel

model = LocalBarrierModel(default_barrier=300.0)
model.add_state_count_rule(
    name="crowded_local_env",
    sites="local_env",
    state="occupied",
    min_count=3,
    barrier=450.0,
)
model.to("model.json")
```

Then use the saved model in a KMC input file:

```yaml
kmc:
  type: default
  default:
    model_file: model.json
```

The model file declares its own type, so `BaseModel.from_config(...)` and the
simulation runner can dispatch to `LocalBarrierModel`:

```json
{
  "filetype": "kmcpy.model_file",
  "model_type": "local_barrier",
  "local_barrier": {
    "default_properties": {"barrier": 300.0},
    "rules": [
      {
        "name": "crowded_local_env",
        "type": "state_count",
        "sites": "local_env",
        "state": "occupied",
        "min_count": 3,
        "properties": {"barrier": 450.0}
      }
    ]
  }
}
```

## How Rules Are Evaluated

For each candidate event, the model checks rules in list order. The first
matching rule supplies a numeric property dictionary. By default, `compute(...)`
returns the `barrier` property in meV.

If no rule matches:

- `default_properties` are used if provided,
- `default_barrier=300.0` is treated as `default_properties={"barrier": 300.0}`,
- otherwise the lookup raises an error with the event sites and current
  occupations.

Put narrow rules before broad rules:

```python
model = LocalBarrierModel(default_barrier=300.0)

# More specific rule first.
model.add_pattern_rule(
    name="exact_shape",
    sites="canonical",
    pattern=["occupied", "vacant", "*", "occupied"],
    barrier=250.0,
)

# Broader rule second.
model.add_state_count_rule(
    name="crowded",
    sites="local_env",
    state="occupied",
    min_count=3,
    barrier=450.0,
)
```

`compute_probability(...)` uses the selected barrier in the Arrhenius form:

```text
rate = hop_available * attempt_frequency * exp(-barrier / (kB * temperature))
```

where `hop_available` is 1 only when the current endpoint states match a
mobile-vacancy hop, and the temperature and attempt frequency come from
`RuntimeConfig`.

## Occupation Values

The compact KMC occupation vector stores nonnegative integer state indices. If
an active site allows `q` species/states, valid state values are `0..q-1`; the
meaning of each state index follows the active-site species order from the
structure/site mapping.

For binary models, these aliases are available:

- `occupied`, `match`, and `template` mean state `0`
- `vacant`, `vacancy`, `mismatch`, and `other` mean state `1`

Rules can use either the aliases or explicit integer state values:

```python
model.add_state_count_rule(
    sites="local_env",
    state=2,
    min_count=2,
    barrier=410.0,
)
```

## Site Selectors

Rules need to know which sites to inspect. Use one of these selectors:

- `local_env`: `event.local_env_indices`
- `mobile_ion`: `event.mobile_ion_indices`
- `canonical`: mobile-ion sites followed by local-environment sites, skipping duplicates
- `from`: the first mobile-ion site
- `to`: the second mobile-ion site
- `all`: every site in the compact KMC state
- an explicit list of active-site indices, such as `[2, 5, 8]`

`canonical` is important for exact rules and pattern rules because it gives a
stable order. For an event with:

```python
event.mobile_ion_indices = (0, 1)
event.local_env_indices = (1, 2, 3)
```

the canonical site order is:

```python
(0, 1, 2, 3)
```

Site `1` appears only once because it is already one of the mobile-ion sites.

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

You can also store other numeric properties, but `compute_probability(...)`
uses the `barrier` property unless `probability_property` is changed:

```python
model = LocalBarrierModel(
    default_properties={
        "barrier": 300.0,
        "label_id": 1.0,
    }
)
```

## Count Sites In A State

Use `state_count` when the barrier depends on how many selected sites have a
given state index in the current KMC occupation vector. Binary rules can use
`"occupied"` and `"vacant"` aliases; multicomponent rules should use explicit
state indices such as `2` or `3`.

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

Example: lower the barrier when the destination site is vacant:

```python
model.add_state_count_rule(
    name="open_destination",
    sites="to",
    state="vacant",
    count=1,
    barrier=220.0,
)
```

Example: use one barrier when at most one local-environment site is occupied:

```python
model.add_state_count_rule(
    name="open_local_env",
    sites="local_env",
    state="occupied",
    max_count=1,
    barrier=240.0,
)
```

Example: use a different barrier when at least two local-environment sites are
in state `2`:

```python
model.add_state_count_rule(
    name="state_two_rich",
    sites="local_env",
    state=2,
    min_count=2,
    barrier=510.0,
)
```

Supported count fields:

- `count`: exactly this many sites
- `min_count`: at least this many sites
- `max_count`: at most this many sites

Do not combine `count` with `min_count` or `max_count`. Use `min_count` and
`max_count` together only when you want a range.

## Count Species

Use `species_count` when a rule depends on chemical identity, for example
"more than 3 Si in the local environment". Since the KMC state stores compact
occupation values, the model needs `site_species` to translate each selected
site and occupation value into a species label.

```python
model = LocalBarrierModel(
    default_barrier=300.0,
    site_species={
        1: {0: "P", 1: "Si"},
        2: {0: "Si", 1: "P"},
        3: {0: "Si", 1: "P"},
        4: {0: "Al", 1: "Si"},
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

`site_species` can include as many state indices as the selected sites can
carry:

```python
site_species = {
    10: {0: "Na", 1: "Vacancy", 2: "Mg"},
    11: {0: "Na", 1: "Vacancy", 2: "Ca"},
}
```

Here `min_count=4` means "more than 3". The selected sites are
`event.local_env_indices`, and each selected `(site_index, occupation)` pair is
looked up in `site_species`.

```yaml
model_type: local_barrier
local_barrier:
  default_barrier: 300.0
  site_species:
    "1": {"0": P, "1": Si}
    "2": {"0": Si, "1": P}
    "3": {"0": Si, "1": P}
    "4": {"0": Al, "1": Si}
  rules:
    - name: si_rich
      type: species_count
      sites: local_env
      species: Si
      min_count: 4
      barrier: 420.0
```

Example: count either Si or Al as framework-blocking species:

```python
model.add_species_count_rule(
    name="framework_rich",
    sites="local_env",
    species=["Si", "Al"],
    min_count=4,
    barrier=500.0,
)
```

Example: combine species rules with a fallback:

```python
model = LocalBarrierModel(
    default_barrier=300.0,
    site_species=site_species,
)
model.add_species_count_rule(
    name="si_rich",
    sites="local_env",
    species="Si",
    min_count=4,
    barrier=420.0,
)
model.add_species_count_rule(
    name="p_rich",
    sites="local_env",
    species="P",
    min_count=4,
    barrier=280.0,
)
```

If a counted site is missing from `site_species`, the model raises an error
instead of silently guessing a species.

## Pattern Rules

Use `pattern` for a small number of explicit occupation patterns. `*` is a
wildcard. Pattern entries can be binary aliases or explicit nonnegative state
indices. For many exact patterns, use exact rules instead.

```python
model.add_pattern_rule(
    name="pattern_1",
    sites="canonical",
    pattern=["occupied", "vacant", "*", "occupied"],
    barrier=250.0,
)
```

The pattern above is checked against canonical site order. With mobile-ion
sites `(0, 1)` and local-environment sites `(1, 2, 3)`, the pattern means:

```text
site 0: occupied
site 1: vacant
site 2: anything
site 3: occupied
```

Example: match only the two mobile-ion sites:

```python
model.add_pattern_rule(
    name="forward_template",
    sites="mobile_ion",
    pattern=["occupied", "vacant"],
    barrier=260.0,
)
```

Example: match a multicomponent local pattern:

```python
model.add_pattern_rule(
    name="state_two_pattern",
    sites="canonical",
    pattern=[2, "vacant", "*", 3],
    barrier=275.0,
)
```

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
            "occupations": [1, 0, 1, 0],
            "properties": {"barrier": 250.0},
        }
    ]
)
```

The `occupations` list must be in canonical order. For
`mobile_ion_indices=[0, 1]` and `local_env_indices=[1, 2, 3]`, that means the
occupation list corresponds to sites `[0, 1, 2, 3]`.

You can also capture an exact rule from an event and state snapshot:

```python
rule = LocalBarrierModel.entry_from_event_state(
    event=event,
    state=state,
    properties={"barrier": 250.0},
)
model = LocalBarrierModel(rules=[rule])
```

Example: add exact rules incrementally:

```python
model = LocalBarrierModel(default_barrier=300.0)
model.add_exact_rule(
    name="known_environment_0",
    mobile_ion_indices=[0, 1],
    local_env_indices=[1, 2, 3],
    occupations=[1, 0, 1, 0],
    barrier=250.0,
)
```

Duplicate exact entries for the same event sites and occupation pattern are
rejected. This makes accidental table collisions visible when building the
model.

## Building From A Rule List

For generated models, it is often simpler to build a list of rule dictionaries
and pass them to the constructor:

```python
rules = [
    {
        "name": "si_rich",
        "type": "species_count",
        "sites": "local_env",
        "species": "Si",
        "min_count": 4,
        "properties": {"barrier": 420.0},
    },
    {
        "name": "open_env",
        "type": "state_count",
        "sites": "local_env",
        "state": "vacant",
        "min_count": 3,
        "properties": {"barrier": 260.0},
    },
]

model = LocalBarrierModel(
    rules=rules,
    default_barrier=300.0,
    site_species=site_species,
)
```

This produces the same model as calling `add_species_count_rule(...)` and
`add_state_count_rule(...)`.

## Rule Type Tradeoffs

Use this section as a quick check after reading the detailed examples:

- `constant_barrier`: every event has the same barrier.
- `state_count`: the rule only cares about counts of one occupation state.
- `species_count`: the rule cares about chemical labels such as Si, Al, or P.
- `pattern`: a small number of wildcard occupation patterns are enough.
- `exact`: each event/environment pattern has its own known barrier.

Prefer count rules over exact rules when the physical criterion is naturally a
count. Prefer exact rules when you are importing a table of precomputed
barriers.

## Loading And Saving

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

Load the model directly:

```python
from kmcpy.models import LocalBarrierModel

model = LocalBarrierModel.from_file("model.json")
```

Or let the generic model loader dispatch from the simulation config:

```python
from kmcpy.models import BaseModel
from kmcpy.simulator.config import Configuration

config = Configuration(
    structure_file="structure.cif",
    event_file="events.json",
    model_file="model.json",
)
model = BaseModel.from_config(config)
```

## Common Errors

`No local barrier rule matched`
: Add a `default_barrier`, or add a rule that covers the event and occupation
  pattern in the error message.

`Pattern rule ... has length ...`
: The pattern length must match the number of selected sites. Check the `sites`
  selector and the event's `mobile_ion_indices`/`local_env_indices`.

`species_count rules require site_species`
: Add a `site_species` entry for every site that the rule may count, including
  the occupation states that can occur there.

`Duplicate exact local-barrier rule`
: Two exact rules describe the same event sites and occupation pattern. Remove
  one or merge the data before building the model.
