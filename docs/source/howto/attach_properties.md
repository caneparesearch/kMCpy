# Attach Custom Properties

Use attached properties when you want to measure something during a KMC run that
is not one of kMCpy's built-in transport properties.

Examples include:

- the fraction of occupied active sites,
- a phase/order parameter,
- a count of ions in a region,
- a diagnostic value from a custom model.

The `KMC` object owns the simulation. The returned `Tracker` owns sampled
property records and output writing.

## Minimal Example

```python
from kmcpy.simulator.kmc import KMC


def occupied_fraction(state, step, sim_time):
    occupied = sum(1 for occ in state.occupations if occ == 0)
    return occupied / len(state.occupations)


kmc = KMC.from_config(config)
kmc.attach(occupied_fraction, interval=100, name="occupied_fraction")

tracker = kmc.run()
records = tracker.get_property_records("occupied_fraction")
```

This samples `occupied_fraction(...)` every 100 production events.

## Callback Signature

Every attached property receives:

```python
def property_fn(state, step, sim_time):
    ...
```

- `state`: current mutable `State`
- `step`: production event index
- `sim_time`: current simulation time in seconds

Return a JSON-serializable value such as a number, string, list, or dictionary.

## Sampling Cadence

Set a cadence for one property:

```python
kmc.attach(occupied_fraction, interval=100)
```

or sample by simulation time:

```python
kmc.attach(occupied_fraction, time_interval=1e-8)
```

Set a default cadence for attached properties that do not provide their own:

```python
kmc.set_property_frequency(interval=200)
```

## Built-In Properties

kMCpy also samples built-in transport properties. These are enabled by default:

| Property | Unit |
| --- | --- |
| `msd` | Angstrom^2 |
| `jump_diffusivity` | cm^2/s |
| `tracer_diffusivity` | cm^2/s |
| `conductivity` | mS/cm |
| `havens_ratio` | dimensionless |
| `correlation_factor` | dimensionless |

Disable or enable a built-in property:

```python
kmc.set_property_enabled("conductivity", False)
kmc.set_property_enabled("conductivity", True)
```

Disabled built-ins remain in the legacy CSV schema with `NaN` values so old
post-processing scripts still see the same columns.

## Output Files

Built-in properties are written to:

```text
results_<label>.csv.gz
results_units_<label>.json.gz
```

Attached custom properties are written to:

```text
properties_<label>.json.gz
```

You can also read them from the returned tracker:

```python
records = tracker.get_property_records("occupied_fraction")
units = tracker.result_units
```

## Configure From YAML

Runtime property settings can be placed in the input file:

```yaml
kmc:
  type: default
  default:
    property_sampling_interval: 200
    property_sampling_time_interval: null
    builtin_property_enabled:
      conductivity: false
    property_callbacks:
      - callable: "myproject.kmc_props:occupied_fraction"
        name: "occupied_fraction"
        interval: 100
        enabled: true
```

Callbacks loaded from YAML must be importable from a string reference.

## Error Handling

By default, an exception in a property callback stops the simulation. Provide an
error handler when a diagnostic property should not make the run fail:

```python
def on_error(exc, state, step, sim_time):
    print(f"occupied_fraction failed at step={step}: {exc}")
    return True  # continue the simulation


kmc.attach(occupied_fraction, interval=100, on_error=on_error)
```

Return `True` to continue. Return `False` or re-raise to stop.

## Checklist

- Keep callbacks cheap; they run inside the KMC loop.
- Do not mutate `state` inside a property callback.
- Return JSON-serializable values.
- Use explicit `name=...` values for stable output keys.
- Use `on_error` only for non-critical diagnostics.
