# Tracker And Output Files

[`KMC`](../modules/kmc.rst) runs the algorithm.
[`State`](../modules/state.rst) owns the mutable occupations.
[`Tracker`](../modules/tracker.rst) observes the trajectory and writes output.

## Built-In Outputs

After a normal run, kMCpy writes files such as:

```text
results_<label>.csv.gz
results_units_<label>.json.gz
current_occ_<label>.csv.gz
displacement_<label>.csv.gz
hop_counter_<label>.csv.gz
```

The results table contains transport summaries sampled during the run. Units
are written in the `results_units` sidecar and are also available in Python:

```python
tracker = run(config)
print(tracker.result_units)
print(tracker.return_current_info())
```

`return_current_info()` returns:

```text
(time, msd, jump_diffusivity, tracer_diffusivity, conductivity, havens_ratio, correlation_factor)
```

## Attach Custom Properties

Use custom properties when you need to sample a quantity during the simulation:

```python
from kmcpy import KMC


def vacancy_count(state, step, sim_time):
    return sum(value == 1 for value in state.occupations)

kmc = KMC.from_config(config)
kmc.attach(
    vacancy_count,
    name="vacancy_count",
    interval=10,
    store=True,
)
tracker = kmc.run()
```

For production workflows, configure property callbacks through
`Configuration.property_callbacks` so the run remains reproducible from the
input file.

## Output Ownership

The tracker owns sampled records and trajectory summaries. It reads the final
occupations from its `State`; callers do not need to pass final occupations back
into `write_results()`.

This keeps the workflow simple:

```python
tracker = run(config)
tracker.write_results(label="extra_label")
```

For advanced property scheduling and callback serialization, see
[Attach custom properties](../howto/attach_properties.md).

Next: [Analyze Results](analysis.md).
