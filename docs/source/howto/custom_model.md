# Write A Custom Model

Most users should start with `LocalBarrierModel`, `LocalClusterExpansion`, or
`SiteEnergyModel`. Write a custom model only when those do not represent the
scientific quantity you need.

## What kMC Needs

The simulator needs a rate model with:

```python
compute_probability(event=event, runtime_config=runtime_config, simulation_state=state)
```

The method should return an event rate in Hz. It can use the current
occupations from `simulation_state.occupations` and the event endpoints from
`event.mobile_ion_indices`.

## Minimal Example

```python
from kmcpy.models import BaseModel
from kmcpy.units import BOLTZMANN_CONSTANT_MEV_PER_K


class ConstantRateBarrierModel(BaseModel):
    def __init__(self, barrier_mev, name="ConstantRateBarrierModel"):
        super().__init__(name=name)
        self.barrier_mev = float(barrier_mev)

    def compute_probability(self, event, runtime_config, simulation_state):
        import numpy as np

        return runtime_config.attempt_frequency * np.exp(
            -self.barrier_mev
            / (BOLTZMANN_CONSTANT_MEV_PER_K * runtime_config.temperature)
        )

    def as_dict(self):
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "barrier_mev": self.barrier_mev,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            barrier_mev=data["barrier_mev"],
            name=data.get("name", "ConstantRateBarrierModel"),
        )
```

This is intentionally small. Avoid adding a new abstraction unless multiple
models truly share meaningful logic.

## Optional Hooks

Stateful models can implement:

```python
initialize_state(simulation_state=state, event_lib=event_lib, structure=structure, config=config)
apply_event(event=event, simulation_state=state)
```

Use these hooks to build caches once and update them after accepted events. Do
not rebuild full external occupations inside every `compute_probability(...)`
call.

## Serialization

kMCpy follows Monty-style serialization:

- `as_dict()` returns structured data,
- `from_dict(...)` restores the object,
- `to("model.json")` writes the model,
- `from_file("model.json")` loads it.

For reusable public models, add tests that cover serialization, rate
calculation, and integration with `KMC.from_config(...)`.
