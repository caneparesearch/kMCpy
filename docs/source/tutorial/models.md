# Choose A Model

Every kMC event needs a rate. In kMCpy, rates come from a model. Choose the
smallest model that represents the physics you need.

| Model | Use When | Data Needed |
|---|---|---|
| `LocalBarrierModel` | Barriers can be written as explicit local rules. | Constants, count rules, wildcard rules, or exact local environments. |
| `LocalClusterExpansion` | Barriers or site-energy differences should be fitted from local-environment data. | NEB or other calculated targets and matching local structures. |
| `CompositeLCEModel` | You have an LCE for `E_KRA` and optionally another model for site-energy differences. | A fitted KRA LCE and optional site model. |
| `SiteEnergyModel` | A callable or external code supplies site-energy differences. | A runtime object or callable plus site/state mappings if needed. |

## LocalBarrierModel

Use `LocalBarrierModel` when the rule is direct and readable:

```python
from kmcpy.models import LocalBarrierModel

model = LocalBarrierModel.constant_barrier(300.0)  # meV
model.to("local_barrier_model.json")
```

Rules are evaluated in order. The first match provides the barrier. A common
pattern is to put specific rules first and a default barrier last:

```python
from kmcpy.models import LocalBarrierModel

model = LocalBarrierModel(
    default_barrier=350.0,
    rules=[
        {
            "type": "species_count",
            "species": "Si",
            "min_count": 3,
            "barrier": 420.0,
        },
    ],
)
```

This is transparent and easy to debug, but it does not interpolate beyond the
rules you write.

## LocalClusterExpansion

A `LocalClusterExpansion` maps a local occupation vector around an event to a
scalar:

$$
y(\sigma) = E_0 + \sum_j \alpha_j \Phi_j(\sigma)
$$

where `sigma` is the ordered local occupation vector, `Phi_j` are cluster
features, and `alpha_j` are fitted coefficients.

The same `LocalClusterExpansion.compute(...)` method is used for any fitted
scalar. Its meaning comes from where the LCE is used:

- as `kra_model`, it returns `E_KRA`;
- as `site_model`, it returns the fitted site-energy-difference contribution for
  the canonical event orientation.

This avoids separate user-facing compute APIs for barrier and site terms.

## CompositeLCEModel

For transition-state-theory rates, kMCpy can combine a KRA model and a
site-energy-difference model:

```python
from kmcpy.models import CompositeLCEModel

model = CompositeLCEModel(
    kra_model=kra_lce,
    site_model=site_lce,  # optional
)
model.to("model.json")
```

The effective barrier is

$$
E_{\mathrm{eff}} = E_{\mathrm{KRA}} + \frac{\Delta E_{\mathrm{site}}}{2}
$$

where the site model provides the signed event energy change.

## SiteEnergyModel

Use `SiteEnergyModel` when an external runtime, such as a cluster-expansion
evaluator, supplies only the site-energy difference. kMCpy initializes the
external occupation once, precomputes array-backed site/state mappings, and then
passes only the two changed endpoints to the callable for each proposed event.

This is the right direction when a full occupation conversion at every kMC step
would be too slow.
