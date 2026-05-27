# Choose And Build A Model

Every kMC event needs a rate. In kMCpy, rates come from a model. Choose the
smallest model that represents the physics you need, then build that model
before preparing the final simulation input.

| Model | Use When | Data Needed |
|---|---|---|
| [`LocalBarrierModel`](../modules/local_barrier_model.rst) | Barriers can be written as explicit local rules. | Constants, count rules, wildcard rules, or exact local environments. |
| [`LocalClusterExpansion`](../modules/local_cluster_expansion.rst) | A local scalar should be fitted from local-environment data. | A reference local lattice, cluster cutoffs, and fitted coefficients from NEB or other target data. |
| [`CompositeLCEModel`](../modules/composite_lce_model.rst) | You have one model for `E_KRA` and optionally another model for site-energy differences. | A fitted KRA model and optional site-energy-difference model. |
| [`SiteEnergyModel`](../modules/site_energy.rst) | A callable or external code supplies site-energy differences. | A runtime object or callable plus site/state mappings. |

## Local Barrier Branch

Use `LocalBarrierModel` when the barrier rule is direct and readable. A constant
barrier is the smallest valid model:

```python
from kmcpy.models import LocalBarrierModel

model = LocalBarrierModel.constant_barrier(300.0)  # meV
model.to("local_barrier_model.json")
```

`constant_barrier(barrier)` takes one required argument:

- `barrier`: activation barrier in meV, used for every valid event.

Rules are evaluated in order. The first matching rule provides the barrier; if
no rule matches, `default_barrier` is used. For species-count rules, provide
`site_species` so kMCpy can translate each active-site occupation state into a
chemical label:

```python
from kmcpy.models import LocalBarrierModel

model = LocalBarrierModel(
    default_barrier=350.0,
    site_species={
        5: {0: "P", 1: "Si"},
        6: {0: "Si", 1: "P"},
        7: {0: "Al", 1: "Si"},
        8: {0: "Si", 1: "P"},
    },
)
model.add_species_count_rule(
    name="si_rich",
    sites="local_env",
    species="Si",
    min_count=3,
    barrier=420.0,
)
model.to("local_barrier_model.json")
```

The important arguments are:

- `default_barrier`: fallback barrier in meV.
- `site_species`: mapping `{active_site_index: {state_index: species_label}}`
  used by species-count rules.
- `sites`: which sites the rule counts. `"local_env"` counts the event local
  environment. `"canonical"` counts the two mobile-ion endpoints followed by
  the local-environment sites with duplicates removed.
- `min_count`, `max_count`, or `count`: the count condition.
- `barrier`: barrier in meV returned when the rule matches.

For exact-match tables, wildcard patterns, and more rule examples, see
[Use advanced local barrier rules](../howto/local_barrier_model.md).

## LCE Branch

Use `LocalClusterExpansion` when a local scalar should be fitted from local
environment data. The scalar can be `E_KRA`, a site-energy-difference term, or
another model quantity. The meaning comes from where the LCE is used later.

An LCE is built in two stages:

1. Build a [`LocalLatticeStructure`](../modules/local_lattice_structure.rst).
   This defines the local environment and the local site order.
2. Build the [`LocalClusterExpansion`](../modules/local_cluster_expansion.rst).
   This enumerates point, pair, triplet, and quadruplet clusters inside the
   local environment.

```python
from pymatgen.core import Structure

from kmcpy.models import LocalClusterExpansion
from kmcpy.structure import LocalLatticeStructure

structure = Structure.from_file("nasicon.cif")
site_mapping = {
    "Na": ["Na", "X"],
    "Zr": "Zr",
    "Si": ["Si", "P"],
    "O": "O",
}

local_lattice = LocalLatticeStructure(
    template_structure=structure,
    center=0,
    cutoff=4.0,
    site_mapping=site_mapping,
    basis_type="chebyshev",
    local_site_order="kmcpy_default",
)

kra_lce = LocalClusterExpansion()
kra_lce.build(
    local_lattice_structure=local_lattice,
    cutoff_cluster=[6.0, 6.0, 0.0],
)
```

The important `LocalLatticeStructure` arguments are:

- `template_structure`: structure containing every possible mobile-ion site.
- `center`: local-environment origin. It can be a structure site index or a
  fractional-coordinate point. If it is a real site, the local site order
  controls whether the center is included in the occupation vector.
- `cutoff`: radius used to select local-environment sites.
- `site_mapping`: allowed occupation states for active sites and fixed species
  for inactive sites.
- `basis_type`: occupation encoding. Use `"chebyshev"` for multicomponent
  active sites.
- `local_site_order`: deterministic order for the local occupation vector. See
  [Control local site order](../howto/local_site_order.md).

The important `LocalClusterExpansion.build(...)` arguments are:

- `local_lattice_structure`: the ordered local lattice just built.
- `cutoff_cluster`: `[pair, triplet, quadruplet]` cutoffs in angstrom. The LCE
  always includes point clusters; the list controls higher-body clusters.

The fitted scalar is

$$
y(\sigma) = E_0 + \sum_j \alpha_j \Phi_j(\sigma)
$$

where `sigma` is the ordered local occupation vector, `Phi_j` are decorated
cluster-orbit features, and `alpha_j` are fitted coefficients.

At this point the LCE has the correct feature order, but it has no fitted
coefficients. Continue to [Local Environments And NEB Data](local_environments_neb.md)
and [Fit A Local Cluster Expansion](lce_fitting.md).

## Composite LCE Models

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

The same `LocalClusterExpansion.compute(...)` method is used for any fitted
scalar:

- as `kra_model`, it returns `E_KRA`;
- as `site_model`, it returns the fitted site-energy-difference contribution for
  the canonical event orientation.

## External Site-Energy Models

Use `SiteEnergyModel` when an external runtime, such as a cluster-expansion
evaluator, supplies only the site-energy difference. kMCpy initializes the
external occupation once, precomputes array-backed site/state mappings, and then
passes only the changed endpoints to the callable for each proposed event.

See [Use site-energy-difference models](../howto/external_site_energy.md) for
the external-code workflow.
