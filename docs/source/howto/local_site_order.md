# Control Local Site Order

Local cluster expansion coefficients depend on the order of the local
occupation vector. If the same physical sites are ordered differently, the
correlation-vector components no longer mean the same thing, and fitted
coefficients should not be reused.

Use this page when you need to reproduce an existing fitted model or compare
correlation vectors across workflows.

## Default Order

For new models, use the default order unless you need a named published order:

```python
from kmcpy.structure import LocalLatticeStructure

local_lattice = LocalLatticeStructure(
    template_structure=structure,
    site_mapping={"Na": ["Na", "X"], "Si": ["Si", "P"]},
    center=0,
    cutoff=5.0,
)
```

The chosen local site order is serialized with the LCE model, together with a local
environment hash.

## Use The NASICON 2022 Order

To reproduce the NASICON local environments from Deng et al.,
[*Fundamental investigations on the sodium-ion transport properties of mixed
polyanion solid-state battery electrolytes*](https://www.nature.com/articles/s41467-022-32190-7),
request the historical order explicitly:

```python
local_lattice = LocalLatticeStructure(
    template_structure=structure,
    site_mapping={
        "Na": ["Na", "X"],
        "Zr": "Zr",
        "Si": ["Si", "P"],
        "P": ["Si", "P"],
        "O": "O",
    },
    center=3,
    cutoff=5.0,
    local_site_order="nasicon_nat_commun_2022",
)
```

This order:

- uses the selected active-site Na as the geometric center,
- removes that center site from the local occupation vector,
- sorts the remaining local sites by species and Cartesian `x` coordinate.

That matches the single-unit local-environment order used by the original
NASICON workflow.

## Serialized Metadata

When an LCE is serialized, kMCpy records the local site order and an
order-sensitive local environment hash:

```json
{
  "local_site_order": {
    "name": "nasicon_nat_commun_2022",
    "sort_keys": ["species", "cartesian_x"],
    "exclude_center_site": true
  },
  "local_environment_hash": "..."
}
```

Fitted parameter files also keep the local environment hash. This lets kMCpy
detect when ECIs are being attached to a different ordered local environment.

## Practical Rule

Do not regenerate local correlation vectors with a different local site order
and reuse existing `keci` values. The model may run, but predicted barriers can
change because the coefficient order no longer matches the feature order.

## Checklist

- Use the default order for new models.
- Set `local_site_order` only when reproducing a specific workflow.
- Keep model files and fitted parameter files together.
- Treat a changed `local_environment_hash` as a sign that the fitted parameters
  no longer describe the same ordered local environment.
