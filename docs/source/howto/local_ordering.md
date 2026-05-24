# Local Site Ordering

Local cluster expansion (LCE) coefficients are tied to the order of the local
occupation vector used during fitting. If the same physical sites are ordered
differently, the correlation vector components change meaning and old fitted
coefficients should not be reused.

`LocalLatticeStructure` records this rule as an ordering convention. The default
is unchanged:

```python
from kmcpy.structure import LocalLatticeStructure

local_lattice = LocalLatticeStructure(
    template_structure=structure,
    site_mapping={"Na": ["Na", "X"], "Si": ["Si", "P"]},
    center=0,
    cutoff=5.0,
)
```

For reproducing the NASICON results from Deng et al.,
[*Fundamental investigations on the sodium-ion transport properties of mixed
polyanion solid-state battery electrolytes*](https://www.nature.com/articles/s41467-022-32190-7),
opt into the historical convention:

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
    ordering_convention="nasicon_nat_commun_2022",
)
```

`nasicon_nat_commun_2022` uses the selected active-site Na as the geometric center,
removes that center site from the local occupation vector, and sorts the
remaining local sites by species and Cartesian `x` coordinate. This matches the
single-unit local environment convention used by the original NASICON code.

When an LCE is serialized, kMCpy includes the convention metadata and an
order-sensitive local environment hash. These fields make the fitted model's
feature ordering explicit:

```json
{
  "ordering_convention": {
    "name": "nasicon_nat_commun_2022",
    "sort_keys": ["species", "cartesian_x"],
    "exclude_center_site": true
  },
  "local_environment_hash": "..."
}
```

For exact reproduction of that paper's model inputs, prefer converted legacy
artifacts because they preserve the original `cluster_site_indices`.
Regenerating local correlation vectors with a different ordering convention and
reusing old `keci` values can change predicted barriers substantially.
