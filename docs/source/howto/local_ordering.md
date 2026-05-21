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
    specie_site_mapping={"Na": ["Na", "X"], "Si": ["Si", "P"]},
    center=0,
    cutoff=5.0,
)
```

For NASICON publication reproduction, opt into the historical convention:

```python
local_lattice = LocalLatticeStructure(
    template_structure=structure,
    specie_site_mapping={"Na": ["Na", "X"], "Si": ["Si", "P"], "P": ["Si", "P"]},
    center=3,
    cutoff=5.0,
    exclude_species=["Zr4+", "O2-", "O", "Zr"],
    ordering_convention="nasicon_publication_v1",
)
```

`nasicon_publication_v1` uses the selected Na site as the geometric center,
removes that center site from the local occupation vector, and sorts the
remaining local sites by species and Cartesian `x` coordinate. This matches the
single-unit local environment convention used by the original NASICON code.

When an LCE is serialized, kMCpy includes the convention metadata and an
order-sensitive local environment hash. These fields make the fitted model's
feature ordering explicit:

```json
{
  "ordering_convention": {
    "name": "nasicon_publication_v1",
    "sort_keys": ["species", "cartesian_x"],
    "exclude_center_site": true
  },
  "local_environment_hash": "..."
}
```

For exact publication reproduction, prefer converted legacy artifacts because
they preserve the original `cluster_site_indices`. Regenerating local
correlation vectors with a different ordering convention and reusing old `keci`
values can change predicted barriers substantially.
