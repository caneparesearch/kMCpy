# Local Environment Enumeration

Use `enumerate_local_environments` to build ordered local configurations from a
`LatticeStructure`. The function returns pymatgen structures together with the
compact active-site and local occupation vectors, so the result can be passed directly into
local-cluster-expansion or NEB preparation workflows.

```python
from pymatgen.core import Lattice, Structure

from kmcpy.structure import LatticeStructure, enumerate_local_environments

lattice = Lattice.cubic(10.0)
template = Structure(
    lattice,
    ["Na", "Si", "Si", "Cl"],
    [[0, 0, 0], [1, 0, 0], [2, 0, 0], [5, 0, 0]],
    coords_are_cartesian=True,
)

lattice_model = LatticeStructure(
    template_structure=template,
    site_mapping={
        "Na": ["Na", "X"],
        "Si": ["Si", "P"],
        "Cl": "Cl",
    },
)

environments = enumerate_local_environments(
    lattice_model,
    center=0,
    cutoff=2.5,
    variable_species=["Si", "P"],
    variable_site_indices=[1, 2],
    species_counts={"Si": 1, "P": 1},
)

for environment in environments:
    print(environment.label, environment.full_occupation.values)
```

`species_counts` is an exact count over the variable sites. With
`basis_type="chebyshev"`, the first species in a site mapping is `-1`, while the
second species or vacancy is `+1`. For example, `{"Si": ["Si", "P"]}` gives
`Si=-1` and `P=+1`.

## Pymatgen Transformations

For larger disordered local environments, pass a pymatgen transformation object
that was configured by the caller. kMCpy calls `apply_transformation(...)` and
normalizes the ordered structures back into the same enumeration result type.

```python
from pymatgen.transformations.standard_transformations import (
    OrderDisorderedStructureTransformation,
)

transformation = OrderDisorderedStructureTransformation(no_oxi_states=True)

environments = enumerate_local_environments(
    lattice_model,
    center=0,
    cutoff=2.5,
    variable_species=["Si", "P"],
    variable_site_indices=[1, 2],
    species_counts={"Si": 1, "P": 1},
    transformation=transformation,
    max_results=10,
)
```

`EnumerateStructureTransformation` can also be passed in, but it depends on the
external enumlib executables used by pymatgen. kMCpy does not install or invoke
those binaries itself.

## NEB Endpoint Pairs

Use `enumerate_neb_endpoint_pairs` when preparing NEB calculations for one
known hop in a small cell. The helper combines local-environment enumeration
with initial/final endpoint construction.

```python
from kmcpy.structure import enumerate_neb_endpoint_pairs

endpoint_pairs = enumerate_neb_endpoint_pairs(
    lattice_model,
    mobile_ion_indices=(1, 2),
    cutoff=2.5,
    variable_species=["Na", "X"],
    variable_site_indices=[1, 2],
    species_counts={"Na": 1, "X": 1},
)

for pair in endpoint_pairs:
    initial_structure = pair.initial
    final_structure = pair.final
```

`mobile_ion_indices` is the pair of compact active-site indices for the hop. You can
also pass an object produced by kMCpy event generation, as long as it has a
`mobile_ion_indices` attribute.

The endpoint helper returns structures only. It does not write VASP inputs,
create numbered NEB image directories, or generate interpolated intermediate
images.
