# Enumerate Local Environments

Use local-environment enumeration when you need ordered structures for fitting,
testing, or preparing NEB calculations around one hop.

The enumeration helpers start from a `LatticeStructure`, vary selected active
sites, and return:

- pymatgen structures,
- compact active-site occupation vectors,
- local occupation vectors in the same order used by kMCpy models.

## Build A Lattice Model

```python
from pymatgen.core import Lattice, Structure

from kmcpy.structure import LatticeStructure

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
```

`site_mapping` defines which species or vacancy states each active site can
take. Fixed species, such as `"Cl": "Cl"`, are not enumerated.

## Enumerate A Local Environment

```python
from kmcpy.structure import enumerate_local_environments

environments = enumerate_local_environments(
    lattice_model,
    center=0,
    cutoff=2.5,
    variable_species=["Si", "P"],
    variable_site_indices=[1, 2],
    species_counts={"Si": 1, "P": 1},
)

for environment in environments:
    print(environment.label)
    print(environment.full_occupation.values)
```

`species_counts` is an exact count over the variable sites. In this example,
the two variable sites must contain one `Si` and one `P`.

## Occupation Values

With `basis_type="chebyshev"`, kMCpy stores allowed species as integer state
indices. For example:

```python
site_mapping = {"Si": ["Si", "P", "Ge"]}
```

uses:

```text
Si = 0
P  = 1
Ge = 2
```

The LCE basis then uses `q - 1` non-constant basis functions for a site with
`q` allowed states.

## Use Pymatgen Transformations

For larger or more complex disordered environments, pass a pymatgen
transformation object. kMCpy calls `apply_transformation(...)` and normalizes
the result back into the same local-environment result type.

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

`EnumerateStructureTransformation` can also be used, but it depends on the
external enumlib executables used by pymatgen. kMCpy does not install those
binaries.

## Build NEB Endpoint Pairs

Use `enumerate_neb_endpoint_pairs` when you need initial and final structures
for one known hop in a small cell:

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

`mobile_ion_indices` are compact active-site indices. You can also pass a kMCpy
event object if it has a `mobile_ion_indices` attribute.

The helper returns structures only. It does not write VASP inputs, create NEB
image directories, or generate interpolated intermediate images.

## Checklist

- Make sure `variable_site_indices` use compact active-site indices.
- Use `species_counts` when composition must be fixed.
- Use the same site order and basis convention as the model you plan to fit.
- Keep enumlib-dependent transformations optional in reproducible workflows.
