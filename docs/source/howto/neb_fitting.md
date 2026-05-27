# NEB Fitting Inputs

Use `NEBDataLoader` when NEB barriers come from a list of structure files. The
loader maps each structure to the reference local lattice, computes the
correlation matrix, and writes the text files consumed by
`LocalClusterExpansion.fit`.

```python
from kmcpy.io.neb import NEBDataLoader
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
from kmcpy.structure.local_lattice_structure import LocalLatticeStructure

model = LocalClusterExpansion.from_json("lce.json")

reference = LocalLatticeStructure(
    template_structure=reference_structure,
    center=0,
    cutoff=4.0,
    site_mapping={
        "Na": ["Na", "X"],
        "Zr": "Zr",
        "Si": ["Si", "P"],
        "O": "O",
    },
    basis_type="chebyshev",
)

loader = NEBDataLoader.from_structures(
    ["neb_0001.cif", "neb_0002.cif"],
    [120.5, 140.1],
    model=model,
    reference_local_lattice_structure=reference,
)

fit_inputs = loader.write_fitting_inputs("fit_data", weight=1.0)

model_params, y_pred, y_true = LocalClusterExpansion().fit(
    alpha=1.5,
    max_iter=1_000_000,
    **fit_inputs,
)
```

If the model was just built in memory with `LocalClusterExpansion.build(...)`,
the reference local lattice is already attached to the model and
`reference_local_lattice_structure` can be omitted.

With `basis_type="chebyshev"`, kMCpy chooses the site basis from the number of
allowed species on each active site. Occupations are stored as discrete
species-state indices: `{"Na": ["Na", "X"]}` gives `Na=0` and `X=1`.
Mappings with more than two species use the same convention and add `q - 1`
Chebyshev basis functions for a site with `q` allowed species. For example,
`{"Al": ["Al", "X", "Mg", "Si"]}` stores `Al=0`, `X=1`, `Mg=2`, and `Si=3`,
and contributes three non-constant site basis functions to decorated LCE
cluster features.
