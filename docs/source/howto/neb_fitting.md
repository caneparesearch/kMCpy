# NEB Fitting Inputs

Use `NEBDataLoader` when NEB barriers come from a list of structure files. The
loader maps each structure to the reference local lattice, computes the
correlation matrix, and writes the text files consumed by
`LocalClusterExpansion.fit`.

```python
from kmcpy.io import NEBDataLoader
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
from kmcpy.structure.local_lattice_structure import LocalLatticeStructure

model = LocalClusterExpansion.from_json("lce.json")

reference = LocalLatticeStructure(
    template_structure=reference_structure,
    center=0,
    cutoff=4.0,
    specie_site_mapping={
        "Na": ["Na", "X"],
        "Zr": "Zr",
        "Si": ["Si", "P"],
        "O": "O",
    },
    basis_type="chebyshev",
    exclude_species=["O2-", "O", "Zr4+", "Zr"],
)

loader = NEBDataLoader.from_structure_files(
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

With `basis_type="chebyshev"`, kMCpy uses the binary site-state convention:
the first species in each mapping is `-1`, and the second or missing state is
`+1`. For example, `{"Na": ["Na", "X"], "Si": ["Si", "P"]}` gives
`Na=-1`, `X=+1`, `Si=-1`, and `P=+1`.
