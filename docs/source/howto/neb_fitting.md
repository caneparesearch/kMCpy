# Prepare NEB Fitting Inputs

Use `NEBDataLoader` when you have NEB barriers for a list of structures and
want to fit a `LocalClusterExpansion`.

The loader does three things:

1. maps each NEB structure onto the reference local lattice,
2. computes the LCE correlation matrix,
3. writes the text files consumed by `LocalClusterExpansion.fit(...)`.

## Minimal Workflow

```python
from kmcpy.io.neb import NEBDataLoader
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
from kmcpy.structure.local_lattice_structure import LocalLatticeStructure

model = LocalClusterExpansion.from_file("lce.json")

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

model_params, y_pred, y_true = model.fit(
    alpha=1.5,
    max_iter=1_000_000,
    **fit_inputs,
)
```

The barrier values above are in `meV`.

## When The Reference Can Be Omitted

If the model was built in memory with `LocalClusterExpansion.build(...)`, the
reference local lattice is already attached to the model:

```python
model = LocalClusterExpansion()
model.build(reference, cutoff_cluster=[4.0, 4.0, 4.0])

loader = NEBDataLoader.from_structures(
    structure_files,
    barriers_mev,
    model=model,
)
```

If you load a model file and the local lattice is not attached, pass
`reference_local_lattice_structure` explicitly.

## Occupation And Basis Convention

With `basis_type="chebyshev"`, kMCpy stores occupations as discrete state
indices. For example:

```python
site_mapping = {"Na": ["Na", "X"]}
```

uses:

```text
Na = 0
X  = 1
```

For more than two allowed states:

```python
site_mapping = {"Al": ["Al", "X", "Mg", "Si"]}
```

kMCpy stores:

```text
Al = 0
X  = 1
Mg = 2
Si = 3
```

A site with `q` allowed states contributes `q - 1` non-constant Chebyshev basis
functions to decorated LCE cluster features.

## Output Files

`write_fitting_inputs(...)` returns paths suitable for `model.fit(...)`, such
as correlation, target barrier, and weight files. Keep these files with the
model and fitted parameter output so the fitting dataset is reproducible.

## Checklist

- Barrier targets should be in `meV`.
- The reference local lattice must match the LCE local site order.
- The structure files must map cleanly onto the reference lattice.
- Use the same `basis_type` and `site_mapping` used to build the model.
- Keep generated fitting inputs and fitted parameters together.
