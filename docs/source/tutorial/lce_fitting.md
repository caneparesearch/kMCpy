# Fit A Local Cluster Expansion

This page explains the structure of an LCE and how to fit it from local
environment data.

## LCE Objects

An LCE starts from a `LocalLatticeStructure`. This object defines:

- the local-environment center,
- the cutoff sphere,
- the local site order,
- the basis function used to encode species states.

```python
from pymatgen.core import Structure
from kmcpy.structure import LocalLatticeStructure
from kmcpy.models import LocalClusterExpansion

structure = Structure.from_file("nasicon.cif")
site_mapping = {"Na": ["Na", "X"], "Zr": "Zr", "Si": ["Si", "P"], "O": "O"}

local_lattice = LocalLatticeStructure(
    template_structure=structure,
    center=0,
    cutoff=4.0,
    site_mapping=site_mapping,
    basis_type="chebyshev",
    local_site_order="kmcpy_default",
)

lce = LocalClusterExpansion()
lce.build(local_lattice_structure=local_lattice, cutoff_cluster=[6.0, 6.0, 0.0])
```

`center` chooses the local-environment origin. `local_site_order` only controls
how selected local sites are ordered and whether a real center site is excluded.

## Cluster, Orbit, Correlation

A cluster is a set of local sites: point, pair, triplet, or quadruplet. An orbit
is a group of symmetry-equivalent clusters. The correlation vector evaluates the
basis-decorated orbit functions for one local occupation.

The fitted scalar is

$$
y(\sigma) = E_0 + \sum_j \alpha_j \Phi_j(\sigma)
$$

where:

- `sigma` is the ordered local occupation vector,
- `Phi_j` is one decorated cluster-orbit feature,
- `alpha_j` is the fitted coefficient,
- `E_0` is the empty-cluster term.

For a multicomponent site with `q` allowed states, the Chebyshev basis uses
`q - 1` non-constant site functions. Cluster features are products of these site
functions, so multicomponent sites add more decorated features.

## Fit Parameters

After writing fitting inputs with `NEBDataLoader`, fit the coefficients:

```python
params, y_pred, y_true = lce.fit(
    alpha=1e-4,
    corr_fname="fit_kra/correlation_matrix.txt",
    ekra_fname="fit_kra/e_kra.txt",
    weight_fname="fit_kra/weight.txt",
    lce_params_fname="fit_kra/lce_params.json",
)

lce.set_parameters(params)
lce.to("kra_lce.json")
```

For a composite model, fit the KRA LCE and site-energy-difference LCE
separately, then combine them:

```python
from kmcpy.models import CompositeLCEModel

model = CompositeLCEModel(kra_model=kra_lce, site_model=site_lce)
model.to("model.json")
```

## Underfit And Overfit

NEB data is usually expensive, so the number of training structures is often
small compared with the number of possible local environments. Do not chase a
perfect training error without checking whether the model is physically useful.

Typical symptoms:

- Underfit: the model has too few active features or too strong regularization;
  both training RMSE and validation error are large.
- Overfit: training RMSE is very small, but leave-one-out or held-out error is
  large; the model is fitting noise or sparse sampling artifacts.

Some residual fitting error is normal for sparse NEB datasets. Prefer a stable
model with sensible errors and few active coefficients over a model that only
reproduces the training set.

## Practical Fitting Checks

Before using an LCE in kMC:

- confirm all training structures map to the expected local occupation length,
- inspect the correlation matrix shape,
- compare `y_true` and `y_pred`,
- inspect RMSE and LOOCV,
- keep the model, fitting parameters, local site order, and training data
  together.
