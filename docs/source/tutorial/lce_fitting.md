# Fit A Local Cluster Expansion

This page starts after you have built the
[`LocalClusterExpansion`](../modules/local_cluster_expansion.rst) in
[Choose And Build A Model](models.md) and written fitting files with
[`NEBDataLoader`](../modules/neb.rst) in [Local Environments And NEB Data](local_environments_neb.md).

Fitting does not decide the local environment. It only finds coefficients for
the fixed feature order already defined by the LCE object.

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

After writing fitting inputs with `NEBDataLoader.write_fitting_inputs(...)`, fit
the coefficients:

```python
fit_files = loader.write_fitting_inputs(output_dir="fit_kra")

params, y_pred, y_true = kra_lce.fit(
    **fit_files,
    alpha=1e-4,
    lce_params_fname="fit_kra/lce_params.json",
)

kra_lce.set_parameters(params)
kra_lce.to("kra_lce.json")
```

The important [`LocalClusterExpansion.fit(...)`](../modules/local_cluster_expansion.rst)
arguments are:

- `alpha`: Lasso regularization strength. Larger values usually produce fewer
  active coefficients.
- `corr_fname`: correlation matrix file from `NEBDataLoader`.
- `ekra_fname`: target-value file. The name is historical; the values can be
  `E_KRA` or another fitted scalar as long as the model usage is consistent.
- `weight_fname`: sample weights, one per target value.
- `lce_params_fname`: output JSON for fitted coefficients and metadata.
- `max_iter`: maximum Lasso iterations.

`fit(...)` returns:

- `params`: fitted LCE parameters.
- `y_pred`: model predictions for the training rows.
- `y_true`: target values loaded from `ekra_fname`.

Call `set_parameters(params)` before saving or using the LCE in kMC.

For a composite model, fit the KRA LCE and site-energy-difference model
separately, then combine them:

```python
from kmcpy.models import CompositeLCEModel

model = CompositeLCEModel(kra_model=kra_lce, site_model=site_lce)
model.to("model.json")
```

If you only have a KRA model, omit `site_model`.

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

Next: [Prepare Input And Run kMC](run_kmc.md).
