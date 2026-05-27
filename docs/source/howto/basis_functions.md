# Choose Basis Functions

Basis functions convert occupation states into numerical features for local
cluster expansion.

## Occupation State Indices

kMCpy stores active-site occupations as state indices. For a site with allowed
species:

```python
["Na", "X"]
```

state `0` means `Na` and state `1` means vacancy `X`.

For:

```python
["Si", "P", "Al"]
```

state `0` means `Si`, state `1` means `P`, and state `2` means `Al`.

The allowed species order in `site_mapping` therefore matters.

## Chebyshev Basis

For a site with `q` allowed species, kMCpy uses `q - 1` non-constant Chebyshev
site functions. A cluster feature is a product of the selected site functions
over the sites in that cluster.

This means multicomponent sites naturally create more decorated features:

| Site States | Non-Constant Site Functions |
|---|---|
| 2 | 1 |
| 3 | 2 |
| 4 | 3 |

A pair of two four-state sites can therefore contribute up to `3 x 3`
decorated pair features.

## Cost

More species per site increases the number of decorated cluster features. The
largest cost usually appears during correlation-matrix construction and fitting,
not during the accepted-hop update itself.

Keep the local cutoff and cluster cutoffs physically motivated. A larger basis
is useful only if the training data can constrain it.

## Practical Rules

- Use `basis_type="chebyshev"` for multicomponent active sites.
- Keep the same `site_mapping` species order throughout fitting and kMC.
- Refit if you change the basis, local site order, cutoff, or allowed species.
- Check the correlation matrix shape before fitting.
