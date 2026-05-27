# Conda Packaging

This recipe builds a Conda package from the current checkout. It is useful for
local validation and as the starting point for a conda-forge feedstock recipe.

```shell
conda install -c conda-forge conda-build
conda build conda/recipe
conda create -n kmcpy-conda-test python=3.11
conda install -n kmcpy-conda-test --use-local kmcpy
conda run -n kmcpy-conda-test python -c "import kmcpy; print(kmcpy.__version__)"
```

Public user installs such as

```shell
conda install -c conda-forge kmcpy
```

require the package to be published to a Conda channel. For conda-forge, submit
the recipe to the feedstock after the PyPI source distribution is published and
replace the local `source.path` entry with the release source URL and SHA256.
