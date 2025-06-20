# Installation
## Prerequisite
Check `pyproject.toml` for the required packages. The following Python packages are required to run kMCpy:
- pymatgen: for structure manipulation
- numba: for fast computation of kMCpy routines
- scikit-learn: for fitting local cluster expansion model
- pytest: for unit tests
- joblib
- glob2

```{admonition} For Windows users
:class: warning

You need to install [Microsoft C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) to compile `pymatgen`.
```

## Command line environment
### Method 1: Install from PyPI
You can quickly install the latest version of kMCpy through [PyPI](https://pypi.org/project/kmcpy/) to your environment.

```shell
pip install kmcpy
```

### Method 2: Install using UV and virtual environment
However, it is highly recommended to install kMCpy using [UV](https://docs.astral.sh/uv/getting-started/installation/) and use it with virtual environment.

```shell
uv venv #optional if you have already created a venv
source .venv/bin/activate
uv sync
```

### For developers 
```shell
uv venv #optional if you have already created a venv
source .venv/bin/activate
uv sync --extra dev
uv pip install -e . # this makes the installation using the editable mode
```

## Graphic user interface (GUI)
`wxpython` needs `conda` to be installed.
```shell
conda create -n kmcpy python wxpython -c conda-forge
conda activate kmcpy
pip install -r requirement_gui.txt .
```

## Build documentation
- Documentation is built using `pandoc` and `sphinx-build`.
- You can access the documentation from: `./docs/html/index.html`.
```shell
source .venv/bin/activate
uv sync --extra doc
python build_doc.py
```