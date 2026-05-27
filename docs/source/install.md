# Installation

## Install With `pip`

Install the latest PyPI release directly into a Python environment:

```shell
pip install kmcpy
```

Using a virtual environment is recommended:

```shell
python -m venv kmcpy-env
source kmcpy-env/bin/activate  # On Windows use `kmcpy-env\Scripts\activate`
pip install kmcpy
```

## Install With `uv pip`

If you manage Python environments with [uv](https://docs.astral.sh/uv/), use
the same PyPI package:

```shell
uv venv kmcpy-env
source kmcpy-env/bin/activate
uv pip install kmcpy
```

## Install With Conda

Use Conda to create the Python environment, then install the PyPI package
inside that environment:

```shell
conda create -n kmcpy python=3.11 pip
conda activate kmcpy
python -m pip install kmcpy
```

## Install From Source

Clone the repository, navigate to its root directory, and install with `pip`:

```shell
pip install .
```

For development with `pip`:

```shell
pip install -e ".[dev]"
```

For development with `uv`:

```shell
uv sync
uv sync --extra dev
uv pip install -e . # this makes the installation using the editable mode
```

> **⚠️ Warning for Windows users:**  
> You need to install [Microsoft C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) to compile `pymatgen`.


## Build documentation
You can access the documentation at [https://kmcpy.readthedocs.io/](https://kmcpy.readthedocs.io/). However, if you want to build the documentation locally, you can do so by following these steps:
```shell
uv sync --extra doc
python scripts/build_doc.py
```
