# Installation

## Method 1: Install using `pip`
You can quickly install the latest version of kMCpy through [PyPI](https://pypi.org/project/kmcpy/) to your environment.

```shell
pip install kmcpy
```

## Method 2: Install from source using `pip`

You can install from the source code using `pip`. Assuming you have cloned the repository, navigate to the root directory of the kMCpy repository and run:
```shell
pip install .
```
For development, you can clone the repository and install it in editable mode using 

```shell
pip install -e ".[dev]"
```
This allows you to modify the source code and see changes immediately without reinstalling.

kMCpy also has a basic graphical user interface (GUI). It is based on`wxpython`. You might need to install [GTK](https://www.gtk.org/) for `wxpython`. You can install other additional dependencies for the GUI by running:
```shell
pip install -e ".[gui]"
```

## Method 3: Install from source using [UV](https://docs.astral.sh/uv/getting-started/installation/)
It is highly recommended to install kMCpy from source using [UV](https://docs.astral.sh/uv/getting-started/installation/) and use it with virtual environment.
```shell
uv sync
```
For development, you can install it in editable mode using:
```shell
uv sync --extra dev
uv pip install -e . # this makes the installation using the editable mode
```
For GUI, you can install the additional dependencies by running:
```shell
uv sync --extra gui
```

> **⚠️ Warning for Windows users:**  
> You need to install [Microsoft C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) to compile `pymatgen`.


## Build documentation
You can access the documentation at [https://kmcpy.readthedocs.io/](https://kmcpy.readthedocs.io/). However, if you want to build the documentation locally, you can do so by following these steps:
```shell
uv sync --extra doc
python scripts/build_doc.py
```