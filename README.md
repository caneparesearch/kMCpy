
<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/caneparesearch/kMCpy/master/docs/source/_static/kmcpy_logo_dark.svg">
    <img alt="Logo" src="https://raw.githubusercontent.com/caneparesearch/kMCpy/master/docs/source/_static/kmcpy_logo.svg" height="120">
  </picture>
</h1>

[![GitHub release](https://img.shields.io/github/release/caneparesearch/kmcpy.svg)](https://GitHub.com/caneparesearch/kmcpy/releases/)
[![Documentation Status](https://readthedocs.org/projects/kmcpy/badge/)](https://kmcpy.readthedocs.io/en/latest/)
[![CI Status](https://github.com/caneparesearch/kmcpy/actions/workflows/test-ubuntu.yml/badge.svg)](https://github.com/caneparesearch/kmcpy/actions/workflows/test-ubuntu.yml)
[![PyPI Downloads](https://img.shields.io/pypi/dm/kmcpy?logo=pypi&logoColor=white&color=blue&label=PyPI)](https://pypi.org/project/kmcpy)
[![Requires Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![Paper](https://img.shields.io/badge/Comp.Mater.Sci.-2023.112394-blue?logo=elsevier&logoColor=white)](https://doi.org/10.1016/j.commatsci.2023.112394)

kMCpy is an open-source Python package for studying ion diffusion using the kinetic Monte Carlo (kMC) technique. It offers a comprehensive Python-based approach to compute kinetic properties, suitable for research, development, and prediction of new functional materials.

Key features include a local cluster expansion model toolkit, a rejection-free kinetic Monte Carlo (rf-kMC) solver, and tools to extract ion transport properties like diffusivities and conductivities. The local cluster expansion model toolkit facilitates model fitting from ab initio or empirical barrier calculations. Post-training, the local cluster expansion model can compute migration barriers in crystalline materials within the transition state theory.

Advantages of using kMCpy:

1.  Written entirely in Python with a modular design, promoting developer-centricity and easy feature addition.
2.  Cross-platform compatibility, supporting Windows, macOS, and Linux.
3.  Performance-optimized kMC routines using [Numba](https://numba.pydata.org/), resulting in significant speed improvements.

This code was recently employed to investigate the transport properties of Na-ion in [NaSiCON solid electrolyte](https://www.nature.com/articles/s41467-022-32190-7). In this study, rf-kMC was used to model Na-ion conductivity in NaSiCON, leading to the discovery that maximum conductivity is achieved at Na = 3.4.

## Installation

### Method 1: Install using `pip` in a virtual environment (recommended)
You can quickly install the latest version of kMCpy through [PyPI](https://pypi.org/project/kmcpy/) to your environment.

```shell
pip install kmcpy
```
> **⚠️ Virtual Environment**  
> It is highly recommended to install kMCpy in a virtual environment to avoid dependency conflicts with other packages. You can use [uv](https://docs.astral.sh/uv/getting-started/installation/) or [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to create a virtual environment.
> For example, using `venv`:
> ```shell
> python -m venv kmcpy-env
> source kmcpy-env/bin/activate  # On Windows use `kmcpy-env\Scripts
> ```
> Then you can install kMCpy in the virtual environment using `pip install kmcpy`.
> To deactivate the virtual environment, you can use the command `deactivate`.
> For `conda`, you should also use `pip install kmcpy` to install `kMCpy` after activating the conda environment.
> For `uv`, you can use `uv pip install kmcpy` to install `kMCpy` after creating and activating the virtual environment.

### Method 2: Install from source using `pip`

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

### Method 3: Install from source using [UV](https://docs.astral.sh/uv/getting-started/installation/)
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

## Run kMCpy
### API usage
You can run kMC through API. You can find more details in the `examples` directory. You can see the examples in the `examples` directory for how to use kMCpy in your own scripts. The examples cover various aspects of kMCpy, including how to build a model and use it for simulations.

### Command line usage
A wrapper is provided if you want to run kMCpy through command line only. There is a wrapper script `run_kmc` that allows you to run kMCpy from the command line. You can use it to run a kMCpy simulation with a JSON/YAML input file. The input file should contain the necessary parameters for the simulation. It should be noted that you need to have all the input files that needed to run kMC.
```shell
run_kmc input.json
```

To print out all arguments, you can run:
```shell
run_kmc --help
```

### GUI usage
You can start the GUI from command line. The basic usage is as follows:
```shell
start_kmcpy_gui
```
Then  a window will pop up, allowing you to select the input file and run the simulation.

## Citation
If you use kMCpy in your research, please cite it as follows:

```bibtex
@article{deng2022fundamental,
          title={Fundamental investigations on the sodium-ion transport properties of mixed polyanion solid-state battery electrolytes},
          author={Deng, Zeyu and Mishra, Tara P and Mahayoni, Eunike and Ma, Qianli and Tieu, Aaron Jue Kang and Guillon, Olivier and Chotard, Jean-No{\"e}l and Seznec, Vincent and Cheetham, Anthony K and Masquelier, Christian and Gautam, Gopalakrishnan Sai and Canepa, Pieremanuele},
          journal={Nature Communications},
          volume={13},
          number={1},
          pages={1--14},
          year={2022},
          publisher={Nature Publishing Group}
        }
@article{deng2023kmcpy,
          title = {kMCpy: A python package to simulate transport properties in solids with kinetic Monte Carlo},
          journal = {Computational Materials Science},
          volume = {229},
          pages = {112394},
          year = {2023},
          issn = {0927-0256},
          doi = {https://doi.org/10.1016/j.commatsci.2023.112394},
          author = {Zeyu Deng and Tara P. Mishra and Weihang Xie and Daanyal Ahmed Saeed and Gopalakrishnan Sai Gautam and Pieremanuele Canepa},
          }
```