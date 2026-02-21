
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

Kinetic Monte Carlo Simulation with Python (kMCpy) is an open-source Python package for studying ion diffusion using the kinetic Monte Carlo (kMC) technique. It offers a comprehensive Python-based approach to compute kinetic properties, suitable for research, development, and prediction of new functional materials.

Key features include a local cluster expansion model toolkit, a rejection-free kinetic Monte Carlo (rf-kMC) solver, and tools to extract ion transport properties like diffusivities and conductivities. The local cluster expansion model toolkit facilitates model fitting from ab initio or empirical barrier calculations. Post-training, the local cluster expansion model can compute migration barriers in crystalline materials within the transition state theory.

Advantages of using kMCpy:

1.  Written entirely in Python with a modular design, promoting developer-centricity and easy feature addition.
2.  Cross-platform compatibility, supporting Windows, macOS, and Linux.
3.  Performance-optimized kMC routines using [Numba](https://numba.pydata.org/), resulting in significant speed improvements.

> [!warning] kMCpy is under active development 
> kMCpy is still under active development. While we strive to maintain backward compatibility, some changes may occur that could affect existing workflows. We recommend users to check the release notes and documentation for any updates or changes that might impact their usage.

## Installation

### Method 1: Install using `pip` (recommended)
You can quickly install the latest version of kMCpy through [PyPI](https://pypi.org/project/kmcpy/) to your environment.

```shell
pip install kmcpy
```
> [!note] Virtual Environment
> It is highly recommended to install kMCpy in a virtual environment to avoid dependency conflicts with other packages. You can use [uv](https://docs.astral.sh/uv/getting-started/installation/) or [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to create a virtual environment.
> For example, using `venv`:
> ```shell
> python -m venv kmcpy-env
> source kmcpy-env/bin/activate  # On Windows use `kmcpy-env\Scripts\activate.bat`
> ```
> Then you can install kMCpy in the virtual environment using `pip install kmcpy`.
> To deactivate the virtual environment, you can use the command `deactivate`.
> For `conda`, you should also use `pip install kmcpy` to install `kMCpy` after activating the conda environment.
> For `uv`, you can use `uv pip install kmcpy` to install `kMCpy` after creating and activating the virtual environment.

### Method 2: Install from source

You can install kMCpy from source using either `pip` or [UV](https://docs.astral.sh/uv/getting-started/installation/). First, clone the repository and navigate to its root directory.

#### Using pip

To install normally:
```shell
pip install .
```

For development (editable mode):
```shell
pip install -e ".[dev]"
```

#### Using UV (recommended)

To install all dependencies:
```shell
uv sync
```

For development (editable mode):
```shell
uv sync --extra dev
uv pip install -e .
```

> [!warning] Windows users 
> Windows users (not applicable to WSL) need to install [Microsoft C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) for `pymatgen`. 


## Build documentation
You can access the documentation at [https://kmcpy.readthedocs.io/](https://kmcpy.readthedocs.io/). However, if you want to build the documentation locally, you can do so by following these steps:
```shell
uv sync --extra doc
python scripts/build_doc.py
```

## Quickstart
Run a minimal end-to-end simulation with bundled example files:

```shell
uv sync
uv run python -c "from kmcpy.simulator.config import SimulationConfig; SimulationConfig.help_parameters()"
uv run python example/minimal_example.py
```

`SimulationConfig` routes arguments into two groups:

1. `system` parameters define what you simulate (structure, events, model files).
2. `runtime` parameters define how you simulate (temperature, passes, random seed).

If you pass an unknown keyword, kMCpy raises a clear error and points to `SimulationConfig.help_parameters()`.

## Run kMCpy
### API usage
You can run kMC through API. See the `example` directory for scripts and notebook workflows covering setup, event generation, and simulations.

You can also attach custom property callbacks during a run:

```python
from kmcpy.simulator.kmc import KMC

kmc = KMC.from_config(config)

def custom_property(state, step, sim_time):
    occupied = sum(1 for occ in state.occupations if occ > 0)
    return occupied / len(state.occupations)

kmc.attach(custom_property, interval=100, name="occupied_fraction")
kmc.disable_property("conductivity")  # Optional: disable selected built-in fields
tracker = kmc.run(config)

# Stored custom callback records
records = tracker.get_custom_results("occupied_fraction")
```

### Command line usage
A wrapper is provided if you want to run kMCpy through command line only. There is a wrapper script `run_kmc` that allows you to run kMCpy from the command line. You can use it to run a kMCpy simulation with a JSON/YAML input file. The input file should contain the necessary parameters for the simulation. It should be noted that you need to have all the input files that needed to run kMC.
```shell
run_kmc input.json
```

To print out all arguments, you can run:
```shell
run_kmc --help
```

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
