# Kinetic Monte Carlo Simulation using Python (kMCpy)
![image](https://raw.githubusercontent.com/caneparesearch/kMCpy/master/docs/source/_static/kmcpy_logo.png)
- Author: Zeyu Deng
- Email: dengzeyu@gmail.com

kMCpy is an open-source Python package for studying atomic migration using the kinetic Monte Carlo technique. It offers a comprehensive Python-based approach to compute kinetic properties, suitable for research, development, and prediction of new functional materials.

Key features include a local cluster expansion model toolkit, a rejection-free kinetic Monte Carlo (rf-kMC) solver, and tools to extract ion transport properties like diffusivities and conductivities. The local cluster expansion model toolkit facilitates model fitting from ab initio or empirical barrier calculations. Post-training, the local cluster expansion model can compute migration barriers in crystalline materials within the transition state theory.

Advantages of using kMCpy:

1.  Written entirely in Python with a modular design, promoting developer-centricity and easy feature addition.
2.  Cross-platform compatibility, supporting Windows, macOS, and Linux.
3.  Performance-optimized kMC routines using [Numba](https://numba.pydata.org/), resulting in significant speed improvements.

This code was recently employed to investigate the transport properties of Na-ion in [NaSiCON solid electrolyte](https://www.nature.com/articles/s41467-022-32190-7). In this study, rf-kMC was used to model Na-ion conductivity in NaSiCON, leading to the discovery that maximum conductivity is achieved at Na = 3.4.

## Installation

### Command line environment
#### Method 1: Install using `pip`
You can quickly install the latest version of kMCpy through [PyPI](https://pypi.org/project/kmcpy/) to your environment.

```shell
pip install kmcpy
```

#### Method 2: Install from source using `pip`

You can install from the source code using `pip`. Assuming you have cloned the repository, navigate to the root directory of the kMCpy repository and run:
```shell
pip install .
```
#### Method 3: Install from source using [UV](https://docs.astral.sh/uv/getting-started/installation/)
It is highly recommended to install kMCpy from source using [UV](https://docs.astral.sh/uv/getting-started/installation/) and use it with virtual environment.
```shell
uv sync
```

For development, you can clone the repository and install it in editable mode using `pip install -e .`. This allows you to modify the source code and see changes immediately without reinstalling. For UV, you can do following:
```shell
uv sync --extra dev
uv pip install -e . # this makes the installation using the editable mode
```

> **⚠️ Warning for Windows users:**  
> You need to install [Microsoft C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) to compile `pymatgen`.

### Graphic user interface (GUI)
kMCpy also has a basic GUI. It is based on`wxpython`.
```shell
uv sync --extra gui
start_kmcpy_gui.py
```

## Build documentation
You can access the documentation at [https://kmcpy.readthedocs.io/](https://kmcpy.readthedocs.io/). However, if you want to build the documentation locally, you can do so by following these steps:
```shell
source .venv/bin/activate
uv sync --extra doc
python build_doc.py
```

## Run kMCpy
It is recommended to run kMCpy using the API. See examples for more details. A wrapper is also provided if you want to run kMCpy through command line only. 
You can see the examples in the `examples` directory for how to use kMCpy in your own scripts. The examples cover various aspects of kMCpy, including how to build a model and use it for simulations.

### Command line usage
There is a wrapper script `run_kmc` that allows you to run kMCpy from the command line. You can use it to run a kMCpy simulation with a JSON input file. The input file should contain the necessary parameters for the simulation. It should be noted that you need to have all the input files that needed to run kMC.
```shell
run_kmc input.json
```

For help, you can run:
```shell
run_kmc --help
```

### GUI usage
You can run kMCpy using the command line interface. The basic usage is as follows:
```shell
start_kmcpy_gui
```

## Citation
If you use kMCpy in your research, please cite it as follows:

```bibtex
@article{deng2022fundamental,
          title={Fundamental investigations on the sodium-ion transport properties of mixed polyanion solid-state battery electrolytes},
          author={Deng, Zeyu and Mishra, Tara P and Mahayoni, Eunike and Ma, Qianli and Tieu, Aaron Jue Kang and Guillon, Olivier and Chotard, Jean-No{\"e}l and Seznec, Vincent and Cheetham, Anthony K and Masquelier, Christian and others},
          journal={Nature communications},
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