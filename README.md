# Kinetic Monte Carlo Simulation using Python (kMCpy)
![image](docs/source/_static/kmcpy_logo.png)
- Author: Zeyu Deng
- Email: dengzeyu@gmail.com

kMCpy is an open source python based package intended to study the migration of atoms using the kinetic Monte Carlo technique. kMCpy provides an all python, systematic way to compute kinetic properties, which can be readily used for investigation, development, and prediction of new functional materials. 

This package includes a local cluster expansion model toolkit, a rejection-free kinetic Monte Carlo (rf-kMC) solver, and several python classes to extract ion transport properties such as diffusivities and conductivities. 

The local cluster expansion model toolkit can be used to fit a model from barrier calculated from first-principles or any other empirical methods. Following the training process the local cluster expansion model can compute migration barriers in crystalline materials within the transition state theory.

Some of the advantages of using this package are:

1. kMCpy is fully based on python and is modular which makes it developer centric thus facilitating quick addition of new features.

2. It is cross-platform and supports most operating systems such as Windows, macOS, and Linux.

3. Intensive kMC routines has been optimized into machine code in the fly using Numba (https://numba.pydata.org), which results in manifold increase in performance. 


This code has been recently used to explore the transport properties of Na-ion in NaSICON solid electrolyte (https://www.nature.com/articles/s41467-022-32190-7).
Some of the relevant aspects of the code from the mentioned paper are shown below. 

The rf-kMC as a part of this code was used to model the Na-ion conductivity in the $\mathrm{Na_{1+x}Zr_{2}Si_{x}P_{3-x}O_{12}}$ which led to the discovery of maximum conductivity of the solid electrolyte is achieved for Na=3.4.

![image](docs/source/_static/computed_conductivity.png)

   Calculated Na+ diffusivity (a), conductivity (b), Haven's ratio (c) and averaged correlation factor (d) of $\mathrm{Na_{1+x}Zr_{2}Si_{x}P_{3-x}O_{12}}$ at several temperatures: 373 (dark blue circles), 473 (orange squares) and 573 (red triangles) K, respectively. In panel (b), the computed ionic conductivities are compared with the experimental values of this work (Supplementary Fig. 6) at selected temperatures. Experimental values in (b) from this work are depicted with light blue (373 K), yellow (473 K), and red (573 K) crosses belonging to the same $\mathrm{Na_{1+x}Zr_{2}Si_{x}P_{3-x}O_{12}}$ compositions but of pellets with different compacities (>70 and >90%, see legend).



# Prerequisite Packages:
- kMCpy: python, pymatgen, numba, scikit-learn, joblib, glob2, pytest
## Note for Windows user

If experiencing error information like this when installing kMCpy:

```
error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
      [end of output]

 note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for pymatgen
  Building wheel for kMCpy (setup.py) ... done
  Created wheel for kMCpy: filename=kMCpy-0.1.dev0-py3-none-any.whl size=124937 sha256=1282afef8589ee100a8d4fa1b53748d2d69a2e041d8c8662cfa8374a23222d60
  Stored in directory: c:\users\wdagutilityaccount\appdata\local\pip\cache\wheels\a6\a2\a6\4675cd18beeaea66ca25508dcaef9c1b59689e7794a770d602
Successfully built kMCpy
Failed to build pymatgen
ERROR: Could not build wheels for pymatgen, which is required to install pyproject.toml-based projects
```

Please visit the prompted website, follow the instruction to download Microsoft C++ build tools, install the "desktop development with C++" component and retry installing kMCpy. 

# Installation Guide:

## Command line environment
It is highly recommended to install kMCpy using [UV](https://docs.astral.sh/uv/getting-started/installation/) and use it with virtual environment.

```shell
uv venv #optional if you have already created a venv
source .venv/bin/activate
uv sync
uv pip install .
```

## For developers 
```shell
uv venv #optional if you have already created a venv
source .venv/bin/activate
uv sync
uv pip install -e .
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
uv sync --all-groups
python build_doc.py
```

# Run kMCpy
It is recommended to run kMCpy using the API. See examples for more details. A wrapper is also provided if you want to run kMCpy through command line only. 

- If GUI enabled: try `pythonw gui_wrapper.py` or `python gui_wrapper`
- If GUI not enabled: `wrapper.py PATH_TO_INPUT.json`
