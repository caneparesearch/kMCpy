![image](docs/source/_static/kmcpy_logo.png)

# kMCpy: A python package to simulate transport properties using Kinetic Monte Carlo



- Author: Zeyu Deng
- Email: dengzeyu@gmail.com


kMCpy is an open source python based package intended to study the migration of atoms using the kinetic Monte Carlo technique. kMCpy provides an all python, systematic way to compute kinetic properties, which can be readily used for investigation, development, and prediction of new functional materials. 

This package includes a local cluster expansion model toolkit, a rejection-free kinetic Monte Carlo (rf-kMC) solver, and several python classes to extract ion transport properties such as diffusivities and conductivities. 

The local cluster expansion model toolkit can be used to fit a model from barrier calculated from first-principles or any other empirical methods. Following the training process the local cluster expansion model can compute migration barriers in crystalline materials within the transition state theory.

Some of the advantages of using this package are:

1. kMCpy is fully based on python and is modular which makes it developer centric thus facilitating quick addition of new features.

2. It is cross-platform and supports most operating systems such as Windows, macOS, and Linux.

3. Intensive kMC routines has been optimized into machine code in the fly using Numba (https://numba.pydata.org/), which results in manifold increase in performance. 

The documentation of kMCpy is accessible at this link (https://kmcpy.readthedocs.io). 


# Installation Guide:

## Note for Microsoft Windows users

if experiencing error information like this when installing kmpcy:

```
error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
      [end of output]

 note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for pymatgen
  Building wheel for kmcPy (setup.py) ... done
  Created wheel for kmcPy: filename=kmcPy-0.1.dev0-py3-none-any.whl size=124937 sha256=1282afef8589ee100a8d4fa1b53748d2d69a2e041d8c8662cfa8374a23222d60
  Stored in directory: c:\users\wdagutilityaccount\appdata\local\pip\cache\wheels\a6\a2\a6\4675cd18beeaea66ca25508dcaef9c1b59689e7794a770d602
Successfully built kmcPy
Failed to build pymatgen
ERROR: Could not build wheels for pymatgen, which is required to install pyproject.toml-based projects
```

Please visit the prompted website, follow the instruction to download Microsoft C++ build tools, install the "desktop development with C++" component and retry installing kMCpy. 

## With GUI enabled (Recommended for Windows, Macos, Linux personal computer)

```
conda create -n kmcpy python=3.8 hdf5 wxpython -c conda-forge
conda activate kmcpy
pip install -r requirement_gui.txt .
```


## With no GUI enabled (to access the command line environment)

```
conda create -n kmcpy python=3.8 hdf5 -c conda-forge
conda activate kmcpy
pip install -r requirement.txt .
```

## For developers and building the documentation

```
conda create -n kmcpy python=3.8 hdf5 wxpython -c conda-forge
conda activate kmcpy
pip install -r requirement_gui.txt -e .
cd docs
pip install -r doc_requirements.txt
cd ..
python dev_deploy.py
```

# Running kMCpy:

The wrapper is in the executable/ folder

if GUI enabled:

try:

`pythonw gui_wrapper.py` 

or

`python gui_wrapper`


if GUI not enabled:

`wrapper.py PATH_TO_INPUT.json`


