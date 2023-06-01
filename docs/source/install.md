# Installation

## Note for Microsoft Windows users

if experiencing error information like this when installing kmpcy:

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

## With GUI enabled (Recommended for Windows, Macos, Linux personal computer)

```
conda create -n kmcpy python=3.8 hdf5 wxpython -c conda-forge
conda activate kmcpy
pip install -r requirement_gui.txt .
```


## With no GUI enabled (with access to command line environment)

```
conda create -n kmcpy python=3.8 hdf5 -c conda-forge
conda activate kmcpy
pip install -r requirement.txt .
```

## For developers and building documentation

```
conda create -n kmcpy python=3.8 hdf5 wxpython -c conda-forge
conda activate kmcpy
pip install -r requirement_gui.txt -e .
cd docs
pip install -r doc_requirements.txt
cd ..
python dev_deploy.py
```
