# Kinetic Monte Carlo Simulation using Python (kmcPy)
- Author: Zeyu Deng
- Email: dengzeyu@gmail.com

# Prerequisite Packages:
python numpy scipy pandas numba tables pymatgen

# Installation Guide:
1. With GUI enabled (recommended in MacOS, ubuntu, windows, etc...)

```
conda create -n kmcpy python=3.8 hdf5 wxpython -c conda-forge
conda activate kmcpy
pip install -r requirement_gui.txt .
```

1.1 with no GUI enabled (for command line, running on server)

```
conda create -n kmcpy python=3.8 hdf5 -c conda-forge
conda activate kmcpy
pip install -r requirement.txt .
```

1.2 for developers and building docs

```
conda create -n kmcpy python=3.8 hdf5 wxpython -c conda-forge
conda activate kmcpy
pip install -r requirement_gui.txt -e .
cd doc
pip install -r doc_requirements.txt
cd ..
python dev_deploy.py
```

# Run:

The wrapper is in the executable/ folder

run this to add the them in the path

```
export PATH=`pwd`/executable:$PATH
```

if GUI not enabled:

`wrapper.py PATH_TO_INPUT.json`

if enabled:

`gui_wrapper.py` 

