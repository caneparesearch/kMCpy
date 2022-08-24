# Kinetic Monte Carlo Simulation using Python (kmcPy)
- Author: Zeyu Deng
- Email: dengzeyu@gmail.com

# Prerequisite Packages:
python numpy scipy pandas numba tables pymatgen

# Installation Guide:
1. Create a conda environment

```
conda create -n kmcpy python=3.8 hdf5 wxpython -c conda-forge
conda activate kmcpy
```

2. Install required packages and kmcPy

`pip install -r requirement.txt .`

3. For developer, use editable mode (developer mode) of pip

`pip install -r requirement.txt -e .`


for building docs:
```
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

