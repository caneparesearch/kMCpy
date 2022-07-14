# Kinetic Monte Carlo Simulation using Python (kmcPy)
- Author: Zeyu Deng
- Email: dengzeyu@gmail.com

# Prerequisite Packages:
python numpy scipy pandas numba tables pymatgen

# Installation Guide:
1. Create a conda environment
`conda create -n kmcpy`
`conda activate kmcpy`
2. Install pip and dependencies
`conda install python=3.8 hdf5`

3. Install required packages and kmcPy
    - if no need of gui:
        `pip install -r requirement.txt .`
    - if enable GUI
        `conda install wxpython`
        `pip install -r requirement.txt .`
4. For developer, use editable mode (developer mode) of pip
`conda install wxpython`
`pip install -r requirement.txt .`


# Run Example:
- `python run_kmc.py T `
- T is temperature in Kelvin
- This will run kMC started from the occupancy stored in initial_state.json 
- `python run_kmc.py 573`


# build docs:
```
cd doc
pip install -r doc_requirements.txt
cd ..
python dev_deploy.py
```
