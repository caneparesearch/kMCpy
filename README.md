# Kinetic Monte Carlo Simulation using Python (kmcPy)
- Author: Zeyu Deng
- Email: dengzeyu@gmail.com

# Prerequisite Packages:
python numpy scipy pandas numba tables pymatgen

# Installation Guide:
## Ubuntu x86-64
1. Create a conda environment
`conda create -n kmcpy`
`conda activate kmcpy`
2. Install pip
`conda install python=3.8`
2.1 for Apple M1 chip:
`conda instal hdf5`
3. Install required packages and kmcPy
`pip install -r requirement.txt .`
4. For developer, use editable mode (developer mode) of pip
`pip install -r requirement.txt -e .`


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
