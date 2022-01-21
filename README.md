# Kinetic Monte Carlo Simulation using Python (kmcPy)
- Author: Zeyu Deng
- Email: dengzeyu@gmail.com

# Prerequisite Packages:
python numpy scipy pandas numba tables pymatgen

# Installation Guide:
1. Create a conda environment
`conda create -n kmcpy`
2. Install pip
`conda install pip`
3. Install required packages and kmcPy
`pip install -r requirement.txt .`
4. For developer, use editable mode (developer mode) of pip
`pip install -r requirement.txt -e .`

# Run Example:
- `python run_kmc.py T `
- T is temperature in Kelvin
- This will run kMC started from the occupancy stored in initial_state.json 
- `python run_kmc.py 573`
