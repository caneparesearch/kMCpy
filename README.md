# Kinetic Monte Carlo Simulation using Python (kmcPy)
Source code to reproduce NASICON KMC simulation
Author: Zeyu Deng
Email: dengzeyu@gmail.com

# Prerequisite Packages:
python numpy scipy pandas numba tables pymatgen

# Installation Guide:
1. Create a conda environment
conda create -n kmcpy
2. Install pip
conda install pip
3. Install required packages and kmcPy
pip install -r requirement.txt .
4. For developer, use editable mode (developer mode) of pip
pip install -r requirement.txt -e .


# Example:
1. Go into examples folder
2. python run_kmc.py comp structure_index T 
comp is the parametric composition from 0 to 1, for Na1+xZr2SixP3-xO12, x = 3-3*comp
structure_index can be any integer from 0 to 49
T is temperature in Kelvin

python run_kmc.py 0.1 1 573
