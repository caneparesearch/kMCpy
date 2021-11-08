# Kinetic Monte Carlo Simulation using Python (kmcPy)
Source code to reproduce NASICON KMC simulation
Author: Zeyu Deng
Email: dengzeyu@gmail.com

# Prerequisite Packages:
python numpy scipy pandas numba tables pymatgen

# Steps:
1. Go into kmc folder
2. python run_kmc.py comp structure_index T 
comp is the parametric composition from 0 to 1, for Na1+xZr2SixP3-xO12, x = 3-3*comp
structure_index can be any integer from 0 to 49
T is temperature in Kelvin

# Example
python run_kmc.py 0.1 1 573
