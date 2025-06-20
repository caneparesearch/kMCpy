.. image:: _static/kmcpy_logo.png
   :width: 300 px
   :alt: kMCpy
   :align: center

kMCpy: A python package to simulate transport properties using Kinetic Monte Carlo
==================================================================================


kMCpy is an open-source Python package for studying atomic migration using the kinetic Monte Carlo technique. It offers a comprehensive Python-based approach to compute kinetic properties, suitable for research, development, and prediction of new functional materials.

Key features include a local cluster expansion model toolkit, a rejection-free kinetic Monte Carlo (rf-kMC) solver, and tools to extract ion transport properties like diffusivities and conductivities. The local cluster expansion model toolkit facilitates model fitting from ab initio or empirical barrier calculations. Post-training, the local cluster expansion model can compute migration barriers in crystalline materials within the transition state theory.

Advantages of using kMCpy:

1.  Written entirely in Python with a modular design, promoting developer-centricity and easy feature addition.
2.  Cross-platform compatibility, supporting Windows, macOS, and Linux.
3.  Performance-optimized kMC routines using `Numba <https://numba.pydata.org/>`_, resulting in significant speed improvements.

This code was recently employed to investigate `the transport properties of Na-ion in NaSiCON solid electrolyte <https://www.nature.com/articles/s41467-022-32190-7>`_. In this study, rf-kMC was used to model Na-ion conductivity in NaSiCON, leading to the discovery that maximum conductivity is achieved at Na=3.4.

.. toctree::
   :maxdepth: 1
   :hidden:

   install
   modules/api
   about

