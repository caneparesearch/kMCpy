.. image:: _static/kmcpy_logo.svg
   :width: 400 px
   :alt: kMCpy
   :align: center
   :class: only-light

.. image:: _static/kmcpy_logo_dark.svg
   :width: 400 px
   :alt: kMCpy
   :align: center
   :class: only-dark

kMCpy Documentation
==================================================================================

kMCpy is an open-source Python package for studying atomic migration in
crystalline materials with kinetic Monte Carlo. It provides a Python workflow
for preparing site occupations and event libraries, assigning event barriers,
running rejection-free kMC, and extracting transport properties.

Version 0.3.0 focuses on clearer scientific workflows:

- ``Configuration`` stores immutable setup and runtime controls.
- ``State`` owns mutable occupations during a simulation.
- ``LocalBarrierModel`` supports constant, rule-based, and exact local
  environment barriers.
- ``LocalClusterExpansion`` supports multicomponent Chebyshev basis functions.
- ``SiteEnergyModel`` can connect mapped external site-energy evaluators without
  rebuilding occupations at every kMC step.
- ``ActiveSiteOrder`` and ``LocalSiteOrder`` record the global and local site
  sequences used by fitted models and external adapters.

Where to start:

1. Install kMCpy using :doc:`install`.
2. Run one small simulation with :doc:`quickstart`.
3. Use :doc:`cli` for command-line input generation and simulation runs.
4. Read :doc:`mechanism` for the kMC equations and transport definitions.
5. Learn the full workflow in :doc:`tutorial/index`.
6. Use :doc:`howto/index` for advanced customization.
7. Use :doc:`reference/index` when you need exact API documentation.

The methodology has been used to study Na-ion transport in NASICON solid
electrolytes. See :doc:`about` for citation information.

.. toctree::
   :maxdepth: 1
   :caption: Start Here

   install
   quickstart
   cli

.. toctree::
   :maxdepth: 1
   :caption: Theory

   mechanism

.. toctree::
   :maxdepth: 1
   :caption: Tutorial

   tutorial/index

.. toctree::
   :maxdepth: 1
   :caption: Advanced

   howto/index

.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   about
