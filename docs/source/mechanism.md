# Mechanism and Theory

This section provides a detailed explanation of the mechanisms and methods implemented in kMCpy.

## Overview

kMCpy implements a **rejection-free kinetic Monte Carlo (rf-kMC)** algorithm coupled with a **Local Cluster Expansion (LCE)** model to simulate ion transport in crystalline materials. This combination allows for efficient computation of transport properties such as diffusivity and ionic conductivity.

## Kinetic Monte Carlo (kMC)

### What is kMC?

Kinetic Monte Carlo (kMC) is a stochastic simulation method used to model the time evolution of systems where discrete events occur at known rates. In the context of ion transport:

- **Events**: Individual ion hops from one site to another
- **Rates**: Probability per unit time that a hop occurs, determined by the migration barrier and temperature
- **Time evolution**: System evolves by selecting and executing events based on their rates

### Rejection-Free Algorithm

kMCpy uses a **rejection-free** (also called "first-reaction") algorithm, which is more efficient than rejection-based methods:

1. **Calculate rates**: For each possible event *i*, compute the rate *r<sub>i</sub>* using transition state theory:

   ```
   r_i = ν exp(-E_b / k_B T)
   ```

   where:
   - *ν* is the attempt frequency (typically 10^12 - 10^13 Hz)
   - *E<sub>b</sub>* is the migration barrier
   - *k<sub>B</sub>* is Boltzmann's constant
   - *T* is temperature

2. **Select event**: Choose an event *j* with probability proportional to its rate:

   ```
   P_j = r_j / Σr_i
   ```

3. **Advance time**: Update the simulation time by:

   ```
   Δt = -ln(u) / Σr_i
   ```

   where *u* is a uniform random number in (0, 1)

4. **Execute event**: Update the system state by performing the selected hop

5. **Update rates**: Recalculate rates for affected events and repeat

This approach is "rejection-free" because every step executes an event—there are no rejected moves.

### Why Use kMC?

kMC bridges the gap between ab initio molecular dynamics (too slow for long timescales) and continuum diffusion models (too coarse for atomic-scale mechanisms):

- **Time scales**: kMC can reach microseconds to seconds, far beyond MD's nanoseconds
- **Atomic resolution**: Unlike continuum models, kMC tracks individual ions and captures local environment effects
- **Temperature dependence**: Naturally captures thermally activated processes

## Local Cluster Expansion (LCE)

### Purpose

The LCE model predicts migration barriers on-the-fly during kMC simulations, eliminating the need to pre-compute barriers for every possible local configuration.

### How It Works

The LCE model expresses the migration barrier as a function of the local environment around the migrating ion:

```
E_b = E_0 + Σ α_i f_i(σ)
```

where:
- *E<sub>0</sub>* is the base barrier (empty cluster contribution)
- *α<sub>i</sub>* are fitted expansion coefficients
- *f<sub>i</sub>(σ)* are basis functions describing the local environment
- *σ* is the occupation state of neighboring sites

### Basis Functions

kMCpy supports multiple basis function types:

1. **Chebyshev polynomials**: Orthogonal polynomials that efficiently represent smooth functions
2. **Indicator functions**: Binary functions indicating presence/absence of specific configurations

The basis functions depend on:
- **Occupation**: Which sites around the hop are occupied
- **Distance**: How far neighboring ions are from the hopping ion
- **Symmetry**: Crystallographic equivalence of configurations

### Training the LCE Model

The LCE coefficients are fitted using regularized linear regression on training data from:

- **Ab initio calculations**: NEB (Nudged Elastic Band) or CI-NEB calculations
- **Empirical potentials**: Classical MD or force-field-based barrier calculations

The fitting process:

1. **Generate training data**: Compute barriers for diverse local configurations
2. **Build correlation matrix**: Evaluate basis functions for each training sample
3. **Fit coefficients**: Use ridge regression (L2 regularization) to obtain *α<sub>i</sub>*
4. **Validate**: Check RMSE and leave-one-out cross-validation (LOOCV) score

### Site Energy Model

In addition to migration barriers, kMCpy can include a **site energy** LCE model that predicts the relative stability of different sites. This affects:

- Equilibrium occupancy distribution
- Event rates (via Boltzmann factors)
- Transport anisotropy

## Transport Properties

### Quantities Computed

From a kMC trajectory, kMCpy extracts:

1. **Mean Squared Displacement (MSD)**:
   ```
   MSD(t) = ⟨|r_i(t) - r_i(0)|²⟩
   ```

2. **Tracer Diffusivity** (*D*<sub>tracer</sub>):
   ```
   D_tracer = lim[t→∞] MSD(t) / (6t)
   ```
   Measures self-diffusion of individual ions

3. **Jump Diffusivity** (*D*<sub>J</sub>):
   ```
   D_J = lim[t→∞] ⟨Δr_cm²⟩ / (6t)
   ```
   Measures collective center-of-mass motion

4. **Haven Ratio** (*H<sub>R</sub>*):
   ```
   H_R = D_J / D_tracer
   ```
   Quantifies correlation effects (*H<sub>R</sub>* = 1 means uncorrelated motion)

5. **Ionic Conductivity** (σ):
   ```
   σ = (n q² / k_B T) D_J
   ```
   where *n* is carrier concentration and *q* is charge

6. **Correlation Factor** (*f*):
   ```
   f = D_tracer / D_random
   ```
   Compares actual diffusion to random walk

### Convergence

Reliable transport properties require:
- Sufficient **equilibration passes** to reach steady state
- Many **production passes** for statistical averaging
- Large enough **supercell** to minimize finite-size effects
- Long enough **time** for the diffusive regime (MSD ∝ *t*)

## Workflow Summary

A typical kMCpy simulation follows these steps:

1. **Structure preparation**:
   - Load crystal structure (CIF file)
   - Define mobile and immutable species
   - Build supercell

2. **Model setup**:
   - Train LCE barrier model from NEB data
   - (Optional) Train LCE site energy model
   - Generate event library

3. **kMC simulation**:
   - Initialize occupation state
   - Run equilibration phase
   - Run production phase with trajectory tracking

4. **Analysis**:
   - Compute MSD, diffusivities, conductivity
   - Export displacement and hop counter data
   - Analyze temperature or composition dependence

## Advantages of kMCpy's Implementation

- **Performance**: Numba-compiled event selection and rate updates achieve near-C speed
- **Modularity**: Clean separation between model (LCE), simulator (kMC), and analysis (Tracker)
- **Flexibility**: Support for custom basis functions, multi-species systems, and anisotropic lattices
- **Reproducibility**: JSON-based configuration and random seed control ensure reproducible results

## Further Reading

For detailed theoretical background and validation, please see:

- **kMCpy methodology**: [Deng et al., *Comp. Mater. Sci.* **229**, 112394 (2023)](https://doi.org/10.1016/j.commatsci.2023.112394)
- **Application to NASICON**: [Deng et al., *Nat. Commun.* **13**, 4470 (2022)](https://doi.org/10.1038/s41467-022-32190-7)
- **Kinetic Monte Carlo review**: [Voter, *Phys. Rev. B* **57**, 13985 (1998)](https://doi.org/10.1103/PhysRevB.57.R13985)
- **Cluster expansion methods**: [Sanchez et al., *Physica A* **128**, 334 (1984)](https://doi.org/10.1016/0378-4371(84)90096-7)
