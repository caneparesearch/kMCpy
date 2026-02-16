# Theory

This section explains the theoretical foundations underlying kMCpy's implementation of kinetic Monte Carlo simulations for ion transport in crystalline materials.

## Overview

kMCpy simulates ion transport in crystalline materials using a rejection-free kinetic Monte Carlo (rf-kMC) algorithm combined with a Local Cluster Expansion (LCE) model. This approach enables efficient computation of transport properties including diffusivity and ionic conductivity.

## Kinetic Monte Carlo (kMC)

### What is kMC?

Kinetic Monte Carlo is a stochastic simulation method for modeling the time evolution of systems where discrete events occur at known rates. For ion transport simulations, these events are individual ion hops between crystallographic sites. The rate at which each hop occurs depends on the migration barrier and temperature. By selecting and executing events probabilistically, the system evolves through a sequence of states that captures the thermally activated diffusion process.

### Rejection-Free Algorithm

kMCpy uses the rejection-free algorithm (also called the "n-fold way" or "Bortz-Kalos-Lebowitz (BKL) algorithm"), which is more efficient than rejection-based Monte Carlo methods. In this approach, every simulation step executes an event—there are no rejected moves. The algorithm proceeds as follows:

1. **Calculate rates**: For each possible event *i*, compute the rate *r<sub>i</sub>* using transition state theory:

   $$r_i = \nu \exp\left(-\frac{E_b}{k_B T}\right)$$

   where $\nu$ is the attempt frequency (typically 10<sup>12</sup> - 10<sup>13</sup> Hz), $E_b$ is the migration barrier, $k_B$ is Boltzmann's constant, and $T$ is temperature.

2. **Select event**: Choose an event *j* with probability proportional to its rate:

   $$P_j = \frac{r_j}{\sum_i r_i}$$

3. **Advance time**: Update the simulation time by drawing a random time interval:

   $$\Delta t = \frac{-\ln(u)}{\sum_i r_i}$$

   where $u$ is a uniform random number in (0, 1).

4. **Execute event**: Update the system state by performing the selected hop.

5. **Update rates**: Recalculate rates for events affected by the state change and repeat.

### Why Use kMC?

Kinetic Monte Carlo bridges the gap between ab initio molecular dynamics (which is computationally too expensive for long timescales) and continuum diffusion models (which lack atomic-level detail). kMC simulations can reach microseconds to seconds—far beyond the nanosecond timescales accessible to molecular dynamics. At the same time, kMC maintains atomic resolution, tracking individual ions and capturing how local environment affects transport. This makes kMC particularly valuable for studying thermally activated processes where rare events dominate the long-time behavior.

## Local Cluster Expansion (LCE)

### Purpose

Computing migration barriers for every possible local configuration in a crystal is computationally prohibitive. The LCE model solves this problem by predicting migration barriers on-the-fly during simulations based on the local environment around each hop. This eliminates the need to pre-compute and store barriers for all configurations.

### Model Formulation

The LCE model expresses the migration barrier as a sum of basis functions that describe the local environment:

$$E_b = E_0 + \sum_i \alpha_i f_i(\sigma)$$

where $E_0$ is the base barrier for an empty cluster, $\alpha_i$ are fitted expansion coefficients, $f_i(\sigma)$ are basis functions evaluated on the local environment, and $\sigma$ represents the occupation state of neighboring sites. The basis functions sum over all symmetry-equivalent configurations within each cluster orbit.

### Basis Functions

kMCpy supports multiple types of basis functions to represent the local environment:

**Chebyshev polynomials** are orthogonal polynomials that efficiently represent smooth variations in barrier height as a function of the local environment. These are particularly useful when the barrier depends continuously on factors like the distance to neighboring ions.

**Indicator functions** are binary functions that signal the presence or absence of specific atomic configurations. These are useful for capturing discrete structural features that affect migration barriers.

The basis functions encode information about which sites around the hop are occupied, how far neighboring ions are from the hopping ion, and crystallographic symmetry equivalences.

### Training the LCE Model

The expansion coefficients $\alpha_i$ are determined by fitting to training data, typically obtained from ab initio calculations such as Nudged Elastic Band (NEB) or Climbing Image NEB (CI-NEB) methods, or from empirical potentials using classical molecular dynamics.

The fitting procedure involves:

1. Computing barriers for a diverse set of local configurations to create training data.
2. Evaluating the basis functions for each training sample to build a correlation matrix.
3. Using ridge regression (L2 regularization) to fit the coefficients $\alpha_i$ while avoiding overfitting.
4. Validating the model by checking the root mean squared error (RMSE) and leave-one-out cross-validation (LOOCV) score.

### Model

kMCpy uses a composite model that combines two LCE components: one for migration barriers (E_KRA) and one for site energy differences. This separation is important because the rate of an ion hop depends both on the barrier height and on the relative stability of the initial and final sites.

The **site energy model** predicts the energy of an ion at a particular site based on its local environment. When an ion hops from site A to site B, the energy difference affects the effective barrier. If site B is lower in energy, the forward hop is easier than the reverse hop.

The **barrier model** (E_KRA) predicts the barrier height at the transition state, representing the energy cost of moving an ion through the activated complex between sites.

kMCpy combines these contributions to compute the effective barrier for each hop:

$$E_{\text{eff}} = E_{\text{KRA}} + \frac{\text{direction} \times \Delta E_{\text{site}}}{2}$$

where direction indicates whether the hop is forward (+1) or backward (-1), and $\Delta E_{\text{site}}$ is the site energy difference between the final and initial sites. This formulation ensures that detailed balance is maintained: the ratio of forward to backward hop rates satisfies the Boltzmann factor for the site energy difference.

## Transport Properties

From a kMC trajectory, kMCpy computes several quantities that characterize ion transport:

**Mean Squared Displacement (MSD)** tracks how far ions move over time:
$$\text{MSD}(t) = \langle |r_i(t) - r_i(0)|^2 \rangle$$

This quantity increases linearly with time in the diffusive regime.

**Tracer Diffusivity** ($D_{\text{tracer}}$) measures how individual ions diffuse:
$$D_{\text{tracer}} = \lim_{t\to\infty} \frac{\text{MSD}(t)}{6t}$$

This quantity, also called self-diffusivity, tracks each ion's displacement independently. It describes how fast a tagged particle diffuses through the lattice and represents the diffusion coefficient you would measure in a tracer experiment.

**Jump Diffusivity** ($D_J$) measures the collective motion of all mobile ions:
$$D_J = \lim_{t\to\infty} \frac{\langle \Delta r_{\text{cm}}^2 \rangle}{6t}$$

Unlike tracer diffusivity, jump diffusivity accounts for correlations between ion movements. This is the diffusivity that enters the Nernst-Einstein relation connecting diffusion to ionic conductivity. When ions move in a correlated fashion (for example, if one ion's motion tends to block another), jump diffusivity can be significantly different from tracer diffusivity.

**Haven Ratio** ($H_R$) quantifies correlations:
$$H_R = \frac{D_J}{D_{\text{tracer}}}$$

A value of $H_R = 1$ indicates uncorrelated motion—each ion diffuses independently. Values $H_R < 1$ indicate correlated motion, common in materials with strong ion-ion interactions where one ion's movement affects others. Values $H_R > 1$ can occur in vacancy-mediated diffusion or when ions move in an anti-correlated fashion.

**Ionic Conductivity** ($\sigma$) relates diffusion to charge transport:
$$\sigma = \frac{n q^2}{k_B T} D_J$$

where $n$ is the mobile ion concentration and $q$ is the ionic charge.

**Correlation Factor** ($f$) compares actual diffusion to a random walk:
$$f = \frac{D_{\text{tracer}}}{D_{\text{random}}}$$

This factor accounts for how site-blocking and correlated motion reduce diffusivity below what would occur for a random walk on the same lattice.

### Convergence Requirements

Obtaining reliable transport properties requires careful attention to simulation parameters. The system must first reach steady state through equilibration passes before production passes can be used for averaging. The supercell must be large enough to minimize finite-size effects, and the simulation must run long enough to reach the diffusive regime where MSD grows linearly with time ($\text{MSD} \propto t$). Many production passes are needed to reduce statistical uncertainty in the computed transport coefficients.

## Further Reading

For detailed theoretical background and validation, please see:

- **kMCpy methodology**: [Deng et al., *Comp. Mater. Sci.* **229**, 112394 (2023)](https://doi.org/10.1016/j.commatsci.2023.112394)
- **Application to NASICON**: [Deng et al., *Nat. Commun.* **13**, 4470 (2022)](https://doi.org/10.1038/s41467-022-32190-7)
- **First-principles methods for ion transport**: [van der Ven et al., *Chem. Rev.* **120**, 6977-7019 (2020)](https://doi.org/10.1021/acs.chemrev.9b00601)
