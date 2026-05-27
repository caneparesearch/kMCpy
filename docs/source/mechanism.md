# Theory

This section explains the theoretical foundations underlying kMCpy's implementation of kinetic Monte Carlo simulations for ion transport in crystalline materials.

## Overview

kMCpy simulates ion transport in crystalline materials using a rejection-free kinetic Monte Carlo (rf-kMC) algorithm. The hop rates can come from fitted `LocalClusterExpansion` models or direct local barrier models such as `LocalBarrierModel`. These models provide migration barriers for the same kMC engine, enabling efficient computation of transport properties including diffusivity and ionic conductivity.

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

## Barrier Models

kMCpy currently provides two model families for assigning hop barriers:

- `LocalClusterExpansion`: a fitted model that predicts barriers from local occupation features. Use this when you want interpolation over many local configurations from a fitted training set.
- `LocalBarrierModel`: an ordered rule model for constant barriers, count rules, species-count rules, wildcard patterns, and exact catalog-style matches. Use this when barrier logic can be written directly.

Both models operate in the active-site index space used by the event library and simulation state. The simulation config points to a serialized `model_file`; for kMCpy model-file envelopes, the model type is stored in that file.

## Unit Conventions

kMCpy numeric APIs use fixed units. The conventions are exposed in code through
`kmcpy.units`, `Configuration.field_units()`, and `Tracker.result_units`.

| Quantity | Unit |
|---|---|
| migration barrier and fitted energy terms | meV |
| event probability/rate and attempt frequency | Hz |
| temperature | K |
| simulation time | s |
| length, displacement, elementary hop distance | Angstrom |
| volume | Angstrom^3 |
| mean squared displacement | Angstrom^2 |
| jump/tracer diffusivity | cm^2/s |
| conductivity | mS/cm |
| mobile ion charge | `|e|` |
| Haven ratio and correlation factor | dimensionless |

`Tracker.write_results(...)` writes the usual result CSV and a
`results_units*.json.gz` sidecar so result columns remain machine-readable
without changing their historical names.

## Local Cluster Expansion (LCE)

### Purpose

Computing migration barriers for every possible local configuration in a crystal is computationally prohibitive. The LCE model solves this problem by predicting migration barriers on-the-fly during simulations based on the local environment around each hop. This eliminates the need to pre-compute and store barriers for all configurations.

### Model Formulation

The LCE model expresses the migration barrier as a sum of basis functions that describe the local environment:

$$E_b = E_0 + \sum_i \alpha_i f_i(\sigma)$$

where $E_0$ is the base barrier for an empty cluster, $\alpha_i$ are fitted expansion coefficients, $f_i(\sigma)$ are basis functions evaluated on the local environment, and $\sigma$ represents the occupation state of neighboring sites. The basis functions sum over all symmetry-equivalent configurations within each cluster orbit.

### Basis Functions

kMCpy supports multiple types of basis functions to represent the local environment:

**Chebyshev site functions** encode the discrete species state on each active
site. If a site allows `q` species, kMCpy stores the species as state indices
`0..q-1` and evaluates `q - 1` non-constant Chebyshev functions for that site.
Cluster features are then decorated products of the selected site functions,
so a pair of two four-species sites contributes up to `3 x 3` decorated pair
functions.

**Indicator functions** are binary functions that signal the presence or absence of specific atomic configurations. These are useful for capturing discrete structural features that affect migration barriers.

The basis functions encode information about which sites around the hop are occupied, how far neighboring ions are from the hopping ion, and crystallographic symmetry equivalences.

### Local Site Ordering

The LCE correlation vector is order-sensitive: each fitted coefficient corresponds to a specific component of the local occupation vector. kMCpy records this feature order through a local site ordering convention. The default convention preserves current kMCpy behavior, while `nasicon_nat_commun_2022` reproduces the historical NASICON single-unit convention: use the selected Na as the geometric center, remove that center site from the occupation vector, then sort the remaining local sites by species and Cartesian `x` coordinate.

Old fitted coefficients should only be reused with the same `cluster_site_indices` and ordering convention, or with an explicitly verified remapping. See the [local ordering how-to](howto/local_ordering.md) for usage.

### Training the LCE Model

The expansion coefficients $\alpha_i$ are determined by fitting to training data, typically obtained from ab initio calculations such as Nudged Elastic Band (NEB) or Climbing Image NEB (CI-NEB) methods, or from empirical potentials using classical molecular dynamics.

The fitting procedure involves:

1. Computing barriers for a diverse set of local configurations to create training data.
2. Evaluating the basis functions for each training sample to build a correlation matrix.
3. Using ridge regression (L2 regularization) to fit the coefficients $\alpha_i$ while avoiding overfitting.
4. Validating the model by checking the root mean squared error (RMSE) and leave-one-out cross-validation (LOOCV) score.

### Composite LCE Model

kMCpy uses a composite model that combines two LCE components: one for migration barriers (E_KRA) and one for site energy differences. This separation is important because the rate of an ion hop depends both on the barrier height and on the relative stability of the initial and final sites.

The **site energy model** predicts the energy of an ion at a particular site based on its local environment. When an ion hops from site A to site B, the energy difference affects the effective barrier. If site B is lower in energy, the forward hop is easier than the reverse hop.

The **barrier model** (E_KRA) predicts the barrier height at the transition state, representing the energy cost of moving an ion through the activated complex between sites.

kMCpy combines these contributions to compute the effective barrier for each hop:

$$E_{\text{eff}} = E_{\text{KRA}} + \frac{\text{direction} \times \Delta E_{\text{site}}}{2}$$

where direction indicates whether the hop is forward (+1), backward (-1), or
currently unavailable (0). kMCpy derives this from precomputed mobile/vacancy
state codes for each event endpoint, so multistate active sites do not rely on
numeric subtraction of occupation labels. $\Delta E_{\text{site}}$ is the site
energy difference between the final and initial sites. This formulation ensures
that detailed balance is maintained: the ratio of forward to backward hop rates
satisfies the Boltzmann factor for the site energy difference.

## Local Barrier Model

`LocalBarrierModel` is the direct-rule alternative to LCE fitting. It checks ordered local rules and returns the first matching barrier. A rule can represent a constant fallback, a count of occupied or vacant sites, a count of chemical species such as "at least 4 Si in the local environment", a wildcard occupation pattern, or an exact event/local-occupation match.

Exact rules are keyed by the hopping sites, local environment sites, and occupations of the canonical local site list. If no rule matches and no default barrier is provided, lookup fails so missing data can be corrected explicitly.

See the [local barrier model how-to](howto/local_barrier_model.md) for rule examples.

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
$$H_R = \frac{D_{\text{tracer}}}{D_J}$$

A value of $H_R = 1$ indicates uncorrelated motion. Values away from 1 indicate correlated charge and tracer transport.

**Ionic Conductivity** ($\sigma$) relates diffusion to charge transport:
$$\sigma = \frac{n q^2}{k_B T} D_J$$

where $n$ is the mobile ion concentration and $q$ is the ionic charge. kMCpy
uses $D_J$ in cm<sup>2</sup>/s, carrier concentration in
1/Angstrom<sup>3</sup>, charge in `|e|`, $k_B T$ in meV, and reports
conductivity in mS/cm.

**Correlation Factor** ($f$) compares the net displacement of diffusing ions to an uncorrelated random walk with the same total number of hops:
$$f = \frac{\sum_i |\Delta R_i|^2}{a^2 \sum_i n_i}$$

where $\Delta R_i$ connects the endpoints of ion $i$'s trajectory, $n_i$ is the number of hops made by that ion, and $a$ is the elementary hop distance. This aggregate form is equivalent to a hop-count-weighted average of the single-particle correlation factors, so ions with zero hops naturally do not contribute. The correlation factor measures correlations between successive hops of the same ion and is distinct from the Haven ratio.

### Convergence Requirements

Obtaining reliable transport properties requires careful attention to simulation parameters. The system must first reach steady state through equilibration passes before production passes can be used for averaging. The supercell must be large enough to minimize finite-size effects, and the simulation must run long enough to reach the diffusive regime where MSD grows linearly with time ($\text{MSD} \propto t$). Many production passes are needed to reduce statistical uncertainty in the computed transport coefficients.

## Further Reading

For detailed theoretical background and validation, please see:

- **kMCpy methodology**: [Deng et al., *Comp. Mater. Sci.* **229**, 112394 (2023)](https://doi.org/10.1016/j.commatsci.2023.112394)
- **Application to NASICON**: [Deng et al., *Nat. Commun.* **13**, 4470 (2022)](https://doi.org/10.1038/s41467-022-32190-7)
- **First-principles methods for ion transport**: [van der Ven et al., *Chem. Rev.* **120**, 6977-7019 (2020)](https://doi.org/10.1021/acs.chemrev.9b00601)
