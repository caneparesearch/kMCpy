# Theory

## Kinetic Monte Carlo

Kinetic Monte Carlo is a stochastic method for systems that evolve through
discrete events with known rates. In kMCpy, the events are mobile-ion hops
between crystallographic sites. Each accepted hop changes the occupation state,
advances the simulation clock, and updates the affected event rates.

This page focuses on the equations. For how to prepare structures, event
libraries, and models, see the [workflow tutorial](tutorial/index.md).

## Rejection-Free Algorithm

kMCpy uses the rejection-free algorithm, also called the n-fold way or
Bortz-Kalos-Lebowitz algorithm. Every kMC step executes one event.

For each available event $i$, the rate is

$$
r_i = \nu \exp\left(-\frac{E_i}{k_B T}\right)
$$

where $\nu$ is the attempt frequency, $E_i$ is the effective event barrier,
$k_B$ is Boltzmann's constant, and $T$ is temperature.

The probability of selecting event $j$ is

$$
P_j = \frac{r_j}{\sum_i r_i}
$$

The simulation time increment is sampled as

$$
\Delta t = \frac{-\ln(u)}{\sum_i r_i}
$$

where $u$ is a uniform random number in $(0, 1)$.

After the selected hop is applied, only events affected by the state change need
updated rates.

## Barrier And Site-Energy Contributions

kMCpy model objects ultimately provide the barrier used in the Arrhenius rate.
For local cluster expansion workflows, the effective barrier is usually written

$$
E_{\mathrm{eff}} = E_{\mathrm{KRA}} + \frac{\Delta E_{\mathrm{site}}}{2}
$$

where $E_{\mathrm{KRA}}$ is the kinetic resolved activation barrier and
$\Delta E_{\mathrm{site}} = E_{\mathrm{after}} - E_{\mathrm{before}}$ is the
signed site-energy difference for the event. This form keeps forward and
backward rates consistent with the Boltzmann factor for the site-energy
difference.

The [model tutorial](tutorial/models.md) explains how kMCpy's model classes
provide these quantities.

## Transport Properties

From a kMC trajectory, kMCpy computes several quantities that characterize ion
transport.

### Mean Squared Displacement

Mean squared displacement tracks how far mobile ions move over time:

$$
\mathrm{MSD}(t) = \left\langle |R_i(t) - R_i(0)|^2 \right\rangle
$$

In a diffusive regime, MSD grows approximately linearly with time.

### Tracer Diffusivity

Tracer diffusivity measures individual-ion diffusion:

$$
D_{\mathrm{tracer}} = \lim_{t\to\infty}
\frac{\mathrm{MSD}(t)}{2 d t}
$$

where $d$ is the simulation dimension. This quantity is also called
self-diffusivity.

### Jump Diffusivity

Jump diffusivity measures collective charge motion:

$$
D_J = \lim_{t\to\infty}
\frac{\left\langle \Delta R_{\mathrm{cm}}^2 \right\rangle}{2 d t}
$$

This is the diffusivity used in the Nernst-Einstein relation for ionic
conductivity.

### Haven Ratio

The Haven ratio compares tracer and jump diffusion:

$$
H_R = \frac{D_{\mathrm{tracer}}}{D_J}
$$

A value near 1 suggests weak correlation between tracer and collective charge
motion. Values away from 1 indicate correlated transport.

### Ionic Conductivity

kMCpy reports ionic conductivity using the jump diffusivity:

$$
\sigma = \frac{n q^2}{k_B T} D_J
$$

where $n$ is the mobile-ion concentration and $q$ is the mobile-ion charge.
kMCpy reports conductivity in mS/cm.

### Correlation Factor

The correlation factor compares the net displacement of diffusing ions to an
uncorrelated random walk with the same number of hops:

$$
f = \frac{\sum_i |\Delta R_i|^2}{a^2 \sum_i n_i}
$$

where $\Delta R_i$ is the net displacement of ion $i$, $n_i$ is the number of
hops made by ion $i$, and $a$ is the elementary hop distance.

## Convergence Requirements

Reliable transport properties require enough equilibration and enough production
sampling. The production trajectory should reach a diffusive regime where MSD is
approximately linear in time. Supercells should be large enough to reduce
finite-size effects, and independent seeds are useful for estimating statistical
uncertainty.

## Unit Conventions

kMCpy numeric APIs use fixed units. See
[Units And Interoperability](tutorial/units_interoperability.md) for the full
table and for notes about external-code adapters.

## Further Reading

- kMCpy methodology: [Deng et al., *Computational Materials Science* **229**,
  112394 (2023)](https://doi.org/10.1016/j.commatsci.2023.112394)
- NASICON application: [Deng et al., *Nature Communications* **13**, 4470
  (2022)](https://doi.org/10.1038/s41467-022-32190-7)
- First-principles methods for ion transport: [van der Ven et al., *Chemical
  Reviews* **120**, 6977-7019 (2020)](https://doi.org/10.1021/acs.chemrev.9b00601)
