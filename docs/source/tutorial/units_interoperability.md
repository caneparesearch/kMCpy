# Units And Interoperability

kMCpy uses fixed internal unit conventions. Keep these units explicit when
building models or connecting external codes.

## Unit Table

| Quantity | Unit |
|---|---|
| Migration barrier and fitted energy terms | meV |
| Site-energy difference returned to kMCpy | meV |
| Event rate and attempt frequency | Hz |
| Temperature | K |
| Simulation time | s |
| Length, displacement, elementary hop distance | Angstrom |
| Volume | Angstrom^3 |
| Mean squared displacement | Angstrom^2 |
| Jump and tracer diffusivity | cm^2/s |
| Conductivity | mS/cm |
| Mobile ion charge | `|e|` |
| Haven ratio and correlation factor | dimensionless |

In code:

```python
from kmcpy import Configuration

print(Configuration.field_units())
print(tracker.result_units)
```

## Model Units

[`LocalBarrierModel`](../modules/local_barrier_model.rst) barriers are in meV.

[`LocalClusterExpansion`](../modules/local_cluster_expansion.rst) fitted targets
are in meV. If the LCE is used as a `kra_model`, the fitted target is `E_KRA`.
If it is used as a `site_model`, the fitted target is the site-energy-difference
contribution expected by the composite model.

[`SiteEnergyModel`](../modules/site_energy.rst) can accept external values in
`eV` or `meV` through its `units` argument and converts them to meV for kMCpy.

## Differences From Other Codes

Other atomistic software often has a different primary purpose:

- smol, CLEASE, and CASM are commonly used for cluster expansion and Monte
  Carlo workflows on configurational lattices.
- ASE is a general atomistic structure and calculator interface.
- kMCpy focuses on event-based kinetic Monte Carlo for migration on an
  active-site network.

When connecting these tools, the important question is not just "can both codes
store an occupation?" It is whether they agree on:

- site order,
- species or state encoding,
- units,
- whether an energy is absolute or a local event difference,
- how an accepted kMC event updates the external runtime state.

For site-energy-difference adapters, kMCpy precomputes site and state mappings
once before the run. During the kMC loop, it passes only the changed endpoints
to the external callable.

See [Use site-energy-difference models](../howto/external_site_energy.md) for
mapped external-code examples.
