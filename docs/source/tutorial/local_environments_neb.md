# Local Environments And NEB Data

This page is for the LCE branch of the workflow. If you use a
[`LocalBarrierModel`](../modules/local_barrier_model.rst) with hand-written
rules, you can skip this page and continue to [Prepare Input And Run kMC](run_kmc.md).

For LCE fitting, every target value must be paired with the same ordered local
occupation vector used by the [`LocalClusterExpansion`](../modules/local_cluster_expansion.rst)
built in [Choose And Build A Model](models.md).

## Enumerate Local Environments

Use [`enumerate_neb_endpoint_pairs`](../modules/local_environment_enumerator.rst)
when you want to generate representative ordered configurations around one hop
and turn each configuration into an initial/final NEB endpoint pair:

```python
from pymatgen.core import Structure
from kmcpy.structure import LatticeStructure
from kmcpy.structure.local_environment_enumerator import enumerate_neb_endpoint_pairs

structure = Structure.from_file("nasicon.cif")
site_mapping = {"Na": ["Na", "X"], "Zr": "Zr", "Si": ["Si", "P"], "O": "O"}

lattice = LatticeStructure(
    template_structure=structure,
    site_mapping=site_mapping,
    basis_type="chebyshev",
)

endpoint_pairs = enumerate_neb_endpoint_pairs(
    lattice_structure=lattice,
    mobile_ion_indices=(0, 1),
    center=0,
    cutoff=4.0,
    variable_species=["Si", "P"],
    max_results=20,
    local_site_order="kmcpy_default",
)
```

This uses [`LatticeStructure`](../modules/lattice_structure.rst), the global
active-site lattice. The LCE itself uses `LocalLatticeStructure`, the local
reference extracted from the same structure and `site_mapping`. Keep both built
from the same structure convention.

The important arguments are:

- `lattice_structure`: global structure model with the same `site_mapping` used
  by the simulation.
- `mobile_ion_indices`: compact active-site indices for the hop endpoints.
  These should correspond to the event you plan to fit.
- `center`: local-environment origin. If omitted, kMCpy uses the initial
  mobile-ion site.
- `cutoff`: local-environment radius in angstrom.
- `variable_species`: species that may be enumerated on selected active sites.
- `variable_site_indices`: optional compact active-site indices to vary. If
  omitted, kMCpy varies active sites in the local environment that can host
  `variable_species`.
- `species_counts`: optional exact composition constraint, such as
  `{"Si": 2, "P": 1}`.
- `local_site_order`: must match the order used when building the LCE.
- `max_results`: safety limit on the number of generated environments.

Each endpoint pair contains an initial structure, a final structure, and the
initial/final occupation vectors. The structures can be written and used as the
starting point for NEB calculations. kMCpy does not create VASP input files or
interpolated images; it prepares the ordered endpoints and metadata.

For more enumeration options, see [Enumerate Local Environments](../howto/local_environment_enumeration.md).

## Load NEB Results

After NEB calculations finish, collect the endpoint-derived structures and the
computed target values. The target can be an `E_KRA` barrier or a site-energy
difference, depending on which LCE you are fitting. The example below uses
`kra_lce` and `local_lattice` from [Choose And Build A Model](models.md).

```python
from kmcpy.io.neb import NEBDataLoader

loader = NEBDataLoader(
    model=kra_lce,
    reference_local_lattice_structure=local_lattice,
)
loader.add_structures(
    structures=["neb_000.cif", "neb_001.cif", "neb_002.cif"],
    property_values=[310.0, 355.0, 332.0],  # meV
)

fit_files = loader.write_fitting_inputs(output_dir="fit_kra")
```

The important [`NEBDataLoader`](../modules/neb.rst) arguments are:

- `model`: the LCE whose correlation vector will be evaluated.
- `reference_local_lattice_structure`: the same local lattice used to build the
  LCE. This keeps the occupation vector order fixed.
- `structures`: endpoint-derived structures or paths readable by pymatgen.
- `property_values`: target values in meV, one per structure.
- `output_dir`: directory for the fitting input files.

`write_fitting_inputs(...)` writes:

- `correlation_matrix.txt`,
- `e_kra.txt` by default, or another target filename if you choose one,
- `weight.txt`.

These are the files consumed by `LocalClusterExpansion.fit(...)`.
The returned dictionary can be passed directly into the fitting call on the next
page:

```python
fit_files = loader.write_fitting_inputs(output_dir="fit_kra")
# later:
# params, y_pred, y_true = kra_lce.fit(**fit_files, alpha=1e-4)
```

## Keep The Reference Fixed

The fitted coefficients only mean something for the local environment and local
site order used to build the correlation matrix. Do not mix:

- different local cutoffs,
- different `LocalSiteOrder` rules,
- different active-site mappings,
- correlation matrices from different LCE objects.

If you change any of these, regenerate the fitting inputs.

Next: [Fit A Local Cluster Expansion](lce_fitting.md).
