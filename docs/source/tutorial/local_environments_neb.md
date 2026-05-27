# Local Environments And NEB Data

Local models need local structures. For LCE fitting, every target value must be
paired with the same ordered local occupation vector used by the model.

## Enumerate Local Environments

Use local-environment enumeration when you want to generate representative
ordered configurations around a hop:

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
    cutoff=4.0,
    variable_species=["Si", "P"],
    max_results=20,
    local_site_order="kmcpy_default",
)
```

Each endpoint pair contains an initial structure, a final structure, and the
initial/final occupation vectors. The structures can be written and used as the
starting point for NEB calculations.

## Load NEB Results

After NEB calculations finish, collect the endpoint-derived structures and the
computed target values. The target can be an `E_KRA` barrier or a site-energy
difference, depending on which LCE you are fitting.

```python
from kmcpy.io.neb import NEBDataLoader

loader = NEBDataLoader(
    model=lce,
    reference_local_lattice_structure=local_lattice,
)
loader.add_structures(
    structures=["neb_000.cif", "neb_001.cif", "neb_002.cif"],
    property_values=[310.0, 355.0, 332.0],  # meV
)

fit_files = loader.write_fitting_inputs(output_dir="fit_kra")
```

`write_fitting_inputs(...)` writes:

- `correlation_matrix.txt`,
- `e_kra.txt` by default, or another target filename if you choose one,
- `weight.txt`.

These are the files consumed by `LocalClusterExpansion.fit(...)`.

## Keep The Reference Fixed

The fitted coefficients only mean something for the local environment and local
site order used to build the correlation matrix. Do not mix:

- different local cutoffs,
- different `LocalSiteOrder` rules,
- different active-site mappings,
- correlation matrices from different LCE objects.

If you change any of these, regenerate the fitting inputs.
