# Prepare Structures And Occupations

The structure used by kMCpy should contain all sites that could be occupied by
the mobile species. A vacancy is an occupation state on a site, not a missing
site in the lattice.

## Load A CIF

Most workflows start from a CIF file. You can get one by:

- using a CIF that is already part of your project,
- exporting a CIF from a crystallographic database,
- copying one of the example files from a source checkout, such as
  `example/files/nasicon.cif`.

After downloading or copying the file, load it with pymatgen:

```python
from pymatgen.core import Structure

structure = Structure.from_file("nasicon.cif")
print(structure.composition)
structure.to(filename="nasicon_checked.cif")
```

If the CIF comes from a database, download it with the database tool you trust,
then inspect the labels and occupancies before using it in kMCpy. For NASICON
workflows, labels such as `Na1` and `Na2` are often useful because they allow
event generation to target specific crystallographic mobile-ion sites.

## Build A Structure From Scratch

For simple tests, a pymatgen structure can be created directly:

```python
from pymatgen.core import Lattice, Structure

lattice = Lattice.cubic(4.0)
structure = Structure(
    lattice=lattice,
    species=["Li", "Li", "O"],
    coords=[
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.25, 0.25, 0.25],
    ],
)
structure.to(filename="toy.cif")
```

This is useful for tests, but production kMC usually needs a crystallographic
structure with the correct mobile-ion network.

## Define Mutable Sites

`site_mapping` defines which crystallographic species are mutable:

```python
site_mapping = {
    "Na": ["Na", "X"],
    "Zr": "Zr",
    "Si": ["Si", "P"],
    "O": "O",
}
```

In this example:

- Na sites are active mobile-ion sites. They can be occupied by `Na` or vacancy
  state `X`.
- Si sites are active substitutional sites. They can be `Si` or `P`.
- Zr and O sites are fixed because only one species is allowed.

kMCpy builds an `ActiveSiteOrder` from the structure and `site_mapping`. This is
the compact global sequence used by occupation vectors, event indices, and
external site-energy mappings.

## Prepare Initial Occupations

The initial occupation vector is stored in active-site order. For each active
site, use the state index corresponding to the allowed species list in
`site_mapping`.

For `["Na", "X"]`, state `0` means `Na` and state `1` means `X`. For
`["Si", "P"]`, state `0` means `Si` and state `1` means `P`.

```python
initial_occupations = [
    0, 0, 1, 0,  # Na/vacancy active sites
    0, 1, 0, 0,  # Si/P active sites
]
```

If you already have a structure with vacancies or substitutions, use the
structure tools to convert it to the active-site occupation order. The important
rule is that the occupation vector, event library, and model must use the same
active-site order.

## Two Site Orders To Remember

`ActiveSiteOrder` is the global active-site sequence used by kMC.

`LocalSiteOrder` is the local environment sequence used by an LCE correlation
vector around one event. It does not choose the local-environment center; the
center is supplied by `LocalLatticeStructure`.

This distinction matters most when reusing fitted LCE coefficients or mapping
kMCpy occupations to external codes.
