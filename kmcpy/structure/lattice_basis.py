"""Version-independent choice of lattice vectors for a periodic structure.

Event, initial-state, and model files refer to sites by index, and supercells
are built along the lattice vectors, so the choice of lattice vectors is part
of a simulation's identity. pymatgen may pick different but equivalent
primitive vectors between releases (pymatgen 2026 returns ``(-a, -c, -b)``
where 2025 returned ``(a, b, c)`` for NASICON). :func:`standardize_lattice_basis`
replaces that choice with a deterministic one that depends only on the lattice
itself and the Cartesian frame.
"""

from __future__ import annotations

import functools
import itertools

import numpy as np
from pymatgen.core import Lattice, Structure

# Integer change-of-basis matrices with entries in {-1, 0, 1} and det = +1.
# Every basis with the Niggli-reduced metric is reachable from a Niggli basis
# by one of these.
_UNIMODULAR = np.array(
    [
        matrix
        for matrix in (
            np.array(entries).reshape(3, 3)
            for entries in itertools.product((-1, 0, 1), repeat=9)
        )
        if round(np.linalg.det(matrix)) == 1
    ]
)


def standardize_lattice_basis(structure: Structure, tol: float = 1e-3) -> Structure:
    """Return ``structure`` expressed in kMCpy's canonical lattice basis.

    Site order, species, labels, and site properties are unchanged; each site
    keeps its Cartesian position (fractional coordinates are wrapped into the
    new cell).

    The canonical basis is chosen among all right-handed bases of the lattice
    that share the Niggli-reduced metric: the one whose Cartesian components
    ``(a_x, a_y, a_z, b_x, ..., c_z)`` are lexicographically largest, comparing
    components within ``tol`` (Angstrom) as equal.
    """
    matrix = canonical_lattice_matrix(structure.lattice.matrix, tol=tol)
    if np.allclose(matrix, structure.lattice.matrix, atol=1e-10):
        return structure

    lattice = Lattice(matrix)
    frac_coords = lattice.get_fractional_coords(structure.cart_coords)
    frac_coords -= np.floor(frac_coords + 1e-8)
    return Structure(
        lattice,
        [site.species for site in structure],
        frac_coords,
        site_properties=structure.site_properties,
        labels=[site.label for site in structure],
    )


def canonical_lattice_matrix(matrix, tol: float = 1e-3) -> np.ndarray:
    """Return the canonical basis (rows) of the lattice spanned by ``matrix``."""
    reduced = Lattice(matrix).get_niggli_reduced_lattice().matrix
    if np.linalg.det(reduced) < 0:
        reduced = -reduced
    gram = reduced @ reduced.T
    metric_tol = tol * max(1.0, float(np.max(np.abs(gram))))

    candidates = np.einsum("nij,jk->nik", _UNIMODULAR, reduced)
    grams = np.einsum("nij,nkj->nik", candidates, candidates)
    same_metric = np.all(np.abs(grams - gram) <= metric_tol, axis=(1, 2))

    def compare(left: np.ndarray, right: np.ndarray) -> int:
        for x, y in zip(left.ravel(), right.ravel()):
            if x > y + tol:
                return 1
            if x < y - tol:
                return -1
        return 0

    return max(candidates[same_metric], key=functools.cmp_to_key(compare)).copy()
