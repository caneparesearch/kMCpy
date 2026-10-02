"""CIF loading that keeps the site metadata kMCpy needs.

Structures are parsed with pymatgen's public ``CifParser`` and get two site
properties:

- ``label``: the CIF ``_atom_site_label``. When several CIF rows share one
  position (mixed occupancy), their labels are joined in table order, e.g.
  ``"Si1P1"``.
- ``wyckoff_sequence``: index of the site among the symmetry-generated copies
  of its CIF row (0, 1, 2, ... per label).
- ``local_index``: position of the site in generation order (CIF rows in table
  order, then their symmetry copies), i.e. before pymatgen sorts the sites.

Primitive cells are reduced by kMCpy rather than by the parser and expressed
in a version-independent lattice basis (see
:func:`kmcpy.structure.lattice_basis.standardize_lattice_basis`), because site
indices in event, state, and model files depend on it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser

from kmcpy.structure.lattice_basis import standardize_lattice_basis

# Fractional-coordinate tolerance for CIF rows that describe the same position.
_SAME_POSITION_TOL = 1e-4


def load_labeled_structures_from_cif(
    filename: str,
    primitive: bool = False,
    symmetrized: bool = False,
    **parser_kwargs,
) -> list[Structure]:
    """Load structures from a CIF and preserve label/Wyckoff site metadata."""
    return _labeled_structures(
        CifParser(str(filename), **parser_kwargs), primitive, symmetrized
    )


def load_labeled_structure_from_cif(
    filename: str,
    primitive: bool = False,
    **parser_kwargs,
) -> Structure:
    """Load the first labeled structure from a CIF file."""
    return load_labeled_structures_from_cif(
        filename,
        primitive=primitive,
        **parser_kwargs,
    )[0]


def load_labeled_structure_from_string(
    cif_string: str,
    primitive: bool = False,
    symmetrized: bool = False,
    **parser_kwargs,
) -> Structure:
    """Load a labeled structure from CIF text."""
    return _labeled_structures(
        CifParser.from_str(cif_string, **parser_kwargs), primitive, symmetrized
    )[0]


def _labeled_structures(
    parser: CifParser, primitive: bool, symmetrized: bool
) -> list[Structure]:
    if primitive and symmetrized:
        raise ValueError(
            "Using both 'primitive' and 'symmetrized' arguments is not currently supported "
            "since unexpected behavior might result."
        )
    merged_labels, label_rows = _merged_labels(parser.as_dict())
    magnetic = bool(parser.feature_flags.get("magcif"))

    structures = []
    for structure in parser.parse_structures(primitive=False, symmetrized=symmetrized):
        labels = [merged_labels.get(site.label, site.label) for site in structure]
        counts: dict[str, int] = defaultdict(int)
        sequence = []
        for label in labels:
            sequence.append(counts[label])
            counts[label] += 1
        generation_order = sorted(
            range(len(labels)),
            key=lambda index: (label_rows.get(labels[index], len(label_rows)), sequence[index]),
        )
        local_index = [0] * len(labels)
        for position, index in enumerate(generation_order):
            local_index[index] = position
        structure.add_site_property("wyckoff_sequence", sequence)
        structure.add_site_property("local_index", local_index)
        structure.add_site_property("label", labels)

        if primitive:
            structure = structure.get_primitive_structure(use_site_props=magnetic)
            if not magnetic:
                structure = structure.get_reduced_structure()
            structure = standardize_lattice_basis(structure)
        structures.append(structure)
    return structures


def _merged_labels(cif_data: dict[str, Any]) -> tuple[dict[str, str], dict[str, int]]:
    """Return ``(merged, rows)`` for the CIF atom-site table.

    ``merged`` maps each CIF label to the joined labels of all rows at its
    position; ``rows`` maps each joined label to the table order of that
    position.
    """
    merged: dict[str, str] = {}
    rows: dict[str, int] = {}
    for block in cif_data.values():
        labels = block.get("_atom_site_label")
        if not labels:
            continue
        if isinstance(labels, str):
            labels = [labels]
        try:
            coords = np.array(
                [
                    [_cif_float(value) for value in block[f"_atom_site_fract_{axis}"]]
                    for axis in "xyz"
                ]
            ).T.reshape(len(labels), 3)
        except (KeyError, TypeError, ValueError):
            continue

        groups: list[list[int]] = []
        for index, coord in enumerate(coords):
            for group in groups:
                delta = coords[group[0]] - coord
                if np.all(np.abs(delta - np.round(delta)) < _SAME_POSITION_TOL):
                    group.append(index)
                    break
            else:
                groups.append([index])
        for group in groups:
            joined = "".join(str(labels[index]) for index in group)
            rows.setdefault(joined, len(rows))
            for index in group:
                merged.setdefault(str(labels[index]), joined)
    return merged, rows


def _cif_float(value) -> float:
    """Parse a CIF number, dropping a standard uncertainty such as ``0.123(4)``."""
    return float(str(value).split("(")[0])


__all__ = [
    "load_labeled_structure_from_cif",
    "load_labeled_structures_from_cif",
    "load_labeled_structure_from_string",
]
