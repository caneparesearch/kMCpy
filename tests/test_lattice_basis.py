"""Version-independent lattice basis and active-site geometry checks."""

from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from kmcpy.io.cif import load_labeled_structure_from_cif
from kmcpy.structure.active_site_order import ActiveSiteOrder
from kmcpy.structure.lattice_basis import standardize_lattice_basis

CIF = Path(__file__).parent / "files" / "EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif"
SITE_MAPPING = {"Na": ["Na", "X"], "Zr": "Zr", "Si": ["Si", "P"], "O": "O"}
# pymatgen 2026 returns the NASICON primitive vectors as (-a, -c, -b).
PYMATGEN_2026_BASIS = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])


@pytest.fixture(scope="module")
def primitive():
    return load_labeled_structure_from_cif(str(CIF), primitive=True)


def _in_basis(structure: Structure, transform) -> Structure:
    """Same crystal, lattice vectors ``transform @ matrix``, same Cartesian sites."""
    return Structure(
        Lattice(np.asarray(transform) @ structure.lattice.matrix),
        [site.species for site in structure],
        structure.cart_coords,
        coords_are_cartesian=True,
        site_properties=structure.site_properties,
    )


def _wrapped(frac_coords):
    return np.mod(np.round(frac_coords, 6), 1.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "transform",
    [
        PYMATGEN_2026_BASIS,
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    ],
)
def test_standardized_basis_does_not_depend_on_input_basis(primitive, transform):
    standardized = standardize_lattice_basis(_in_basis(primitive, transform))

    np.testing.assert_allclose(standardized.lattice.matrix, primitive.lattice.matrix, atol=1e-8)
    np.testing.assert_allclose(_wrapped(standardized.frac_coords), _wrapped(primitive.frac_coords), atol=1e-6)
    assert [site.properties["label"] for site in standardized] == [
        site.properties["label"] for site in primitive
    ]


@pytest.mark.unit
def test_cif_primitive_cell_is_already_standard(primitive):
    assert standardize_lattice_basis(primitive) is primitive


@pytest.mark.unit
def test_metadata_from_another_basis_is_rejected(primitive):
    current = ActiveSiteOrder.from_structure_and_mapping(primitive, SITE_MAPPING, (2, 1, 1))
    other_basis = ActiveSiteOrder.from_structure_and_mapping(
        _in_basis(primitive, PYMATGEN_2026_BASIS), SITE_MAPPING, (2, 1, 1)
    )

    # Same species and index layout: the fingerprint alone cannot tell them apart.
    assert other_basis.fingerprint == current.fingerprint
    current.assert_same_order(ActiveSiteOrder.from_dict(current.as_dict()))
    with pytest.raises(ValueError, match="do not match the current structure"):
        current.assert_same_order(other_basis.as_dict())


@pytest.mark.unit
def test_metadata_without_positions_is_accepted_with_a_warning(primitive):
    current = ActiveSiteOrder.from_structure_and_mapping(primitive, SITE_MAPPING, (2, 1, 1))
    legacy = current.as_dict()
    del legacy["supercell_lattice"], legacy["active_site_frac_coords"]

    with pytest.warns(UserWarning, match="no site positions"):
        current.assert_same_order(legacy)


@pytest.mark.unit
def test_cif_labels_merge_rows_that_share_a_position(tmp_path):
    from pymatgen.io.cif import CifWriter

    from kmcpy.io.cif import load_labeled_structure_from_cif

    mixed = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.6), [{"Na": 1}, {"Cl": 0.5, "Br": 0.5}], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    path = tmp_path / "mixed.cif"
    CifWriter(mixed, symprec=0.01).write_file(path)

    structure = load_labeled_structure_from_cif(str(path))
    labels = structure.site_properties["label"]
    assert labels == ["Na0"] * 4 + ["Cl1Br2"] * 4
    assert structure.site_properties["wyckoff_sequence"] == [0, 1, 2, 3, 0, 1, 2, 3]
