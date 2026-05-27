import pytest
from pymatgen.core import Lattice, Structure

from kmcpy.structure import ActiveSiteOrder, LatticeStructure


def _structure():
    return Structure(
        Lattice.cubic(10.0),
        ["Na", "O", "Si", "Na"],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        coords_are_cartesian=True,
    )


def test_active_site_order_infers_mutable_and_fixed_sites():
    mapping = {"Na": ["Na", "X"], "O": "O", "Si": ["Si", "P"]}
    active_site_order = ActiveSiteOrder.from_structure_and_mapping(_structure(), mapping)

    assert active_site_order.primitive_active_indices == (0, 2, 3)
    assert active_site_order.active_to_original == (0, 2, 3)
    assert active_site_order.original_to_active == {0: 0, 2: 1, 3: 2}
    assert active_site_order.active_site_count == 3
    assert [site.species_string for site in active_site_order.active_structure()] == [
        "Na",
        "Si",
        "Na",
    ]


def test_active_site_order_supercell_properties_and_fingerprint_are_stable():
    mapping = {"Na": ["Na", "X"], "O": "O", "Si": ["Si", "P"]}
    first = ActiveSiteOrder.from_structure_and_mapping(
        _structure(), mapping, supercell_shape=(2, 1, 1)
    )
    second = ActiveSiteOrder.from_structure_and_mapping(
        _structure(), mapping, supercell_shape=(2, 1, 1)
    )

    assert first.fingerprint == second.fingerprint
    assert first.active_site_count == 6
    active_structure = first.active_structure()
    assert active_structure.site_properties["_kmcpy_active_site_index"] == list(range(6))
    assert set(active_structure.site_properties["_kmcpy_primitive_site_index"]) == {0, 2, 3}


def test_active_site_order_selects_active_values_and_rejects_wrong_lengths():
    active_site_order = ActiveSiteOrder.from_structure_and_mapping(
        _structure(), {"Na": ["Na", "X"], "O": "O", "Si": ["Si", "P"]}
    )

    assert active_site_order.select_active_values([0, 1, 0]) == [0, 1, 0]
    assert active_site_order.select_active_values([0, 9, 1, 0]) == [0, 1, 0]
    with pytest.raises(ValueError, match="Occupation length"):
        active_site_order.select_active_values([1, 2])


def test_active_site_order_accepts_neutral_mapping_for_oxidized_structure():
    structure = _structure()
    structure.add_oxidation_state_by_element({"Na": 1, "O": -2, "Si": 4})

    active_site_order = ActiveSiteOrder.from_structure_and_mapping(
        structure, {"Na": ["Na", "X"], "O": "O", "Si": ["Si", "P"]}
    )

    assert active_site_order.primitive_active_indices == (0, 2, 3)


def test_lattice_structure_exposes_active_site_order():
    model = LatticeStructure(
        _structure(), {"Na": ["Na", "X"], "O": "O", "Si": ["Si", "P"]}
    )

    active_model = model.get_active_lattice_structure()

    assert model.active_site_order.active_site_count == 3
    assert len(active_model.template_structure) == 3
    assert hasattr(active_model, "source_active_site_order")
