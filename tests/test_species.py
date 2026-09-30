"""Tests for species normalization and ``SiteMapping``."""

import pytest
from pymatgen.core import DummySpecies, Lattice, Species, Structure

from kmcpy.event import EventGenerator, HopStateLookup
from kmcpy.structure import SiteMapping
from kmcpy.structure.active_site_order import ActiveSiteOrder
from kmcpy.structure.lattice_structure import LatticeStructure
from kmcpy.structure.species import is_vacancy_species


@pytest.mark.parametrize(
    "value",
    ["X", "x", "Va", "VA", "va", "Vacancy", "vacancy", "VACANCY", DummySpecies("X")],
)
def test_vacancy_labels_are_case_insensitive(value):
    assert is_vacancy_species(value)


@pytest.mark.parametrize("value", ["V", "Na", "Xe", Species("V"), Species("Na", 1)])
def test_elements_are_not_vacancies(value):
    assert not is_vacancy_species(value)


def _na_zr_structure():
    return Structure(
        Lattice.cubic(4.0),
        ["Na", "Na", "Zr"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0]],
    )


def test_site_mapping_normalizes_entries_and_matches_sites():
    mapping = SiteMapping({"Na": ["Na", "Va"], "Zr": "Zr"})

    assert mapping.as_dict() == {
        Species("Na"): [Species("Na"), DummySpecies("X")],
        Species("Zr"): [Species("Zr")],
    }
    allowed = mapping.allowed_species_by_site(_na_zr_structure())
    assert allowed == [
        (Species("Na"), DummySpecies("X")),
        (Species("Na"), DummySpecies("X")),
        (Species("Zr"),),
    ]
    assert SiteMapping(mapping).as_dict() == mapping.as_dict()


def test_site_mapping_strictness_for_unmapped_sites():
    mapping = SiteMapping({"Na": ["Na", "X"]})
    structure = _na_zr_structure()

    with pytest.raises(ValueError, match="No site_mapping entry"):
        mapping.allowed_species_by_site(structure)
    assert mapping.allowed_species_by_site(structure, strict=False)[2] is None


@pytest.mark.parametrize("vacancy_label", ["X", "Va", "VA", "vacancy", "VACANCY"])
def test_mobile_species_accepts_every_vacancy_label(vacancy_label):
    site_mapping = {"Na": ["Na", vacancy_label], "Si": ["Si", "P"], "O": "O"}

    assert SiteMapping(site_mapping).mobile_species() == ["Na"]
    # Previously "Va"/"VA" failed with "Could not infer mobile species".
    assert EventGenerator._mobile_species_from_site_mapping(site_mapping) == ["Na"]


def test_lattice_structure_and_active_site_order_share_site_mapping():
    structure = _na_zr_structure()
    site_mapping = {"Na": ["Na", "Va"], "Zr": "Zr"}

    lattice_structure = LatticeStructure(structure, site_mapping)
    active_site_order = ActiveSiteOrder.from_structure_and_mapping(
        structure, site_mapping
    )

    assert lattice_structure.allowed_species[:2] == [
        [Species("Na"), DummySpecies("X")],
        [Species("Na"), DummySpecies("X")],
    ]
    assert active_site_order.allowed_species_by_primitive_site[0] == ("Na", "X")
    lookup = HopStateLookup.from_active_site_order(active_site_order, "Na")
    assert lookup.mobile_state_by_site.tolist() == [0, 0]
    assert lookup.vacancy_state_by_site.tolist() == [1, 1]
