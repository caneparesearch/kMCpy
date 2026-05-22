import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.transformations.standard_transformations import (
    OrderDisorderedStructureTransformation,
)

from kmcpy.event import Event
from kmcpy.structure import (
    LatticeStructure,
    LocalEnvironmentEnumeration,
    NEBEndpointPair,
    enumerate_local_environments,
    enumerate_neb_endpoint_pairs,
    generate_neb_endpoint_pair,
)
from kmcpy.structure.basis import Occupation


def _chemical_lattice_model():
    lattice = Lattice.cubic(10.0)
    structure = Structure(
        lattice,
        ["Na", "Si", "Si", "Cl"],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [5, 0, 0]],
        coords_are_cartesian=True,
    )
    return LatticeStructure(
        template_structure=structure,
        specie_site_mapping={
            "Na": ["Na", "X"],
            "Si": ["Si", "P"],
            "Cl": "Cl",
        },
    )


def _mobile_lattice_model():
    lattice = Lattice.cubic(10.0)
    structure = Structure(
        lattice,
        ["Na", "Na", "Na", "Cl"],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [5, 0, 0]],
        coords_are_cartesian=True,
    )
    return LatticeStructure(
        template_structure=structure,
        specie_site_mapping={"Na": ["Na", "X"], "Cl": "Cl"},
    )


def test_enumerates_local_environments_in_deterministic_order():
    model = _chemical_lattice_model()

    results = enumerate_local_environments(
        model,
        center=0,
        cutoff=2.5,
        variable_species=["Si", "P"],
        variable_site_indices=[1, 2],
    )

    assert [result.label for result in results] == [
        "1:Si|2:Si",
        "1:Si|2:P",
        "1:P|2:Si",
        "1:P|2:P",
    ]
    assert all(isinstance(result, LocalEnvironmentEnumeration) for result in results)
    assert results[0].full_occupation.array_equal([-1, -1, -1, -1])
    assert results[-1].full_occupation.array_equal([-1, 1, 1, -1])
    assert results[1].local_site_indices == tuple(results[1].local_site_indices)
    assert results[1].variable_site_indices == (1, 2)


def test_enumeration_filters_by_exact_species_counts():
    model = _chemical_lattice_model()

    results = enumerate_local_environments(
        model,
        center=0,
        cutoff=2.5,
        variable_species=["Si", "P"],
        variable_site_indices=[1, 2],
        species_counts={"Si": 1, "P": 1},
    )

    assert [result.label for result in results] == ["1:Si|2:P", "1:P|2:Si"]


def test_enumeration_handles_vacancy_sites():
    model = _mobile_lattice_model()

    results = enumerate_local_environments(
        model,
        center=0,
        cutoff=2.5,
        variable_species=["Na", "X"],
        variable_site_indices=[0, 1],
        species_counts={"Na": 1, "X": 1},
    )

    assert [result.label for result in results] == ["0:Na|1:X", "0:X|1:Na"]
    assert [len(result.structure) for result in results] == [3, 3]
    assert results[0].full_occupation.array_equal([-1, 1, -1, -1])
    assert results[1].full_occupation.array_equal([1, -1, -1, -1])


def test_enumeration_excludes_species_without_renumbering_sites():
    lattice = Lattice.cubic(10.0)
    structure = Structure(
        lattice,
        ["Na", "O", "Si"],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        coords_are_cartesian=True,
    )
    model = LatticeStructure(
        template_structure=structure,
        specie_site_mapping={"Na": ["Na", "X"], "O": "O", "Si": ["Si", "P"]},
    )

    results = enumerate_local_environments(
        model,
        center=0,
        cutoff=3.0,
        variable_species=["Si", "P"],
        exclude_species=["O2-"],
    )

    assert [result.label for result in results] == ["2:Si", "2:P"]
    assert all(result.local_site_indices == (0, 2) for result in results)
    assert all(result.variable_site_indices == (2,) for result in results)


def test_enumeration_uses_provided_pymatgen_transformation():
    model = _chemical_lattice_model()
    transformation = OrderDisorderedStructureTransformation(no_oxi_states=True)

    results = enumerate_local_environments(
        model,
        center=0,
        cutoff=2.5,
        variable_species=["Si", "P"],
        variable_site_indices=[1, 2],
        species_counts={"Si": 1, "P": 1},
        transformation=transformation,
        max_results=10,
    )

    labels = {result.label for result in results}
    assert labels == {"1:Si|2:P", "1:P|2:Si"}
    assert all(result.metadata["source"] == "transformation" for result in results)


def test_neb_endpoint_pair_uses_mobile_ion_indices_and_preserves_atom_order():
    model = _mobile_lattice_model()
    occupation = Occupation([-1, -1, 1, -1], basis=model.basis)
    generated_event = Event(mobile_ion_indices=(0, 2), local_env_indices=(0, 1, 2))

    pair = generate_neb_endpoint_pair(model, occupation, generated_event)

    assert isinstance(pair, NEBEndpointPair)
    assert pair.mobile_ion_indices == (0, 2)
    assert pair.initial_occupation.array_equal([-1, -1, 1, -1])
    assert pair.final_occupation.array_equal([1, -1, -1, -1])
    assert [site.species_string for site in pair.initial] == ["Na", "Na", "Cl"]
    assert [site.species_string for site in pair.final] == ["Na", "Na", "Cl"]
    np.testing.assert_allclose(pair.initial.cart_coords[0], [0, 0, 0])
    np.testing.assert_allclose(pair.final.cart_coords[0], [2, 0, 0])
    np.testing.assert_allclose(pair.initial.cart_coords[1], [1, 0, 0])
    np.testing.assert_allclose(pair.final.cart_coords[1], [1, 0, 0])


def test_neb_endpoint_pair_accepts_enumeration_and_index_tuple():
    model = _mobile_lattice_model()
    enumeration = enumerate_local_environments(
        model,
        center=0,
        cutoff=2.5,
        variable_species=["Na", "X"],
        variable_site_indices=[1, 2],
        species_counts={"Na": 1, "X": 1},
    )[0]

    pair = generate_neb_endpoint_pair(model, enumeration, (1, 2))

    assert pair.mobile_ion_indices == (1, 2)
    assert pair.initial_occupation[1] == model.basis.match_value
    assert pair.initial_occupation[2] == model.basis.mismatch_value
    assert pair.final_occupation[1] == model.basis.mismatch_value
    assert pair.final_occupation[2] == model.basis.match_value


def test_enumerate_neb_endpoint_pairs_merges_enumeration_and_endpoint_building():
    model = _mobile_lattice_model()
    generated_event = Event(mobile_ion_indices=(1, 2), local_env_indices=(0, 1, 2))

    pairs = enumerate_neb_endpoint_pairs(
        model,
        mobile_ion_indices=generated_event,
        cutoff=2.5,
        variable_species=["Na", "X"],
        variable_site_indices=[1, 2],
        species_counts={"Na": 1, "X": 1},
    )

    assert len(pairs) == 2
    assert [pair.metadata["local_environment_label"] for pair in pairs] == [
        "1:Na|2:X",
        "1:X|2:Na",
    ]
    assert all(pair.mobile_ion_indices == (1, 2) for pair in pairs)
    assert all(pair.initial_occupation[1] == model.basis.match_value for pair in pairs)
    assert all(pair.initial_occupation[2] == model.basis.mismatch_value for pair in pairs)
    assert all(pair.final_occupation[1] == model.basis.mismatch_value for pair in pairs)
    assert all(pair.final_occupation[2] == model.basis.match_value for pair in pairs)


def test_neb_endpoint_pair_rejects_invalid_hops():
    model = _mobile_lattice_model()
    occupation = Occupation([-1, -1, 1, -1], basis=model.basis)

    with pytest.raises(ValueError, match="two distinct"):
        generate_neb_endpoint_pair(model, occupation, (0, 0))

    with pytest.raises(ValueError, match="same first allowed mobile species"):
        generate_neb_endpoint_pair(model, occupation, (0, 3))
