import pytest
import numpy as np
from pymatgen.core import Structure, Lattice
from kmcpy.structure.lattice_structure import LatticeStructure
from kmcpy.structure.local_lattice_structure import LocalLatticeStructure
from kmcpy.structure.local_site_ordering import LocalSiteOrderingConvention

@pytest.fixture
def global_lattice_model_and_env():
    """A fixture to create a global LatticeStructure and a LocalLatticeStructure for testing."""
    # Use a large lattice to avoid periodic boundary issues in the test
    lattice = Lattice.cubic(10.0)
    template_structure = Structure(
        lattice,
        ["Na", "Cl", "Na", "Br"],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        coords_are_cartesian=True
    )
    site_mapping = {"Na": ["Na", "X"], "Cl": ["Cl"], "Br": ["Br"]}  # X represents a vacancy
    
    # Create the global model
    global_model = LatticeStructure(
        template_structure=template_structure,
        site_mapping=site_mapping
    )
    
    # Create a local environment centered on the first Na atom
    local_env = LocalLatticeStructure(
        template_structure=template_structure,
        site_mapping=site_mapping,
        center=[0, 0, 0],
        cutoff=1.5
    )
    
    return global_model, local_env

def test_local_environment_setup(global_lattice_model_and_env):
    """Test that the LocalLatticeStructure is set up correctly."""
    _, local_env = global_lattice_model_and_env
    
    # Fixed Cl/Br sites are excluded from the compact active-site space.
    assert len(local_env.structure) == 1
    assert local_env.structure[0].species_string == "Na"
    assert local_env.site_indices == [0]


def test_nasicon_publication_ordering_excludes_center_and_sorts_by_species_then_x():
    """Publication convention should mimic the single-unit NASICON site order."""
    lattice = Lattice.cubic(20.0)
    template_structure = Structure(
        lattice,
        ["Na", "Na", "Na", "Si", "Si"],
        [
            [5, 5, 5],
            [4, 5, 5],
            [6, 5, 5],
            [3, 5, 5],
            [7, 5, 5],
        ],
        coords_are_cartesian=True,
    )

    local_env = LocalLatticeStructure(
        template_structure=template_structure,
        site_mapping={"Na": ["Na", "X"], "Si": ["Si", "P"]},
        center=0,
        cutoff=3.1,
        ordering_convention="nasicon_nat_commun_2022",
    )

    assert local_env.ordering_convention.name == "nasicon_nat_commun_2022"
    assert local_env.site_indices == [1, 2, 3, 4]
    assert [site.species_string for site in local_env.structure] == [
        "Na",
        "Na",
        "Si",
        "Si",
    ]
    assert [round(float(site.coords[0]), 3) for site in local_env.structure] == [
        -1.0,
        1.0,
        -2.0,
        2.0,
    ]


def test_local_site_ordering_convention_round_trip():
    convention = LocalSiteOrderingConvention.from_name("nasicon_nat_commun_2022")

    restored = LocalSiteOrderingConvention.resolve(convention.as_dict())
    restored_from_name_only = LocalSiteOrderingConvention.resolve(
        {"name": "nasicon_nat_commun_2022"}
    )

    assert restored == convention
    assert restored_from_name_only == convention
    assert restored.exclude_center_site
    assert restored.sort_keys == ("species", "cartesian_x")

def test_get_local_occupation(global_lattice_model_and_env):
    """Test getting occupation for a local environment from a full structure."""
    global_model, local_env = global_lattice_model_and_env
    
    # This structure is identical to the template
    structure_to_test = global_model.template_structure.copy()
    
    # 1. Get the full occupation vector from the global model
    full_occ = global_model.get_occ_from_structure(structure_to_test)
    
    # 2. Get the local occupation by slicing the full vector
    local_occ = full_occ[local_env.site_indices]
    
    # The local environment is perfect (identical to template), so all sites should be occupied
    expected_local_occ = [
        global_model.basis.occupied_value
    ]
    
    assert local_occ.array_equal(expected_local_occ)

def test_get_local_occupation_with_vacancy(global_lattice_model_and_env):
    """Test local occupation when there's a vacancy in the local environment."""
    global_model, local_env = global_lattice_model_and_env
    
    # Create a structure where the active Na site is missing.
    structure_with_vacancy = global_model.template_structure.copy()
    structure_with_vacancy.remove_sites([0])

    active_map = global_model.get_active_site_index_map()
    active_model = global_model.get_active_lattice_structure()
    active_structure = active_map.filter_active_structure(structure_with_vacancy)
    full_occ = active_model.get_occ_from_structure(active_structure)
    local_occ = full_occ[local_env.site_indices]

    expected_local_occ = [
        global_model.basis.vacant_value
    ]
    
    assert local_occ.array_equal(expected_local_occ)


def test_local_lattice_structure_does_not_mutate_input_structure():
    """LocalLatticeStructure should not mutate the structure passed by the caller."""
    lattice = Lattice.cubic(10.0)
    template_structure = Structure(
        lattice,
        ["Na", "Cl", "Na"],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        coords_are_cartesian=True,
    )
    site_mapping = {"Na": ["Na", "X"], "Cl": ["Cl"]}

    original_len = len(template_structure)
    original_species = [site.species_string for site in template_structure]

    LocalLatticeStructure(
        template_structure=template_structure,
        site_mapping=site_mapping,
        center=[0, 0, 0],
        cutoff=2.5,
    )

    assert len(template_structure) == original_len
    assert [site.species_string for site in template_structure] == original_species


def test_local_lattice_structure_removes_fixed_sites_from_active_space():
    lattice = Lattice.cubic(10.0)
    template_structure = Structure(
        lattice,
        ["Na", "O", "Si"],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        coords_are_cartesian=True,
    )

    local_lattice = LocalLatticeStructure(
        template_structure=template_structure,
        site_mapping={"Na": ["Na", "X"], "O": "O", "Si": ["Si", "P"]},
        center=[0, 0, 0],
        cutoff=3.0,
    )

    substituted_structure = Structure(
        lattice,
        ["Na", "O", "P"],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        coords_are_cartesian=True,
    )
    active_structure = local_lattice.active_site_index_map.filter_active_structure(
        substituted_structure
    )
    occ = local_lattice.get_occ_from_structure(active_structure)

    assert len(local_lattice.allowed_species) == len(local_lattice.template_structure)
    assert occ.array_equal([
        local_lattice.basis.match_value,
        local_lattice.basis.mismatch_value,
    ])


def test_local_lattice_structure_accepts_neutral_mapping_for_oxidized_template():
    lattice = Lattice.cubic(10.0)
    template_structure = Structure(
        lattice,
        ["Na", "O", "Si"],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        coords_are_cartesian=True,
    )
    template_structure.add_oxidation_state_by_element({"Na": 1, "O": -2, "Si": 4})

    local_lattice = LocalLatticeStructure(
        template_structure=template_structure,
        site_mapping={"Na": ["Na", "X"], "O": "O", "Si": ["Si", "P"]},
        center=0,
        cutoff=3.0,
    )

    substituted_structure = Structure(
        lattice,
        ["Na", "O", "P"],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        coords_are_cartesian=True,
    )
    substituted_structure.add_oxidation_state_by_element({"Na": 1, "O": -2, "P": 5})
    substituted_structure.remove_oxidation_states()

    active_structure = local_lattice.active_site_index_map.filter_active_structure(
        substituted_structure
    )
    occ = local_lattice.get_occ_from_structure(active_structure)

    assert all(species is not None for species in local_lattice.allowed_species)
    assert occ.array_equal([
        local_lattice.basis.match_value,
        local_lattice.basis.mismatch_value,
    ])


def test_local_lattice_structure_infers_fixed_oxidized_species():
    lattice = Lattice.cubic(10.0)
    template_structure = Structure(
        lattice,
        ["Na", "O", "Si"],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        coords_are_cartesian=True,
    )
    template_structure.add_oxidation_state_by_element({"Na": 1, "O": -2, "Si": 4})

    local_lattice = LocalLatticeStructure(
        template_structure=template_structure,
        site_mapping={"Na": ["Na", "X"], "O": "O", "Si": ["Si", "P"]},
        center=0,
        cutoff=3.0,
    )

    assert [site.species_string for site in local_lattice.template_structure] == [
        "Na",
        "Si",
    ]
    assert local_lattice.site_indices == [0, 1]
    assert len(local_lattice.allowed_species) == 2


def test_sort_neighbor_info_preserves_metadata():
    """Neighbor sorting helper should preserve metadata while applying deterministic order."""
    lattice = Lattice.cubic(10.0)
    structure = Structure(
        lattice,
        ["Na", "Cl", "Br"],
        [[2, 0, 0], [1, 0, 0], [0, 0, 0]],
        coords_are_cartesian=True,
    )

    unsorted_neighbors = [
        {"site": structure[0], "image": (0, 0, 0), "local_index": 2, "label": "Na1"},
        {"site": structure[2], "image": (1, 0, 0), "local_index": 0, "label": "Br1"},
        {"site": structure[1], "image": (0, 1, 0), "local_index": 1, "label": "Cl1"},
    ]

    sorted_neighbors = LocalLatticeStructure.sort_neighbor_info(unsorted_neighbors)
    expected_neighbors = sorted(
        unsorted_neighbors, key=lambda x: (x["site"].specie, x["site"].coords[0])
    )

    assert [id(n) for n in sorted_neighbors] == [id(n) for n in expected_neighbors]
    for neighbor in sorted_neighbors:
        assert "label" in neighbor
        assert "image" in neighbor
        assert "local_index" in neighbor
