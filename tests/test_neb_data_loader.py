import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from kmcpy.io import NEBDataLoader
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
from kmcpy.structure.local_lattice_structure import LocalLatticeStructure


def _build_reference_model():
    lattice = Lattice.cubic(10.0)
    structure = Structure(
        lattice,
        ["Na", "Na", "Cl"],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        coords_are_cartesian=True,
    )
    local_lattice = LocalLatticeStructure(
        template_structure=structure,
        center=[0, 0, 0],
        cutoff=2.5,
        site_mapping={"Na": ["Na", "X"], "Cl": ["Cl"]},
        basis_type="chebyshev",
    )
    model = LocalClusterExpansion()
    model.build(local_lattice, cutoff_cluster=[3.0, 0.0, 0.0])
    return model, local_lattice, structure


def test_loader_adds_structures_directly_and_writes_fit_inputs(tmp_path):
    model, _, structure = _build_reference_model()
    vacancy_structure = structure.copy()
    vacancy_structure.remove_sites([1])

    loader = NEBDataLoader(model=model)
    loader.add_structure(structure, 100.0)
    loader.add_structure(vacancy_structure, 120.0)

    corr = loader.get_correlation_matrix()
    assert corr.shape == (2, len(model.cluster_site_indices))
    assert loader.get_occupation_matrix().shape == (2, 2)
    assert not np.array_equal(corr[0], corr[1])

    paths = loader.write_fitting_inputs(tmp_path, weight=[0.5, 2.0])

    assert set(paths) == {"corr_fname", "ekra_fname", "weight_fname"}
    np.testing.assert_allclose(np.loadtxt(paths["corr_fname"]), corr)
    np.testing.assert_allclose(np.loadtxt(paths["ekra_fname"]), [100.0, 120.0])
    np.testing.assert_allclose(np.loadtxt(paths["weight_fname"]), [0.5, 2.0])


def test_loader_uses_reference_for_serialized_lce_model():
    model, local_lattice, structure = _build_reference_model()
    serialized_model = LocalClusterExpansion.from_dict(model.as_dict())

    with pytest.raises(ValueError, match="reference LocalLatticeStructure"):
        NEBDataLoader(model=serialized_model).add_structure(structure, 100.0)

    loader = NEBDataLoader(
        model=serialized_model,
        reference_local_lattice_structure=local_lattice,
    )
    loader.add_structure(structure, 100.0)

    assert loader.get_correlation_matrix().shape == (
        1,
        len(serialized_model.cluster_site_indices),
    )


def test_lce_computes_correlation_vector_from_structure_with_reference():
    model, local_lattice, structure = _build_reference_model()
    serialized_model = LocalClusterExpansion.from_dict(model.as_dict())
    vacancy_structure = structure.copy()
    vacancy_structure.remove_sites([1])

    occupation, correlation = serialized_model.get_occ_corr_from_structure(
        vacancy_structure,
        reference_local_lattice_structure=local_lattice,
    )

    assert len(occupation) == 2
    assert correlation.shape == (len(serialized_model.cluster_site_indices),)


def test_lce_correlation_vector_reflects_allowed_substitution():
    lattice = Lattice.cubic(10.0)
    structure = Structure(
        lattice,
        ["Si", "Na"],
        [[0, 0, 0], [1, 0, 0]],
        coords_are_cartesian=True,
    )
    local_lattice = LocalLatticeStructure(
        template_structure=structure,
        center=[0, 0, 0],
        cutoff=2.0,
        site_mapping={"Si": ["Si", "P"], "Na": ["Na", "X"]},
        basis_type="chebyshev",
    )
    model = LocalClusterExpansion()
    model.build(local_lattice, cutoff_cluster=[3.0, 0.0, 0.0])
    substituted_structure = Structure(
        lattice,
        ["P", "Na"],
        [[0, 0, 0], [1, 0, 0]],
        coords_are_cartesian=True,
    )

    original_corr = model.get_corr_from_structure(structure)
    substituted_corr = model.get_corr_from_structure(substituted_structure)

    assert not np.array_equal(original_corr, substituted_corr)


def test_loader_rejects_mismatched_reference_ordering():
    structure = Structure(
        Lattice.cubic(10.0),
        ["Na", "Na", "Na"],
        [[0, 0, 0], [0, 1, 0], [1, 0, 0]],
        coords_are_cartesian=True,
    )
    local_lattice = LocalLatticeStructure(
        template_structure=structure,
        center=[0, 0, 0],
        cutoff=2.5,
        site_mapping={"Na": ["Na", "X"]},
        basis_type="chebyshev",
    )
    model = LocalClusterExpansion()
    model.build(local_lattice, cutoff_cluster=[3.0, 0.0, 0.0])
    serialized_model = LocalClusterExpansion.from_dict(model.as_dict())
    mismatched_reference = LocalLatticeStructure(
        template_structure=structure,
        center=[0, 0, 0],
        cutoff=2.5,
        site_mapping={"Na": ["Na", "X"]},
        basis_type="chebyshev",
        ordering_convention={
            "name": "by_x",
            "sort_keys": ["cartesian_x", "cartesian_y"],
        },
    )

    loader = NEBDataLoader(
        model=serialized_model,
        reference_local_lattice_structure=mismatched_reference,
    )

    with pytest.raises(ValueError, match="ordering does not match"):
        loader.add_structure(structure, 100.0)


def test_loader_uses_reference_active_site_mapping():
    lattice = Lattice.cubic(10.0)
    structure = Structure(
        lattice,
        ["Na", "Na", "Cl"],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        coords_are_cartesian=True,
    )
    local_lattice = LocalLatticeStructure(
        template_structure=structure,
        center=[0, 0, 0],
        cutoff=2.5,
        site_mapping={"Na": ["Na", "X"], "Cl": ["Cl"]},
        basis_type="chebyshev",
    )
    model = LocalClusterExpansion()
    model.build(local_lattice, cutoff_cluster=[3.0, 0.0, 0.0])

    loader = NEBDataLoader(model=model)
    loader.add_structure(structure, 100.0)

    np.testing.assert_allclose(
        loader.get_correlation_matrix()[0],
        model.get_corr_from_structure(structure),
    )
    assert loader.get_occupation_matrix().shape == (1, 2)


def test_loader_builds_from_structure_files(tmp_path):
    model, _, structure = _build_reference_model()
    structure_file = tmp_path / "neb_0001.cif"
    structure.to(filename=str(structure_file), fmt="cif")

    loader = NEBDataLoader.from_structure_files(
        [structure_file],
        [95.0],
        model=model,
    )

    assert len(loader) == 1
    assert loader.neb_entries[0].metadata["structure_file"] == str(structure_file)
    np.testing.assert_allclose(loader.get_properties(), [95.0])
