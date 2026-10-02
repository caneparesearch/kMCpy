"""LCE workflow pieces: in-memory fitting and construction from a LatticeStructure."""

from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

import kmcpy
from kmcpy.io.neb import NEBDataLoader
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
from kmcpy.structure.local_environment_enumerator import (
    LocalEnvironmentEnumerator,
    enumerate_local_environments,
)
from kmcpy.structure.local_lattice_structure import LocalLatticeStructure

FILES = Path(__file__).parent / "files"
FIT_DATA = FILES / "fitting" / "local_cluster_expansion"
SITE_MAPPING = {"Na": ["Na", "X"], "Zr": "Zr", "Si": ["Si", "P"], "O": "O"}


@pytest.fixture(scope="module")
def lattice():
    return kmcpy.LatticeStructure.from_cif(
        FILES / "EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif", SITE_MAPPING, primitive=True
    )


@pytest.mark.unit
def test_fit_data_matches_file_based_fit(tmp_path):
    files = dict(
        ekra_fname=str(FIT_DATA / "e_kra.txt"),
        weight_fname=str(FIT_DATA / "weight.txt"),
        corr_fname=str(FIT_DATA / "correlation_matrix.txt"),
    )
    from_files, predicted_files, _ = LocalClusterExpansion().fit(
        alpha=1.5,
        keci_fname=str(tmp_path / "keci.txt"),
        lce_params_fname=None,
        lce_params_history_fname=None,
        **files,
    )

    model = LocalClusterExpansion()
    in_memory, predicted, targets = model.fit_data(
        np.loadtxt(files["corr_fname"]),
        np.loadtxt(files["ekra_fname"]),
        weights=np.loadtxt(files["weight_fname"]),
        alpha=1.5,
    )

    np.testing.assert_array_equal(in_memory.keci, from_files.keci)
    np.testing.assert_array_equal(predicted, predicted_files)
    np.testing.assert_array_equal(targets, np.loadtxt(files["ekra_fname"]))
    assert model.has_parameters() and model.keci == list(in_memory.keci)


@pytest.mark.unit
def test_neb_loader_fit_attaches_parameters():
    structure = Structure(
        Lattice.cubic(10.0),
        ["Na", "Na", "Na", "Cl"],
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]],
        coords_are_cartesian=True,
    )
    local_lattice = LocalLatticeStructure(
        template_structure=structure,
        center=[0, 0, 0],
        cutoff=1.5,
        site_mapping={"Na": ["Na", "X"], "Cl": ["Cl"]},
    )
    model = LocalClusterExpansion()
    model.build(local_lattice, cutoff_cluster=[2.0, 0.0, 0.0])

    loader = NEBDataLoader(model=model)
    # Sites 1 and 2 are symmetry-equivalent, so they share a barrier.
    for removed, barrier in (([], 300.0), ([1], 330.0), ([2], 330.0), ([1, 2], 360.0)):
        variant = structure.copy()
        variant.remove_sites(removed)
        loader.add_structure(variant, barrier)

    parameters, predicted, targets = loader.fit(alpha=1e-6)

    assert model.has_parameters()
    np.testing.assert_allclose(predicted, targets, atol=1.0)
    np.testing.assert_allclose(
        loader.get_correlation_matrix() @ np.array(model.keci) + model.empty_cluster,
        predicted,
    )


@pytest.mark.unit
def test_local_environments_build_from_the_lattice(lattice):
    direct = LocalLatticeStructure(
        template_structure=lattice.template_structure, center=0, cutoff=4.0, site_mapping=SITE_MAPPING
    )
    from_lattice = LocalLatticeStructure.from_lattice_structure(lattice, center=0, cutoff=4.0)
    assert from_lattice.site_indices == direct.site_indices
    assert from_lattice.local_environment_hash == direct.local_environment_hash

    # An event of the supercell can be the center (its first mobile-ion site).
    supercell = lattice.make_supercell((2, 1, 1), in_place=False)
    events = kmcpy.HopEvents(
        cutoffs={("Na+", "Na+"): 4.0, ("Na+", "Si4+"): 4.0}, labels=("Na1", "Na2")
    ).generate(supercell)
    event = events.events[5]
    center = supercell.active_site_order.active_to_primitive[event.mobile_ion_indices[0]]
    assert (
        LocalLatticeStructure.from_lattice_structure(supercell, center=event, cutoff=4.0).local_environment_hash
        == LocalLatticeStructure.from_lattice_structure(supercell, center=center, cutoff=4.0).local_environment_hash
    )

    options = dict(center=0, cutoff=4.0, variable_species=["Si", "P"], max_results=5)
    from_enumerator = LocalEnvironmentEnumerator(lattice).enumerate(**options)
    from_function = enumerate_local_environments(lattice, **options)
    assert [result.label for result in from_enumerator] == [result.label for result in from_function]
    # On a supercell, enumeration uses the supercell's active-site indices.
    assert LocalEnvironmentEnumerator(supercell).active_site_order.active_site_count == supercell.n_active_sites
