#!/usr/bin/env python
"""
This module provides classes and functions to build a Local Cluster Expansion (LCE) model for kinetic Monte Carlo (KMC) simulations, particularly for ionic conductors such as NaSICON materials. The main class, `LocalClusterExpansion`, reads a crystal structure file (e.g., CIF format), processes the structure to define a local migration unit, and generates clusters (points, pairs, triplets, quadruplets) within a specified cutoff. The clusters are grouped into orbits based on symmetry, and the resulting model can be serialized to JSON for use in KMC simulations.
"""
from typing import Literal
from itertools import combinations
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from pymatgen.core.structure import Molecule
from kmcpy.external.structure import StructureKMCpy
import numpy as np
import json
import glob
from kmcpy.event_generator import find_atom_indices
from pymatgen.core.periodic_table import DummySpecies
from pymatgen.core.sites import PeriodicSite
import logging
import kmcpy
from kmcpy.model.model import BaseModel
from copy import deepcopy
import numba as nb
from kmcpy.io import InputSet

logger = logging.getLogger(__name__) 
logging.getLogger('pymatgen').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

class LocalClusterExpansion(BaseModel):
    """
    LocalClusterExpansion will be initialized with a template structure where all the sites are occupied
    cutoff_cluster is the cutoff for pairs and triplet
    cutoff_region is the cutoff for generating local cluster region
    """

    def __init__(self, template_structure_fname:str, mobile_ion_specie_1_identifier:str,
        cutoff_cluster: list = [6, 6, 6], cutoff_region:float = 4.0,
        center_frac_coord = [], mobile_ion_identifier_type: Literal["specie","label"] = "label",
        is_write_basis=False, species_to_be_removed=[], convert_to_primitive_cell=False,
        exclude_site_with_identifier=[], **kwargs) -> None:
        """
        Initialization of the LocalClusterExpansion object.

        There are 2 ways to define the local environment (migration unit):
        1) Use the center of the mobile ion as the center of the local environment (default, center_frac_coord = []), this mobile ion is excluded from the local environment.
        2) Use a dummy site as the center of the local environment (set center_frac_coord).

        Args:
            template_structure_fname (str): Path to the template structure file (e.g., CIF file).
            mobile_ion_specie_1_identifier (str): Identifier for the mobile ion species (e.g., "Na1").
            cutoff_cluster (list, optional): Cutoff distances for clusters [pair, triplet, quadruplet]. Defaults to [6, 6, 6].
            cutoff_region (float, optional): Cutoff for generating the local cluster region. Defaults to 4.0.
            center_frac_coord (list, optional): Fractional coordinates of the center of the local environment. If empty, uses the mobile ion site. Defaults to [].
            mobile_ion_identifier_type (Literal["specie", "label"], optional): Type of identifier for the mobile ion ("specie" or "label"). Defaults to "label".
            is_write_basis (bool, optional): Whether to write the local environment basis to file. Defaults to False.
            species_to_be_removed (list, optional): List of species to remove from the structure. Defaults to [].
            convert_to_primitive_cell (bool, optional): Whether to convert the structure to its primitive cell. Defaults to False.
            exclude_site_with_identifier (list, optional): List of site identifiers to exclude from the local environment. Defaults to [].

        Notes:
            If the next-step KMC is not based on the same LCE object generated in this step, be careful with two things:
            1) The Ekra generated in this step can be transferred to the KMC, provided the orbits are arranged in the same way as here.
            2) The sublattice_indices must correspond exactly to the input structure used in the KMC step, which may require reconstruction of an LCE object using the same KMC-input structure.
        """
        logger.info(kmcpy.get_logo())
        logger.info("Initializing LocalClusterExpansion ...")
        self.name = "LocalClusterExpansion"
        template_structure = StructureKMCpy.from_cif(
            template_structure_fname, primitive=convert_to_primitive_cell
        )
        template_structure.remove_oxidation_states()
        template_structure.remove_species(species_to_be_removed)

        mobile_ion_specie_1_indices = find_atom_indices(
            template_structure,
            mobile_ion_identifier_type=mobile_ion_identifier_type,
            atom_identifier=mobile_ion_specie_1_identifier,
        )

        mobile_ion_specie_1_indices=mobile_ion_specie_1_indices[0]# just use the first one 
        
        if center_frac_coord:
            logger.info(f"Centering the local environment at {center_frac_coord} ...")
            
            dummy_lattice = template_structure.lattice
            self.center_site = PeriodicSite(species=DummySpecies('X'),
                              coords=center_frac_coord,
                              coords_are_cartesian=False,
                              lattice = dummy_lattice)
            logger.debug(f"Dummy site: {self.center_site}")
        else:
            logger.info(f"Centering the local environment at {mobile_ion_specie_1_indices} ...")
            self.center_site = template_structure[
                mobile_ion_specie_1_indices
            ]  # self.center_site: pymatgen.site

            template_structure.remove_sites([mobile_ion_specie_1_indices]) # remove the mobile ion from the template structure

        logger.info(f"Searching local env around {self.center_site} ...")

        # fallback to the initial get cluster structure
        self.MigrationUnit_structure = self.get_cluster_structure(
            structure=template_structure,
            cutoff=cutoff_region,
            center_site=self.center_site,
            is_write_basis=is_write_basis,
            exclude_species=exclude_site_with_identifier,
        )

        # List all possible point, pair and triplet clusters
        atom_index_list = np.arange(0, len(self.MigrationUnit_structure))

        cluster_indexes = (
            list(combinations(atom_index_list, 1))
            + list(combinations(atom_index_list, 2))
            + list(combinations(atom_index_list, 3))
            + list(combinations(atom_index_list, 4))
        )

        logger.info(f"{len(cluster_indexes)} clusters will be generated ...")

        self.clusters = self.clusters_constructor(
            cluster_indexes, [10] + cutoff_cluster
        )

        self.orbits = self.orbits_constructor(self.clusters)

        self.sublattice_indices = [
            [cluster.site_indices for cluster in orbit.clusters]
            for orbit in self.orbits
        ]  # sublattice_indices[orbit,cluster,site]
        logger.info(
            "Type\tIndex\tmax_length\tmin_length\tPoint Group\tMultiplicity"
        )
        for orbit in self.orbits:
            orbit.show_representative_cluster()

    @classmethod
    def from_inputset(cls, inputset: InputSet)-> "LocalClusterExpansion":
        params = {k: v for k, v in inputset._parameters.items() if k != "task"}
        return cls(**params)
    
    def get_cluster_structure(
        self,
        structure,
        center_site,
        cutoff=4,
        is_write_basis=False,
        exclude_species=["Li"],
    ):  # return a molecule structure centeret center_site
        local_env_structure = [
            s[0] for s in structure.get_sites_in_sphere(center_site.coords, cutoff)
        ]
        local_env_list_sorted = sorted(
            sorted(local_env_structure, key=lambda x: x.coords[0]),
            key=lambda x: x.specie,
        )
        local_env_list_sorted_involved = []
        for site in local_env_list_sorted:
            excluded = False
            for exclude_specie in exclude_species:
                if exclude_specie in site.species:
                    excluded = True
            if not excluded:
                local_env_list_sorted_involved.append(site)

        local_env_structure = Molecule.from_sites(local_env_list_sorted_involved)
        local_env_structure.translate_sites(
            np.arange(0, len(local_env_structure), 1).tolist(), -1 * center_site.coords
        )
        if is_write_basis:
            logger.info("Local environment: ")
            logger.info(local_env_structure)
            local_env_structure.to(fmt="xyz", filename="local_env.xyz")
            logger.info(
            "The point group of local environment is: %s",
            PointGroupAnalyzer(local_env_structure).sch_symbol,
            )
        return local_env_structure
    
    def get_occupation_neb_cif(
        self, other_cif_name, species_to_be_removed=["Zr4+", "O2-", "O", "Zr"]
    ):  # input is a cif structure
        occupation = []
        other_structure = StructureKMCpy.from_file(other_cif_name)
        other_structure.remove_oxidation_states()
        other_structure.remove_species(species_to_be_removed)
        other_structure_mol = self.get_cluster_structure(
            other_structure, self.center_site
        )
        for this_site in self.MigrationUnit_structure:
            if self.is_exists(
                this_site, other_structure_mol
            ):  # Chebyshev basis is used here: ±1
                occu = -1
            else:
                occu = 1
            occupation.append(occu)
        return occupation

    def get_correlation_matrix_neb_cif(self, other_cif_names):
        correlation_matrix = []
        occupation_matrix = []
        for other_cif_name in sorted(glob.glob(other_cif_names)):
            occupation = self.get_occupation_neb_cif(other_cif_name)
            correlation = [
                (orbit.multiplicity) * orbit.get_cluster_function(occupation)
                for orbit in self.orbits
            ]
            occupation_matrix.append(occupation)
            correlation_matrix.append(correlation)
            logger.info(f"{other_cif_name}: {occupation}")
        self.correlation_matrix = correlation_matrix

        logger.info(np.round(correlation_matrix, decimals=3))
        np.savetxt(fname="occupation.txt", X=occupation_matrix, fmt="%5d")
        np.savetxt(fname="correlation_matrix.txt", X=correlation_matrix, fmt="%.8f")
        self.correlation_matrix = correlation_matrix
        logger.debug(f"{other_cif_name}\t{occupation}\t{np.around(correlation,decimals=3)}")

    def is_exists(self, this_site, other_structure):
        # 2 things to compare: 1. cartesian coords 2. species at each site
        is_exists = False
        for s_other in other_structure:
            if (np.linalg.norm(this_site.coords - s_other.coords) < 1e-3) and (
                this_site.species == s_other.species
            ):
                is_exists = True
        return is_exists

    def clusters_constructor(self, indexes, cutoff):  # return a list of Cluster
        clusters = []
        logger.info("\nGenerating possible clusters within this migration unit...")
        logger.info(
            "Cutoffs: pair = %s Angst, triplet = %s Angst, quadruplet = %s Angst",
            cutoff[1],
            cutoff[2],
            cutoff[3],
        )
        for site_indices in indexes:
            sites = [self.MigrationUnit_structure[s] for s in site_indices]
            cluster = Cluster(site_indices, sites)
            if cluster.max_length < cutoff[len(cluster.site_indices) - 1]:
                clusters.append(cluster)
        return clusters

    def orbits_constructor(self, clusters):
        """
        return a list of Orbit

        For each orbit, loop over clusters
            for each cluster, check if this cluster exists in this orbit
                if not, attach the cluster to orbit
                else,
        """
        orbit_clusters = []
        grouped_clusters = []
        for i in clusters:
            if i not in orbit_clusters:
                orbit_clusters.append(i)
                grouped_clusters.append([i])
            else:
                grouped_clusters[orbit_clusters.index(i)].append(i)
        orbits = []
        for i in grouped_clusters:
            orbit = Orbit()
            for cl in i:
                orbit.attach_cluster(cl)
            orbits.append(orbit)
        return orbits

    def compute_probability(self, 
                            occ_global,
                            v,
                            temperature,
                            keci,
                            empty_cluster,
                            keci_site,
                            empty_cluster_site):
        """
        Compute the probabilities of each orbit based on the occupancy vector.
        The probabilities are calculated as the average cluster function over all clusters in each orbit.
        """
        logger.debug("Computing probabilities ...")
        occ_sublat = deepcopy(occ_global[self.local_env_indices_list])
        self.calc_corr()
        self.calc_ekra(keci, empty_cluster, keci_site, empty_cluster_site)  # calculate ekra and probability
        probability = self.calc_probability(occ_mobile_ion_specie_1, occ_mobile_ion_specie_2, v, temperature)
        return probability
    
    # @profile
    def initialize_corr(self):
        self.corr = np.empty(shape=len(self.sublattice_indices))
        self.corr_site = np.empty(shape=len(self.sublattice_indices_site))

    # @profile
    def calc_corr(self, corr, occ_sublat, sublattice_indices,
                 corr_site, sublattice_indices_site):
        _calc_corr(corr, occ_sublat, sublattice_indices)
        _calc_corr(corr_site, occ_sublat, sublattice_indices_site)

    # @profile
    def calc_ekra(
        self, keci, empty_cluster, keci_site, empty_cluster_site
    ):  # input is the keci and empty_cluster; ekra = corr*keci + empty_cluster
        ekra = np.inner(self.corr, keci) + empty_cluster
        esite = np.inner(self.corr_site, keci_site) + empty_cluster_site
        return ekra, esite

    # @profile
    def calc_probability(
        self, occ_mobile_ion_specie_1,
        occ_mobile_ion_specie_2, v, temperature
    ) -> float:  # calc_probability() will evaluate migration probability for this event, should be updated everytime when change occupation
        k = 8.617333262145 * 10 ** (-2)  # unit in meV/K
        direction = (occ_mobile_ion_specie_2 - occ_mobile_ion_specie_1) / 2  # 1 if na1 -> na2, -1 if na2 -> na1
        barrier = self.ekra + direction * self.esite / 2  # ekra
        probability = abs(direction) * v * np.exp(-1 * (barrier) / (k * temperature))
        return probability


    def __str__(self):
        return (
            f"\nGLOBAL INFORMATION\n"
            f"Number of orbits = {len(self.orbits)}\n"
            f"Number of clusters = {len(self.clusters)}"
        )

    def __repr__(self):
        return (
            f"name: {self.name},"
            f"LocalClusterExpansion(center_site={self.center_site}, "
            f"MigrationUnit_structure={self.MigrationUnit_structure}, "
            f"clusters={self.clusters}, orbits={self.orbits}, "
            f"sublattice_indices={self.sublattice_indices})"
        )
    
    def write_representative_clusters(self, path="."):
        import os
        logger.info("Writing representative structures to xyz files to %s ...", path)
        if not os.path.exists(path):
            logger.info("Making path: %s", path)
            os.mkdir(path)
        for i, orbit in enumerate(self.orbits):
            orbit.clusters[0].to_xyz(os.path.join(path, f"orbit_{i}.xyz"))

    def as_dict(self):
        
        d = {
                "@module": self.__class__.__module__,
                "@class": self.__class__.__name__,
                "center_site": self.center_site.as_dict(),
                "MigrationUnit_structure": self.MigrationUnit_structure.as_dict(),
                "clusters": [],
                "orbits": [],
                "sublattice_indices": self.sublattice_indices,
            }
     
        for cluster in self.clusters:
            d["clusters"].append(cluster.as_dict())
        for orbit in self.orbits:
            d["orbits"].append(orbit.as_dict())

        return d

    @classmethod
    def from_json(cls, fname):
        logger.info("Loading: %s", fname)
        with open(fname, "rb") as fhandle:
            objDict = json.load(fhandle)
        obj = cls.__new__(cls)
        for key, value in objDict.items():
            setattr(obj, key, value)
        return obj

@nb.njit
def _calc_corr(corr, occ_latt, sublattice_indices):
    i = 0
    for sublat_ind_orbit in sublattice_indices:
        corr[i] = 0
        for sublat_ind_cluster in sublat_ind_orbit:
            corr_cluster = 1
            for occ_site in sublat_ind_cluster:
                corr_cluster *= occ_latt[occ_site]
            corr[i] += corr_cluster
        i += 1


class Orbit:  # orbit is a collection of symmetry equivalent clusters
    def __init__(self):
        self.clusters = []
        self.multiplicity = 0

    def attach_cluster(self, cluster):
        self.clusters.append(cluster)
        self.multiplicity += 1

    def get_cluster_function(
        self, occupancy
    ):  # Phi[orbit] = 1/multiplicity * sum(prod(cluster))
        cluster_function = (1 / self.multiplicity) * sum(
            [c.get_cluster_function(occupancy) for c in self.clusters]
        )
        return cluster_function

    def __str__(self):
        try:
            for i, cluster in enumerate(self.clusters):
                logger.info(
                    "Cluster[%d]: %5s\t%10s\t%8.3f\t%8.3f\t%5s\t%5d",
                    i,
                    cluster.type,
                    str(cluster.site_indices),
                    cluster.max_length,
                    cluster.min_length,
                    cluster.sym,
                    self.multiplicity,
                )
        except TypeError:
            logger.info("No cluster in this orbit!")
    def to_xyz(self, fname):
        self.clusters[0].to_xyz(fname)

    def show_representative_cluster(self):
        logger.info(
            "{0:5s}\t{1:10s}\t{2:8.3f}\t{3:8.3f}\t{4:5s}\t{5:5d}".format(
                self.clusters[0].type,
                str(self.clusters[0].site_indices),
                self.clusters[0].max_length,
                self.clusters[0].min_length,
                self.clusters[0].sym,
                self.multiplicity,
            )
        )

    def as_dict(self):
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "clusters": [],
            "multiplicity": self.multiplicity,
        }
        for cluster in self.clusters:
            d["clusters"].append(cluster.as_dict())
        return d

class Cluster:
    def __init__(self, site_indices, sites):
        cluster_type = {1: "point", 2: "pair", 3: "triplet", 4: "quadruplet"}
        self.site_indices = site_indices
        self.type = cluster_type[len(site_indices)]
        self.structure = Molecule.from_sites(sites)
        self.sym = PointGroupAnalyzer(self.structure).sch_symbol
        if self.type == "point":
            self.max_length = 0
            self.min_length = 0
            self.bond_distances = []
        else:
            self.max_length, self.min_length, self.bond_distances = (
                self.get_bond_distances()
            )

    def __eq__(
        self, other
    ):  # to compare 2 clusters and check if they're the same by comparing atomic distances
        if self.type != other.type:
            return False
        elif self.type == "point" and other.type == "point":
            return self.structure.species == other.structure.species
        elif self.structure.composition != other.structure.composition:
            return
        else:
            return np.linalg.norm(self.bond_distances - other.bond_distances) < 1e-3

    def get_site(self):
        return [self.diff_unit_structure[s] for s in self.index]

    def get_bond_distances(self):
        indices_combination = list(combinations(np.arange(0, len(self.structure)), 2))
        bond_distances = np.array(
            [self.structure.get_distance(*c) for c in indices_combination]
        )
        bond_distances.sort()
        max_length = max(bond_distances)
        min_length = min(bond_distances)
        return max_length, min_length, bond_distances

    def get_cluster_function(self, occupation):
        cluster_function = np.prod([occupation[i] for i in self.site_indices])
        return cluster_function

    def to_xyz(self, fname):
        local_structure_no_oxidation = self.structure.copy()
        local_structure_no_oxidation.remove_oxidation_states()
        local_structure_no_oxidation.to(filename=fname, fmt="xyz")

    def __str__(self):
        logger.info("==============================================================")
        logger.info(
            "This cluster is a %s, constructed by site %s", self.type, self.site_indices
        )
        logger.info(
            "max length = %.3f Angst ,min_length = %.3f Angst",
            self.max_length,
            self.min_length,
        )
        logger.info("Point Group: %s", self.sym)
        try:
            logger.info("Cluster function = %s", self.cluster_function_string)
        except AttributeError:
            pass
        logger.info("==============================================================\n")

    def as_dict(self):
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "site_indices": self.site_indices,
            "type": self.type,
            "structure": self.structure.as_dict(),
            "sym": self.sym,
            "max_length": self.max_length,
            "min_length": self.min_length,
        }
        if type(self.bond_distances) is list:
            d["bond_distances"] = self.bond_distances
        else:
            d["bond_distances"] = self.bond_distances.tolist()
        return d

