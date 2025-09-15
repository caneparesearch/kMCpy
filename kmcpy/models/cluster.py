from itertools import combinations
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from pymatgen.core.structure import Molecule
import numpy as np
import logging

logger = logging.getLogger(__name__) 
logging.getLogger('pymatgen').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)


class Orbit:  # orbit is a collection of symmetry equivalent clusters
    """
    Represents an orbit, which is a collection of symmetry equivalent clusters.

    Attributes:
        clusters (list): List of clusters belonging to this orbit.
        multiplicity (int): Number of clusters in the orbit.

    Methods:
        attach_cluster(cluster):
            Adds a cluster to the orbit and increments the multiplicity.

        get_cluster_function(occupancy):
            Calculates the orbit's cluster function as the average of the cluster functions
            of all clusters in the orbit, given an occupancy.

        __str__():
            Returns a string representation of the orbit, listing all clusters and their properties.

        to_xyz(fname):
            Writes the representative cluster's structure to an XYZ file.

        show_representative_cluster():
            Logs information about the representative cluster in the orbit.

        as_dict():
            Serializes the orbit to a dictionary format, including all clusters and multiplicity.
    """
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
    """
    Represents a cluster of atomic sites in a structure, characterized by its type, geometry, and symmetry.

    Args:
        site_indices (list[int]): Indices of the sites forming the cluster.
        sites (list[Site]): List of site objects representing the atomic positions.

    Attributes:
        site_indices (list[int]): Indices of the sites forming the cluster.
        type (str): Type of the cluster ("point", "pair", "triplet", "quadruplet").
        structure (Molecule): Molecule object constructed from the provided sites.
        sym (str): Schoenflies symbol representing the point group symmetry of the cluster.
        max_length (float): Maximum bond distance within the cluster (0 for "point" clusters).
        min_length (float): Minimum bond distance within the cluster (0 for "point" clusters).
        bond_distances (list[float] or np.ndarray): Sorted list/array of bond distances between sites in the cluster.

    Methods:
        __eq__(other): Compares two clusters for equality based on type, composition, and bond distances.
        get_site(): Returns the sites corresponding to the cluster indices.
        get_bond_distances(): Computes and returns the maximum, minimum, and all bond distances within the cluster.
        get_cluster_function(occupation): Calculates the cluster function value for a given occupation vector.
        to_xyz(fname): Exports the cluster structure to an XYZ file.
        __str__(): Returns a string representation of the cluster with detailed information.
        as_dict(): Serializes the cluster to a dictionary format.

    """
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
