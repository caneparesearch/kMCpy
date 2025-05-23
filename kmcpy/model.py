"""
Functions to build the model. The LocalClusterExpansion reads a cif (for example, a NaSICON cif) and generate typically, the orbital in KMC. Typical output is lce.json

This is related to the Table S3 in KMC support information pdf.
"""

from itertools import combinations
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from pymatgen.core.structure import Molecule
from kmcpy.external.structure import StructureKMCpy
import numpy as np
import json
import glob
from kmcpy.io import convert
from kmcpy.event_generator import find_atom_indices
from pymatgen.core.periodic_table import DummySpecies
from pymatgen.core.sites import PeriodicSite

class LocalClusterExpansion:
    """
    LocalClusterExpansion will be initialized with a template structure where all the sites are occupied
    cutoff_cluster is the cutoff for pairs and triplet
    cutoff_region is the cutoff for generating local cluster region
    """

    def __init__(self):
        pass

    def initialization(
        self,
        center_frac_coord = [],
        mobile_ion_identifier_type="label",
        mobile_ion_specie_1_identifier="Na1",
        cutoff_cluster=[8, 6, 0],
        cutoff_region=4,
        template_cif_fname="EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",
        is_write_basis=False,
        species_to_be_removed=["Zr4+", "O2-", "O", "Zr"],
        convert_to_primitive_cell=False,
        exclude_site_with_identifier=[],
        **kwargs,
    ):
        """
        Initialization of the LocalClusterExpansion object.

        There are 2 ways to define the local environment (migration unit):
        1) use the center of the mobile ion as the center of the local environment (default, center_frac_coord = [])
        2) use a dummy site as the center of the local environment (set center_frac_coord)

        Args:
            center_frac_coord (list, optional): fractional coordinates of the center of the lcoal environment. Defaults to [].
            mobile_ion_identifier_type="label",mobile_ion_specie_1_identifier="Na1": refers to structure_operation.find_atom_indices
            cutoff_cluster (list, optional): cluster cutoff. Defaults to [6,6,6].
            cutoff_region (float, optional): cutoff for finding migration unit. Defaults to 4.
            template_cif_fname (str, optional): generate cluster from which cif?. Defaults to 'EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif'.
            is_write_basis (bool, optional): .?. Defaults to False.
            species_to_be_removed (list, optional): species to be removed which do not involve in the calculation. Defaults to ['Zr4+','O2-','O','Zr'].

        If the next-step KMC is not based on the same lce object generated in this step, then be careful with 2 things:
        1) the Ekra generated in this step can be transferred to the KMC, within which the orbitals are arranged in the same way as here;
        2) the sublattice_indices are exactly corresponding to the input structure used in the KMC step, which might in need of re-cunstruction of a lce object using the same KMC-input structure.

        """

        template_structure = StructureKMCpy.from_cif(
            template_cif_fname, primitive=convert_to_primitive_cell
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
            print(f"Centering the local environment at {center_frac_coord} ...")
            
            dummy_lattice = template_structure.lattice
            self.center_site = PeriodicSite(species=DummySpecies('X'),
                              coords=center_frac_coord,
                              coords_are_cartesian=False,
                              lattice = dummy_lattice)
        else:
            print(f"Centering the local environment at {mobile_ion_specie_1_indices} ...")
            self.center_site = template_structure[
                mobile_ion_specie_1_indices
            ]  # self.center_site: pymatgen.site

        template_structure.remove_sites([mobile_ion_specie_1_indices])

        print("Searching local env around", self.center_site, "...")

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

        print(len(cluster_indexes), "clusters will be generated ...")

        self.clusters = self.clusters_constructor(
            cluster_indexes, [10] + cutoff_cluster
        )

        self.orbits = self.orbits_constructor(self.clusters)

        self.sublattice_indices = [
            [cluster.site_indices for cluster in orbit.clusters]
            for orbit in self.orbits
        ]  # sublattice_indices[orbit,cluster,site]
        print(
            "Type",
            "Index",
            "max_length",
            "min_length",
            "Point Group",
            "Multiplicity",
            sep="\t",
        )
        for orbit in self.orbits:
            orbit.show_representative_cluster()

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
            print("Local environemnt: ")
            print(local_env_structure)
            local_env_structure.to(fmt="xyz", filename="local_env.xyz")
            print(
                "The point group of local environment is: ",
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
            print(other_cif_name, occupation)
        self.correlation_matrix = correlation_matrix

        print(np.round(correlation_matrix, decimals=3))
        np.savetxt(fname="occupation.txt", X=occupation_matrix, fmt="%5d")
        np.savetxt(fname="correlation_matrix.txt", X=correlation_matrix, fmt="%.8f")
        self.correlation_matrix = correlation_matrix
        # print(other_cif_name,occupation,np.around(correlation,decimals=3),sep='\t')

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
        print("\nGenerating possible clusters within this migration unit...")
        print(
            "Cutoffs: pair =",
            cutoff[1],
            "Angst, triplet =",
            cutoff[2],
            "Angst, quadruplet =",
            cutoff[3],
            "Angst",
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

    def __str__(self):
        print("\nGLOBAL INFORMATION")
        print("Number of orbits =", len(self.orbits))
        print("Number of clusters =", len(self.clusters))

    def write_representative_clusters(self, path="."):
        import os

        print("Writing representative structures to xyz files to", path, "...")
        if not os.path.exists(path):
            print("Making path:", path)
            os.mkdir(path)
        for i, orbit in enumerate(self.orbits):
            orbit.clusters[0].to_xyz(path + "/orbit_" + str(i) + ".xyz")

    def to_json(self, fname):
        """example output as exmaples/lce.json

        Args:
            fname (_type_): _description_
        """
        print("Saving:", fname)
        with open(fname, "w") as fhandle:
            d = self.as_dict()
            jsonStr = json.dumps(
                d, indent=4, default=convert
            )  # to get rid of errors of int64
            fhandle.write(jsonStr)

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
    def from_json(self, fname):
        print("Loading:", fname)
        with open(fname, "rb") as fhandle:
            objDict = json.load(fhandle)
        obj = LocalClusterExpansion()
        obj.__dict__ = objDict
        return obj


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
                print(
                    "Cluster[",
                    i,
                    "]: {0:5s}\t{1:10s}\t{2:8.3f}\t{3:8.3f}\t{4:5s}\t{5:5d}".format(
                        cluster.type,
                        str(cluster.site_indices),
                        cluster.max_length,
                        cluster.min_length,
                        cluster.sym,
                        self.multiplicity,
                    ),
                )
        except TypeError:
            print("No cluster in this orbit!")

    def to_xyz(self, fname):
        self.clusters[0].to_xyz(fname)

    def show_representative_cluster(self):
        print(
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
        print("==============================================================")
        print(
            "This cluster is a", self.type, ", constructed by site", self.site_indices
        )
        print(
            f"max length = {self.max_length:.3f} Angst ,min_length = {self.min_length:.3f} Angst"
        )
        print("Point Group: ", self.sym)
        try:
            print("Cluster function = ", self.cluster_function_string)
        except:
            pass
        print("==============================================================\n")

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
