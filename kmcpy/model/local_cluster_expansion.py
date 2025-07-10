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
from kmcpy.event.event_generator import find_atom_indices
from pymatgen.core.periodic_table import DummySpecies
from pymatgen.core.sites import PeriodicSite
import logging
import kmcpy
from kmcpy.model.lattice_model import LatticeModel
from copy import deepcopy
import numba as nb
from kmcpy.io import InputSet
from kmcpy.event import Event
from kmcpy.simulation.state import SimulationState

logger = logging.getLogger(__name__) 
logging.getLogger('pymatgen').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

class LocalClusterExpansion(LatticeModel):
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
            2) The cluster_site_indices must correspond exactly to the input structure used in the KMC step, which may require reconstruction of an LCE object using the same KMC-input structure.
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

        self.cluster_site_indices = [
            [cluster.site_indices for cluster in orbit.clusters]
            for orbit in self.orbits
        ]  # cluster_site_indices[orbit,cluster,site]
        
        # Initialize parent LatticeModel - create minimal species_to_site mapping
        # For now, create a simple mapping based on the template structure
        species_to_site = {}
        for site in template_structure:
            species = site.species.elements[0].symbol
            if species not in species_to_site:
                species_to_site[species] = [species, "X"]  # Allow vacancy
        
        # Call parent constructor
        super().__init__(template_structure, species_to_site, basis_type='occupation')
        
        logger.info(
            "Type\tIndex\tmax_length\tmin_length\tPoint Group\tMultiplicity"
        )
        for orbit in self.orbits:
            orbit.show_representative_cluster()

    @classmethod
    def from_inputset(cls, inputset: InputSet)-> "LocalClusterExpansion":
        params = {k: v for k, v in inputset._parameters.items() if k != "task"}
        return cls(**params)
    
    @classmethod
    def from_json(cls, filename: str):
        """
        Load a LocalClusterExpansion object from a JSON file.
        
        Args:
            filename: Path to the JSON file containing the LocalClusterExpansion data
            
        Returns:
            LocalClusterExpansion: The loaded LocalClusterExpansion object
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Create a new instance without calling __init__
        obj = cls.__new__(cls)
        
        # Restore all attributes from the JSON data
        for key, value in data.items():
            if key.startswith('@'):
                # Skip metadata keys
                continue
            elif key == 'orbits':
                # Reconstruct orbits from the stored data
                obj.orbits = []
                for orbit_data in value:
                    orbit = Orbit()
                    orbit.multiplicity = orbit_data.get('multiplicity', 0)
                    
                    # Reconstruct clusters for each orbit
                    orbit.clusters = []
                    for cluster_data in orbit_data.get('clusters', []):
                        # Create cluster from stored data
                        from pymatgen.core.structure import Molecule
                        cluster = Cluster.__new__(Cluster)
                        cluster.site_indices = cluster_data.get('site_indices', [])
                        cluster.type = cluster_data.get('type', 'point')
                        cluster.structure = Molecule.from_dict(cluster_data.get('structure', {}))
                        cluster.sym = cluster_data.get('sym', '')
                        cluster.max_length = cluster_data.get('max_length', 0)
                        cluster.min_length = cluster_data.get('min_length', 0)
                        cluster.bond_distances = cluster_data.get('bond_distances', [])
                        orbit.clusters.append(cluster)
                    
                    obj.orbits.append(orbit)
            elif key == 'center_site':
                # Reconstruct center_site from its dict representation
                from pymatgen.core.sites import PeriodicSite
                obj.center_site = PeriodicSite.from_dict(value)
            elif key == 'MigrationUnit_structure' or key == 'migration_unit_structure':
                # Reconstruct migration unit structure
                if value.get('@class') == 'Molecule':
                    from pymatgen.core.structure import Molecule
                    obj.MigrationUnit_structure = Molecule.from_dict(value)
                else:
                    from pymatgen.core.structure import Structure
                    obj.MigrationUnit_structure = Structure.from_dict(value)
            elif key == 'template_structure':
                # Reconstruct template structure if present
                from pymatgen.core.structure import Structure
                obj.template_structure = Structure.from_dict(value)
            elif key == 'basis':
                # Reconstruct basis if present, default to ChebychevBasis if unknown
                basis_class = value.get('@class', 'ChebychevBasis')
                if basis_class == 'ChebychevBasis':
                    from kmcpy.model.basis import ChebychevBasis
                    obj.basis = ChebychevBasis()
                elif basis_class == 'OccupationBasis':
                    from kmcpy.model.basis import OccupationBasis
                    obj.basis = OccupationBasis()
                else:
                    # Default to ChebychevBasis if class is not recognized
                    from kmcpy.model.basis import ChebychevBasis
                    obj.basis = ChebychevBasis()
            else:
                # For all other attributes, set them directly
                setattr(obj, key, value)
        
        # Convert cluster_site_indices to numba TypedList format if it exists
        if hasattr(obj, 'cluster_site_indices'):
            from numba.typed import List
            converted_list = List([List(List(int(k) for k in j) for j in i) for i in obj.cluster_site_indices])
            obj.cluster_site_indices = converted_list
        
        return obj

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

    def compute(self, simulation_state:SimulationState, event:Event):
        """
        Compute energy value using stored parameters and correlation coefficients.
        
        This method uses the fitted parameters (keci, empty_cluster) stored in the object
        and the predefined cluster_site_indices to compute energy values.
        
        Args:
            simulation_state: SimulationState object containing occupation vector (preferred)
            event: Event object containing mobile ion indices (required for local environment)
            
        Returns:
            float: The computed energy value
        """
        # Check if parameters are stored
        if not hasattr(self, 'keci') or not hasattr(self, 'empty_cluster'):
            raise ValueError("No stored parameters found. Call set_parameters() or load_parameters_from_file() first.")
            
        # Get occupation array - prefer simulation_state over occ_global
        if simulation_state is not None:
            occ_global = simulation_state.occupations
        elif occ_global is None:
            raise ValueError("Either simulation_state or occ_global must be provided")
            
        # Ensure occ_global is a numpy array for proper indexing
        occ_global = np.array(occ_global)
            
        # Require event for local environment determination
        if event is None:
            raise ValueError("Event object is required for LocalClusterExpansion.compute() to determine local environment")
            
        # Initialize correlation array using stored cluster_site_indices
        corr = np.empty(shape=len(self.cluster_site_indices))
        
        # Extract local occupation using event's local environment indices
        occ_sublat = deepcopy(occ_global[event.local_env_indices])
            
        _calc_corr(corr, occ_sublat, self.cluster_site_indices)
        
        # Compute energy using stored parameters
        result = np.inner(corr, self.keci) + self.empty_cluster
        return result

    def compute_probability(self, *args, **kwargs):
        raise NotImplementedError("You cannot compute probability from a single LCE model, try CompositeLCEModel.")
    
    def set_parameters(self, parameters):
        """
        Set fitted parameters for this LocalClusterExpansion model.
        
        Args:
            parameters: LCEModelParameters object or dict containing keci and empty_cluster
        """
        from kmcpy.model.parameters import LCEModelParameters
        
        if isinstance(parameters, LCEModelParameters):
            self.keci = parameters.keci
            self.empty_cluster = parameters.empty_cluster
            self._parameters = parameters
        elif isinstance(parameters, dict):
            self.keci = parameters['keci']
            self.empty_cluster = parameters['empty_cluster']
            self._parameters = parameters
        else:
            raise TypeError("Parameters must be LCEModelParameters object or dict")
        
        logger.info(f"Parameters set for LocalClusterExpansion: keci length={len(self.keci)}, empty_cluster={self.empty_cluster}")

    def load_parameters_from_file(self, filename: str):
        """
        Load fitted parameters from a JSON file.
        
        Args:
            filename: Path to the JSON file containing LCE parameters
        """
        from kmcpy.model.parameters import LCEModelParameters
        
        parameters = LCEModelParameters.from_json(filename)
        self.set_parameters(parameters)



    def write_representative_clusters(self, filename='representative_clusters.txt'):
        """
        Write representative clusters to a text file.
        
        Args:
            filename: Name of the output file
        """
        with open(filename, 'w') as f:
            f.write("Representative Clusters for LocalClusterExpansion\n")
            f.write("=" * 50 + "\n\n")
            f.write("Type\tIndex\tmax_length\tmin_length\tPoint Group\tMultiplicity\n")
            for i, orbit in enumerate(self.orbits):
                cluster = orbit.clusters[0]  # representative cluster
                f.write(f"{cluster.type}\t{i}\t{cluster.max_length:.3f}\t{cluster.min_length:.3f}\t{cluster.sym}\t{orbit.multiplicity}\n")
        logger.info(f"Representative clusters written to {filename}")
    
    def __str__(self):
        """String representation of the LocalClusterExpansion."""
        lines = [
            f"LocalClusterExpansion: {self.name}",
            f"Number of orbits: {len(self.orbits)}",
            f"Local environment sites: {len(self.MigrationUnit_structure)}",
            f"Center site: {self.center_site.species} at {self.center_site.frac_coords}"
        ]
        return "\n".join(lines)
    
    def __repr__(self):
        """Detailed representation of the LocalClusterExpansion."""
        return f"LocalClusterExpansion(orbits={len(self.orbits)}, sites={len(self.MigrationUnit_structure)})"
    
    def as_dict(self):
        """
        Return a dictionary representation of the LocalClusterExpansion.
        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "name": self.name,
            "orbits": [orbit.as_dict() for orbit in self.orbits],
            "cluster_site_indices": self.cluster_site_indices,
            "center_site": self.center_site.as_dict(),
            "migration_unit_structure": self.MigrationUnit_structure.as_dict()
        }

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

@nb.njit
def _calc_corr(corr, occ_latt, cluster_site_indices):
    """
    Calculate correlation function for cluster expansion.
    
    Args:
        corr: Output correlation array
        occ_latt: Occupation array for the lattice
        cluster_site_indices: Nested list structure [orbit][cluster][site]
    """
    i = 0
    for sublat_ind_orbit in cluster_site_indices:
        corr[i] = 0
        for sublat_ind_cluster in sublat_ind_orbit:
            corr_cluster = 1
            for occ_site in sublat_ind_cluster:
                corr_cluster *= occ_latt[occ_site]
            corr[i] += corr_cluster
        i += 1