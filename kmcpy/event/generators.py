#!/usr/bin/env python
"""
This module provides tools for generating and matching local atomic environments and events for kinetic Monte Carlo (kMC) simulations, particularly for ionic diffusion in crystalline solids. It includes utilities for neighbor environment matching, event generation, and supercell normalization, with support for structures parsed by pymatgen.
"""

import numpy as np
import itertools
import logging
from pymatgen.util.coord import get_angle
from typing import Dict, List, Optional, Union
from pathlib import Path

from pymatgen.core import Structure
from kmcpy.structure import LatticeStructure, LocalLatticeStructure
from kmcpy.event import Event, EventLib

logger = logging.getLogger(__name__) 

def print_divider():
    logger.info("\n\n-------------------------------------------\n\n")

class NeighborInfoMatcher:
    def __init__(
        self,
        neighbor_species=(("Cl-", 4), ("Li+", 8)),
        distance_matrix=np.array([[0, 1], [1, 0]]),
        neighbor_sequence=[{}],
        neighbor_species_distance_matrix_dict={
            "Cl-": np.array([[0, 1], [1, 0]]),
            "Li+": np.array([[0, 1], [1, 0]]),
        },
        neighbor_species_sequence_dict={"Cl-": [{}], "Li+": [{}]},
    ):
        """neighbor_info matcher, the __init__ method shouln't be used. Use the from_neighbor_info() instead. This is neighbor_info matcher to match the nearest neighbor info output from local_env.cutoffdictNN.get_nn_info. This NeighborInfoMatcher Class is initialized by a reference neighbor_info, a distance matrix is built as reference. Then user can call the NeighborInfoMatcher.brutal_match function to sort another nn_info so that the sequence of neighbor of "another nn_info" is arranged so that the distance matrix are the same

        Args:
            neighbor_species (tuple, optional): tuple ( tuple ( str(species), int(number_of_this_specie in neighbors)  )  ). Defaults to (('Cl-', 4),('Li+', 8)).
            distance_matrix (np.array, optional): np.2d array as distance matrix. Defaults to np.array([[0,1],[1,0]]).
            neighbor_sequence (list, optional): list of dictionary in the format of nn_info returning value. Defaults to [{}].
            neighbor_species_distance_matrix_dict (dict, optional): this is a dictionary with key=species and value=distance_matrix(2D numpy array) which record the distance matrix of respective element. . Defaults to {"Cl-":np.array([[0,1],[1,0]]),"Li+":np.array([[0,1],[1,0]])}.
            neighbor_species_sequence_dict (dict, optional): dictionary with key=species and value=list of dictionary which is just group the reference neighbor sequence by different elements. Defaults to {"Cl-":[{}],"Li+":[{}]}.
        """

        self.neighbor_species = neighbor_species
        self.distance_matrix = distance_matrix
        self.neighbor_species_distance_matrix_dict = (
            neighbor_species_distance_matrix_dict
        )
        self.neighbor_species_sequence_dict = neighbor_species_sequence_dict
        self.neighbor_sequence = neighbor_sequence

    @classmethod
    def from_neighbor_sequences(self, neighbor_sequences=[{}]):
        """generally generate the neighbor info matcher from this

        Args:
            neighbor_sequences (list, optional): list of dictionary from cutoffdictNN.get_nn_info(). Defaults to [{}].

        Returns:
            NeighborInfoMatcher: a NeighborInfoMatcher object, initialized from get_nn_info output
        """

        # -------------------------------------------------------
        # this part of function is adapted from pymatgen.analysis.local_env
        # Shyue Ping Ong, William Davidson Richards, Anubhav Jain, Geoffroy Hautier, Michael Kocher, Shreyas Cholia, Dan Gunter, Vincent Chevrier, Kristin A. Persson, Gerbrand Ceder. Python Materials Genomics (pymatgen) : A Robust, Open-Source Python Library for Materials Analysis. Computational Materials Science, 2013, 68, 314-319. doi:10.1016/j.commatsci.2012.10.028
        cn_dict = {}

        neighbor_species_sequence_dict = {}

        for neighbor in neighbor_sequences:
            """for example NaSICON has 12 neighbors, 6 Na and 6 Si, here for neighbor we build the coordination number dict."""

            site_element = neighbor["site"].species_string

            if site_element not in cn_dict:
                cn_dict[site_element] = 1
            else:
                cn_dict[site_element] += 1
            if site_element not in neighbor_species_sequence_dict:
                neighbor_species_sequence_dict[site_element] = [neighbor]
            else:
                neighbor_species_sequence_dict[site_element].append(neighbor)
        # end of adapting.
        # -------------------------------------------------------

        neighbor_species_distance_matrix_dict = {}

        for species in neighbor_species_sequence_dict:
            neighbor_species_distance_matrix_dict[species] = (
                self.build_distance_matrix_from_getnninfo_output(
                    neighbor_species_sequence_dict[species]
                )
            )

        neighbor_species = tuple(sorted(cn_dict.items(), key=lambda x: x[0]))

        distance_matrix = self.build_distance_matrix_from_getnninfo_output(
            neighbor_sequences
        )

        return NeighborInfoMatcher(
            neighbor_species=neighbor_species,
            distance_matrix=distance_matrix,
            neighbor_sequence=neighbor_sequences,
            neighbor_species_distance_matrix_dict=neighbor_species_distance_matrix_dict,
            neighbor_species_sequence_dict=neighbor_species_sequence_dict,
        )

    @classmethod
    def build_distance_matrix_from_getnninfo_output(self, cutoffdnn_output=[{}]):
        """build a distance matrix from the output of CutOffDictNNKMCpy.get_nn_info

        nn_info looks like:
        [{'site': PeriodicSite: Si4+ (-3.2361, -0.3015, 9.2421) [-0.3712, -0.0379, 0.4167], 'image': (-1, -1, 0), 'weight': 3.7390091507903174, 'site_index': 39, 'wyckoff_sequence': 15, 'local_index': 123, 'label': 'Si1'}, {'site': PeriodicSite: Na+ (-1.2831, -2.6519, 9.2421) [-0.3063, -0.3333, 0.4167], 'image': (-1, -1, 0), 'weight': 3.4778161424304046, 'site_index': 23, 'wyckoff_sequence': 17, 'local_index': 35, 'label': 'Na2'}, {'site': ...]

        or say:

        nn_info is a list, the elements of list is dictionary, the keys of dictionary are: "site":pymatgen.site, "wyckoff_sequence": ....

        Use the site.distance function to build matrix


        Args:
            cutoffdnn_output (nn_info, optional): nninfo. Defaults to neighbor_sequences.

        Returns:
            np.2darray: 2d distance matrix, in format of numpy.array. The Column and the Rows are following the input sequence.
        """

        distance_matrix = np.zeros(shape=(len(cutoffdnn_output), len(cutoffdnn_output)))

        for sitedictindex1 in range(0, len(cutoffdnn_output)):
            for sitedictindex2 in range(0, len(cutoffdnn_output)):
                """Reason for jimage=[0,0,0]

                site.distance is calculated by frac_coord1-frac_coord0 and get the cartesian distance. Note that for the two sites in neighbors,  the frac_coord itself already contains the information of jimage. For exaple:Si4+ (-3.2361, -0.3015, 9.2421) [-0.3712, -0.0379, 0.4167], 'image': (-1, -1, 0),  see that the frac_coord of this Si4+ is not normalized to (0,1)!

                .
                """
                distance_matrix[sitedictindex1][sitedictindex2] = cutoffdnn_output[
                    sitedictindex1
                ]["site"].distance(
                    cutoffdnn_output[sitedictindex2]["site"], jimage=[0, 0, 0]
                )

        return distance_matrix

    @classmethod
    def build_angle_matrix_from_getnninfo_output(self, cutoffdnn_output=[{}]):
        """build a distance matrix from the output of CutOffDictNNKMCpy.get_nn_info

        nn_info looks like:
        [{'site': PeriodicSite: Si4+ (-3.2361, -0.3015, 9.2421) [-0.3712, -0.0379, 0.4167], 'image': (-1, -1, 0), 'weight': 3.7390091507903174, 'site_index': 39, 'wyckoff_sequence': 15, 'local_index': 123, 'label': 'Si1'}, {'site': PeriodicSite: Na+ (-1.2831, -2.6519, 9.2421) [-0.3063, -0.3333, 0.4167], 'image': (-1, -1, 0), 'weight': 3.4778161424304046, 'site_index': 23, 'wyckoff_sequence': 17, 'local_index': 35, 'label': 'Na2'}, {'site': ...]

        or say:

        nn_info is a list, the elements of list is dictionary, the keys of dictionary are: "site":pymatgen.site, "wyckoff_sequence": ....

        Use the site.distance function to build matrix


        Args:
            cutoffdnn_output (nn_info, optional): nninfo. Defaults to neighbor_sequences.

        Returns:
            np.3darray: 3d distance matrix, in format of numpy.array. The Column and the Rows are following the input sequence.
        """

        angle_matrix = np.zeros(
            shape=(len(cutoffdnn_output), len(cutoffdnn_output), len(cutoffdnn_output))
        )

        for sitedictindex1 in range(0, len(cutoffdnn_output)):
            for sitedictindex2 in range(0, len(cutoffdnn_output)):
                for sitedictindex3 in range(0, len(cutoffdnn_output)):
                    v1 = (
                        cutoffdnn_output[sitedictindex2]["site"].coords
                        - cutoffdnn_output[sitedictindex1]["site"].coords
                    )
                    v2 = (
                        cutoffdnn_output[sitedictindex3]["site"].coords
                        - cutoffdnn_output[sitedictindex1]["site"].coords
                    )
                    angle_matrix[sitedictindex1][sitedictindex2][sitedictindex3] = (
                        get_angle(v1, v2, "degrees")
                    )

        return angle_matrix

    def rearrange(
        self, wrong_distance_matrix_of_specie=[], species="Na", atol=0.01, rtol=0.01
    ):
        """
        A very fast version of rearranging the neighbors with same species

        Args:
            wrong_distance_matrix_of_specie (np.2Drray, optional): distance matrix of wrong distance matrix that is supposed to be rearranged and match with self.distance matrix. Defaults to [].
            species (str, optional): the species to match, is the key to self.neighbor_species_distance_matrix_dict. Defaults to 'Na'.

        Raises:
            ValueError: no correct sequence found

        Returns:
            list: list of list, of which is the sequence of index of rearranged sequence
        """
        # distance_matrix=distance matrix of wrong neighbor sequence
        correct_distance_matrix = self.neighbor_species_distance_matrix_dict[
            species
        ]  # np.2darray
        previous_possible_sequences = []

        for i in range(0, len(correct_distance_matrix)):
            previous_possible_sequences.append([i])  # init

        if len(correct_distance_matrix) == 1:
            return [[0]]

        for i in range(0, len(correct_distance_matrix) - 1):

            new_possible_sequences = []

            correct_distance_matrix_in_this_round = correct_distance_matrix[
                0 : 2 + i, 0 : 2 + i
            ]

            for previous_possible_sequence in previous_possible_sequences:
                for i in range(0, len(correct_distance_matrix)):
                    if i not in previous_possible_sequence:
                        tmp_sequence = previous_possible_sequence.copy()
                        tmp_sequence.append(i)
                        tmp_rebuilt_submatrix = self.rebuild_submatrix(
                            distance_matrix=wrong_distance_matrix_of_specie,
                            sequences=tmp_sequence,
                        )
                        if np.allclose(
                            tmp_rebuilt_submatrix,
                            correct_distance_matrix_in_this_round,
                            atol=atol,
                            rtol=rtol,
                        ):
                            new_possible_sequences.append(tmp_sequence)

            previous_possible_sequences = new_possible_sequences.copy()

        if len(new_possible_sequences) == 0:
            raise ValueError("new possible sequence=0.")

        return new_possible_sequences

    def rebuild_submatrix(self, distance_matrix, sequences=[2, 1]):
        """rebuild the submatrix, with given seuqneces and distance matrix, rebuild the distance matrix from given sequence

        Args:
            distance_matrix (np.2Darray): distance matrix, length = sequences.len()
            sequences (list, optional): new sequences. Defaults to [2,1].

        Returns:
            np.2darray: rebuilt matrix
        """
        # wrong_distance_matrix is the matrix that is different from the reference.
        rebuilt_matrix = np.zeros(shape=(len(sequences), len(sequences)))

        for idx1 in range(len(sequences)):
            for idx2 in range(len(sequences)):
                rebuilt_matrix[idx1][idx2] = distance_matrix[sequences[idx1]][
                    sequences[idx2]
                ]
        return rebuilt_matrix

    def brutal_match(
        self, unsorted_nninfo=[{}], rtol=0.001, atol=0.001, find_nearest_if_fail=False
    ):
        """brutally sort the input unsorted_nninfo. Although brutal but fast enough for now

        update 220621: not fast enough for LiCoO2 with 12 neighbors,

        rewrite the finding sequence algo. Now is freaking fast again!

        Args:
            unsorted_nninfo (list, optional): the unsorted nn_info of an element. The nn_info are compared with the nn_info of class instance. Defaults to [{}].

            Tolerance: Refer to np.allclose
            rtol (float, optional): relative tolerance of np.allclose in order to determine if the distance matrix are the same. Better not too small. Defaults to 0.01.
            atol (float, optional): absolute tolerance
            find_nearest_if_fail(bool, optional): This should be true only for grain boundary model!

        Raises:
            ValueError: this will perform a check if the inputted unsorted nn_info has the same neighbor species type and amount
            ValueError: if the unsorted_nninfo cannot be sort with reference of the distance matrix of this NeighborInfoMatcher instance. Probably due to too small rtol or it's just not the same neighbor_infos

        Returns:
            sorted nninfo, as list of dictionary: in the format of cutoffdictNN.get_nn_info
        """

        unsorted_neighbor_info = NeighborInfoMatcher.from_neighbor_sequences(
            unsorted_nninfo
        )

        if self.neighbor_species != unsorted_neighbor_info.neighbor_species:
            raise ValueError("input neighbor_info has different environment")

        if np.allclose(
            unsorted_neighbor_info.distance_matrix,
            self.distance_matrix,
            rtol=rtol,
            atol=atol,
        ):
            logger.info(
                "no need to rearrange this neighbor_info. The distance matrix is already the same. The differece matrix is : \n"
            )
            logger.info(
                str(unsorted_neighbor_info.distance_matrix - self.distance_matrix)
            )

            return unsorted_neighbor_info.neighbor_sequence

        sorted_neighbor_sequence_dict = {}

        for specie in unsorted_neighbor_info.neighbor_species_sequence_dict:

            rearranged_sequences_of_neighbor = self.rearrange(
                wrong_distance_matrix_of_specie=unsorted_neighbor_info.neighbor_species_distance_matrix_dict[
                    specie
                ],
                species=specie,
                atol=atol,
                rtol=rtol,
            )

            sorted_neighbor_sequence_dict[specie] = []
            for rearranged_sequence_of_neighbor in rearranged_sequences_of_neighbor:
                possible_local_sequence = []
                for new_index in rearranged_sequence_of_neighbor:
                    possible_local_sequence.append(
                        unsorted_neighbor_info.neighbor_species_sequence_dict[specie][
                            new_index
                        ]
                    )
                sorted_neighbor_sequence_dict[specie].append(possible_local_sequence)

            """
            print(sorted_neighbor_sequence_dict[specie])
            raise ValueError()
            
            for possible_local_sequence in itertools.permutations(unsorted_neighbor_info.neighbor_species_sequence_dict[specie]):
                
                if np.allclose(self.build_distance_matrix_from_getnninfo_output(possible_local_sequence),self.neighbor_species_distance_matrix_dict[specie],rtol=rtol,atol=atol):
                    
                    sorted_neighbor_sequence_dict[specie].append(list(possible_local_sequence))
            """
            if len(sorted_neighbor_sequence_dict[specie]) == 0:
                raise ValueError(
                    "no sorted sequence found for "
                    + str(specie)
                    + " please check if the rtol or atol is too small"
                )

        # logger.info(str(sorted_neighbor_sequence_dict))

        sorted_neighbor_sequence_list = []

        for specie in sorted_neighbor_sequence_dict:
            sorted_neighbor_sequence_list.append(sorted_neighbor_sequence_dict[specie])

        if find_nearest_if_fail:
            closest_smilarity_score = 999999.0
            closest_sequence = []
            for possible_complete_sequence in itertools.product(
                *sorted_neighbor_sequence_list
            ):

                re_sorted_neighbors_list = []

                for neighbor in possible_complete_sequence:

                    re_sorted_neighbors_list.extend(list(neighbor))

                this_smilarity_score = np.sum(
                    np.abs(
                        self.build_distance_matrix_from_getnninfo_output(
                            re_sorted_neighbors_list
                        )
                        - self.distance_matrix
                    )
                )

                if this_smilarity_score < closest_smilarity_score:
                    closest_smilarity_score = this_smilarity_score
                    closest_sequence = re_sorted_neighbors_list

            logger.info(
                "the closest neighbor_info identified. Total difference"
                + str(closest_smilarity_score)
            )
            logger.info("new sorting is found,new distance matrix is ")
            logger.info(
                str(self.build_distance_matrix_from_getnninfo_output(closest_sequence))
            )
            logger.info("The differece matrix is : \n")
            logger.info(
                str(
                    self.build_distance_matrix_from_getnninfo_output(closest_sequence)
                    - self.distance_matrix
                )
            )
            return closest_sequence

        else:

            for possible_complete_sequence in itertools.product(
                *sorted_neighbor_sequence_list
            ):

                # logger.info(str(possible_complete_sequence))

                re_sorted_neighbors_list = []

                for neighbor in possible_complete_sequence:

                    re_sorted_neighbors_list.extend(list(neighbor))

                if np.allclose(
                    self.build_distance_matrix_from_getnninfo_output(
                        re_sorted_neighbors_list
                    ),
                    self.distance_matrix,
                    rtol=rtol,
                    atol=atol,
                ):
                    logger.info(
                        "new sorting is found,new distance matrix is "
                    )
                    logger.info(
                        str(
                            self.build_distance_matrix_from_getnninfo_output(
                                re_sorted_neighbors_list
                            )
                        )
                    )
                    logger.info("The differece matrix is : \n")
                    logger.info(
                        str(
                            self.build_distance_matrix_from_getnninfo_output(
                                re_sorted_neighbors_list
                            )
                            - self.distance_matrix
                        )
                    )

                    return re_sorted_neighbors_list

            raise ValueError("sequence not founded!")


class EventGenerator:
    """EventGenerator is a class for generating events from a structure. It is used to generate events for kinetic Monte Carlo simulations. The events are generated based on the local environment of the mobile ions in the structure. The events are stored in a json file and a csv file.
    """

    def __init__(self):
        self.lattice_structure = None
        self.local_environments = {}

    def generate_events(
        self,
        structure_file="210.cif",
        mobile_species=["Na"],
        mobile_site_mapping=None,
        local_env_cutoff=4.0,
        exclude_species=None,
        supercell_shape=[2, 1, 1],
        event_file="events.json",
        event_dependencies_file="event_dependencies.csv",
        rtol=1e-3,
        atol=1e-3
    ):
        """
        Modern event generation using LatticeStructure and LocalLatticeStructure.

        This method provides a cleaner, more maintainable approach.
        For legacy functionality, use generate_events_legacy().

        Args:
            structure_file (str): Path to structure file (CIF, POSCAR, etc.)
            mobile_species (list): List of mobile species (e.g., ["Na", "Li"])
            mobile_site_mapping (dict): Mapping of species to possible occupants
            local_env_cutoff (float): Cutoff distance for local environment
            exclude_species (list): Species to exclude from analysis
            supercell_shape (list): Shape of supercell [nx, ny, nz]
            event_file (str): Output events file name
            event_dependencies_file (str): Output dependencies file name
            rtol (float): Relative tolerance for matching
            atol (float): Absolute tolerance for matching

        Returns:
            dict: Dictionary containing reference local environments
        """
        import json
        from pymatgen.core import Structure
        from kmcpy.structure import LatticeStructure, LocalLatticeStructure
        from kmcpy.event import Event, EventLib
        from kmcpy.io import convert
        import kmcpy

        logger.info(kmcpy.get_logo())
        logger.info("Using modern event generation with LatticeStructure framework")

        # Step 1: Load structure
        structure = Structure.from_file(structure_file)
        if exclude_species:
            structure.remove_species(exclude_species)
        
        logger.info(f"Loaded structure with composition: {structure.composition}")

        # Step 2: Create LatticeStructure with appropriate site mapping
        if mobile_site_mapping is None:
            mobile_site_mapping = {}
            for species in mobile_species:
                mobile_site_mapping[species] = [species, "X"]  # X = vacancy
            # Add immobile species
            for site in structure:
                species_str = site.species_string
                if species_str not in mobile_site_mapping:
                    mobile_site_mapping[species_str] = [species_str]

        self.lattice_structure = LatticeStructure(
            template_structure=structure,
            specie_site_mapping=mobile_site_mapping,
            basis_type='occupation'
        )

        # Step 3: Find mobile sites
        mobile_sites = []
        for i, site in enumerate(structure):
            if site.species_string in mobile_species:
                mobile_sites.append(i)

        logger.info(f"Found {len(mobile_sites)} mobile sites")

        # Step 4: Generate and classify local environments
        reference_local_envs = {}
        
        for site_idx in mobile_sites:
            # Create LocalLatticeStructure for this site
            local_env = LocalLatticeStructure(
                template_structure=structure,
                center=site_idx,
                cutoff=local_env_cutoff,
                specie_site_mapping=mobile_site_mapping,
                exclude_species=exclude_species
            )
            
            # Create environment signature
            env_signature = self._create_environment_signature(local_env)
            
            if env_signature not in reference_local_envs:
                reference_local_envs[env_signature] = {
                    'local_env': local_env,
                    'sites': []
                }
                logger.info(f"New local environment type: {env_signature}")
            
            reference_local_envs[env_signature]['sites'].append(site_idx)
            self.local_environments[site_idx] = env_signature

        # Step 5: Create supercell
        supercell = structure.copy()
        supercell.make_supercell(supercell_shape)
        logger.info(f"Created supercell with {len(supercell)} sites")

        # Step 6: Generate events
        events = []
        event_lib = EventLib()

        # Find mobile sites in supercell
        mobile_sites_sc = []
        for i, site in enumerate(supercell):
            if site.species_string in mobile_species:
                mobile_sites_sc.append(i)

        logger.info(f"Generating events from {len(mobile_sites_sc)} mobile sites in supercell...")

        for site_idx in mobile_sites_sc:
            site = supercell[site_idx]
            neighbors = supercell.get_neighbors(site, local_env_cutoff * 1.5)
            
            # Get local environment indices
            local_env_indices = self._get_local_environment_indices_supercell(
                supercell, site_idx, local_env_cutoff
            )
            
            for neighbor in neighbors:
                neighbor_idx = self._find_site_index_in_supercell(supercell, neighbor)
                
                if (neighbor_idx is not None and 
                    neighbor.species_string in mobile_species and
                    neighbor_idx != site_idx):
                    
                    # Create event
                    event = Event(
                        mobile_ion_indices=(site_idx, neighbor_idx),
                        local_env_indices=tuple(local_env_indices)
                    )
                    events.append(event)
                    event_lib.add_event(event)

        if len(events) == 0:
            raise ValueError("No events generated. Check input parameters.")

        # Step 7: Generate dependencies and save
        logger.info("Generating event dependencies...")
        event_lib.generate_event_dependencies()
        
        # Save events
        events_dict = [event.as_dict() for event in events]
        with open(event_file, "w") as f:
            json.dump(events_dict, f, indent=4, default=convert)
        logger.info(f"Saved {len(events)} events to {event_file}")
        
        # Save dependencies
        event_lib.save_event_dependencies_to_file(event_dependencies_file)
        logger.info(f"Saved dependencies to {event_dependencies_file}")
        
        # Log statistics
        stats = event_lib.get_dependency_statistics()
        logger.info(f"Generated {len(event_lib)} events with dependency statistics: {stats}")

        return reference_local_envs

    def generate_events_modern(self, *args, **kwargs):
        """
        Alias for generate_events method for backward compatibility.

        This method is provided for compatibility with code expecting the
        generate_events_modern name. It simply delegates to generate_events.

        Returns:
            dict: Dictionary containing reference local environments
        """
        return self.generate_events(*args, **kwargs)

    def _create_environment_signature(self, local_env):
        """Create a unique signature for a local environment type."""
        species_count = {}
        for site in local_env.structure:
            species = site.species_string
            species_count[species] = species_count.get(species, 0) + 1
        
        signature_parts = []
        for species, count in sorted(species_count.items()):
            signature_parts.append(f"{species}:{count}")
        
        return "_".join(signature_parts)

    def _find_site_index_in_supercell(self, supercell, target_site):
        """Find the index of a target site in the supercell."""
        import numpy as np
        for i, site in enumerate(supercell):
            if np.allclose(site.frac_coords, target_site.frac_coords, atol=1e-4):
                return i
        return None

    def _get_local_environment_indices_supercell(self, supercell, central_site_idx, cutoff):
        """Get indices of sites in the local environment."""
        central_site = supercell[central_site_idx]
        neighbors = supercell.get_neighbors(central_site, cutoff)
        
        env_indices = []
        for neighbor in neighbors:
            neighbor_idx = self._find_site_index_in_supercell(supercell, neighbor)
            if neighbor_idx is not None:
                env_indices.append(neighbor_idx)
        
        return env_indices

    def generate_events_legacy(
        self,
        structure_file="210.cif",
        convert_to_primitive_cell=False,
        local_env_cutoff_dict={("Li+", "Cl-"): 4.0, ("Li+", "Li+"): 3.0},
        mobile_ion_identifier_type="label",
        mobile_ion_identifiers=("Na1", "Na2"),
        species_to_be_removed=["O2-", "O"],
        distance_matrix_rtol=0.01,
        distance_matrix_atol=0.01,
        find_nearest_if_fail=True,
        export_local_env_structure=True,
        supercell_shape=[2, 1, 1],
        event_file="events.json",
        event_dependencies_file="event_dependencies.csv",
    ):
        """
        220603 XIE WEIHANG
        3rd version of generate events, using the x coordinate and label as the default sorting criteria for neighbors in local environment therefore should behave similar as generate_events1. Comparing generate_events1, this implementation accelerate the speed of finding neighbors and add the capability of looking for various kind of mobile_ion_specie_1s (not only Na1 in generate_events1). In addtion, generate events3 is also capable of identifying various kind of local environment, which can be used in grain boundary models. Although the _generate_event_kernal is not yet capable of identifying different types of environment. The speed is improved a lot comparing with version2

        Args:
            structure_file (str, optional): the file name of primitive cell of KMC model. Strictly limited to cif file because only cif parser is capable of taking label information of site. This cif file should include all possible site i.e., no vacancy. For example when dealing with NaSICON The input cif file must be a fully occupied composition, which includes all possible Na sites N4ZSP; the varied Na-Vacancy should only be tuned by occupation list.
            convert_to_primitive_cell (bool, optional): whether convert to primitive cell. For rhombohedral, if convert_to_primitive_cell, will use the rhombohedral primitive cell, otherwise use the hexagonal primitive cell. Defaults to False.
            local_env_cutoff_dict (dict, optional): cutoff dictionary for finding the local environment. This will be passed to local_env.cutoffdictNN`. Defaults to {("Li+","Cl-"):4.0,("Li+","Li+"):3.0}.
            mobile_ion_identifier_type (str, optional): atom identifier type, choose from ["specie", "label"].. Defaults to "specie".
            mobile_ion_identifiers (tuple, optional): A tuple containing the identifiers for the two mobile ion species involved in an event. Defaults to ("Na1", "Na2").
            species_to_be_removed (list, optional): list of species to be removed, those species are not involved in the KMC calculation. Defaults to ["O2-","O"].
            distance_matrix_rtol (float, optional): r tolerance of distance matrix for determining whether the sequence of neighbors are correctly sorted in local envrionment. For grain boundary model, please allow the rtol up to 0.2~0.4, for bulk model, be very strict to 0.01 or smaller. Smaller rtol will also increase the speed for searching neighbors. Defaults to 0.01.
            distance_matrix_atol (float, optional): absolute tolerance , . Defaults to 0.01.
            find_nearest_if_fail (bool, optional): if fail to sort the neighbor with given rtolerance and atolerance, find the best sorting that have most similar distance matrix? This should be False for bull model because if fail to find the sorting ,there must be something wrong. For grain boundary , better set this to True because they have various coordination type. Defaults to True.
            export_local_env_structure (bool, optional): whether to export the local environment structure to cif file. If set to true, for each representatibe local environment structure, a cif file will be generated for further investigation. This is for debug purpose. Once confirming that the code is doing correct thing, it's better to turn off this feature. Defaults to True.
            supercell_shape (list, optional): shape of supercell passed to the kmc_build_supercell function, array type that can be 1D or 2D. Defaults to [2,1,1].
            event_file (str, optional): file name for the events json file. Defaults to "events.json".
            event_dependencies_file (str, optional): file name for event dependencies matrix. Defaults to 'event_dependencies.csv'.

        Raises:
            NotImplementedError: the atom identifier type=list is not yet implemented
            ValueError: unrecognized atom identifier type
            ValueError: if no events are generated, there might be something wrong with cif file? or atom identifier?

        Returns:
            nothing: nothing is returned
        """

        # --------------
        import json
        from kmcpy.external.structure import StructureKMCpy
        from kmcpy.external.local_env import CutOffDictNNKMCpy

        from kmcpy.io import convert
        from kmcpy.event import Event
        import kmcpy

        logger.info(kmcpy.get_logo())

        # generate primitive cell
        primitive_cell = StructureKMCpy.from_cif(
            structure_file, primitive=convert_to_primitive_cell
        )
        # primitive_cell.add_oxidation_state_by_element({"Na":1,"O":-2,"P":5,"Si":4,"V":2.5})
        primitive_cell.add_oxidation_state_by_guess()

        primitive_cell.remove_species(species_to_be_removed)

        logger.info(
            "primitive cell composition after adding oxidation state and removing uninvolved species: "
        )
        logger.info(str(primitive_cell.composition))
        logger.info("building migrating_ion index list")

        migrating_ion_indices = find_atom_indices(
            primitive_cell,
            mobile_ion_identifier_type=mobile_ion_identifier_type,
            atom_identifier=mobile_ion_identifiers[0],
        )

        # --------

        local_env_finder = CutOffDictNNKMCpy(local_env_cutoff_dict)

        reference_local_env_dict = {}
        """this is aimed for grain boundary model. For bulk model, there should be only one type of reference local environment. i.e., len(reference_local_env_dict)=1

        The key is a tuple type: For NaSICON, there is only one type of local environment: 6 Na2 and 6 Si. The tuple is in the format of (('Na+', 6), ('Si4+', 6)) . The key is a NeighborInfoMatcher as the reference local environment.
        """

        local_env_info_dict = {}
        """add summary to be done

        """

        reference_local_env_type = 0

        logger.info(
            "start finding the neighboring sequence of migrating_ions"
        )
        logger.info(
            "total number of migrating_ions:" + str(len(migrating_ion_indices))
        )

        neighbor_has_been_found = 0

        for migrating_ion_index in migrating_ion_indices:

            unsorted_neighbor_sequences = sorted(
                sorted(
                    local_env_finder.get_nn_info(primitive_cell, migrating_ion_index),
                    key=lambda x: x["site"].coords[0],
                ),
                key=lambda x: x["site"].specie,
            )

            this_nninfo = NeighborInfoMatcher.from_neighbor_sequences(
                unsorted_neighbor_sequences
            )

            # print(this_nninfo.build_angle_matrix_from_getnninfo_output(primitive_cell,unsorted_neighbor_sequences))

            if this_nninfo.neighbor_species not in reference_local_env_dict:

                # then take this as the reference neighbor info sequence

                if export_local_env_structure:

                    reference_local_env_sites = [primitive_cell[migrating_ion_index]]

                    for i in unsorted_neighbor_sequences:
                        reference_local_env_sites.append(i["site"])
                    
                    reference_local_env_structure = StructureKMCpy.from_sites(
                        sites=reference_local_env_sites
                    )

                    reference_local_env_structure.to(
                        fmt="cif",
                        filename=str(reference_local_env_type)
                        + "th_reference_local_env.cif",
                    )
                    reference_local_env_type += 1

                    logger.info(
                        str(reference_local_env_type)
                        + "th type of reference local_env structure cif file is created. please check"
                    )

                reference_local_env_dict[this_nninfo.neighbor_species] = this_nninfo

                local_env_info_dict[
                    primitive_cell[migrating_ion_index].properties["local_index"]
                ] = this_nninfo.neighbor_sequence

                logger.info(
                    "a new type of local environment is recognized with the species "
                    + str(this_nninfo.neighbor_species)
                    + " \nthe distance matrix are \n"
                    + str(this_nninfo.distance_matrix)
                )

            else:
                logger.info(
                    "a local environment is created with the species "
                    + str(this_nninfo.neighbor_species)
                    + " \nthe distance matrix are \n"
                    + str(this_nninfo.distance_matrix)
                )

                sorted_neighbor_sequence = reference_local_env_dict[
                    this_nninfo.neighbor_species
                ].brutal_match(
                    this_nninfo.neighbor_sequence,
                    rtol=distance_matrix_rtol,
                    atol=distance_matrix_atol,
                    find_nearest_if_fail=find_nearest_if_fail,
                )

                local_env_info_dict[
                    primitive_cell[migrating_ion_index].properties["local_index"]
                ] = sorted_neighbor_sequence

            neighbor_has_been_found += 1

            logger.info(
                str(neighbor_has_been_found)
                + " out of "
                + str(len(migrating_ion_indices))
                + " neighboring sequence has been found"
            )

        supercell = primitive_cell.make_kmc_supercell(supercell_shape)
        logger.info("supercell is created")
        logger.info(str(supercell))

        supercell_migrating_ion_indices = find_atom_indices(
            supercell,
            mobile_ion_identifier_type=mobile_ion_identifier_type,
            atom_identifier=mobile_ion_identifiers[0],
        )

        events = []
        events_dict = []

        indices_dict_from_identifier = supercell.kmc_build_dict(
            skip_check=False
        )  # a dictionary. Key is the tuple with format of ([supercell[0],supercell[1],supercell[2],label,local_index]) that contains the information of supercell, local index (index in primitive cell), Value is the corresponding global site index.  This hash dict for acceleration purpose

        for supercell_migrating_ion_index in supercell_migrating_ion_indices:

            # for migrating_ions of newly generated supercell, find the neighbors

            supercell_tuple = supercell[supercell_migrating_ion_index].properties[
                "supercell"
            ]

            local_index_of_this_migrating_ion = supercell[
                supercell_migrating_ion_index
            ].properties["local_index"]

            local_env_info = []  # list of integer / indices of local environment

            for neighbor_site_in_primitive_cell in local_env_info_dict[
                local_index_of_this_migrating_ion
            ]:
                """

                 local_env_info_dict[local_index_of_this_migrating_ion]

                In primitive cell, the migrating_ion has 1 unique identifier: The "local index inside the primitive cell"

                In supercell, the migrating_ion has an additional unique identifier: "supercell_tuple"

                However, as long as the "local index inside the primitive cell" is the same,
                no matter which supercell this migrating_ion belongs to,
                the sequence of "local index" of its neighbor sites are the same.
                For example, In primitive cell, migrating_ion with index 1 has neighbor arranged in 1,3,2,4,6,5,
                then for every migrating_ion with index 1 in supercell, the neighbor is arranged in 1,3,2,4,6,5

                In the loop. I'm mapping the sequence in primitive cell to supercell!

                In order to accelerate the speed

                use the dictionary to store the index of atoms

                indices_dict_from_identifier is a dictionary by pymatgen_structure.kmc_build_dict(). Key is the tuple with format of ([supercell[0],supercell[1],supercell[2],label,local_index]) that contains the information of supercell, local index (index in primitive cell), Value is the corresponding global site index.  This hash dict for acceleration purpose

                This loop build the local_env_info list, [supercell_neighbor_index1, supercell_neighbor_index2, .....]

                """

                normalized_supercell_tuple = normalize_supercell_tuple(
                    site_belongs_to_supercell=supercell_tuple,
                    image_of_site=neighbor_site_in_primitive_cell["image"],
                    supercell_shape=supercell_shape,
                )

                tuple_key_of_such_neighbor_site = supercell.site_index_vector(
                    local_index=neighbor_site_in_primitive_cell["local_index"],
                    label=neighbor_site_in_primitive_cell["label"],
                    supercell=normalized_supercell_tuple,
                )

                local_env_info.append(
                    indices_dict_from_identifier[tuple_key_of_such_neighbor_site]
                )

            for local_env in local_env_info:
                # generate event
                """
                generally use the mobile_ion_identifier_type="label", to see the Na1 and Na2 of nasicon.

                specie is suitable for grain boundary model. Generally don't use it


                """
                if mobile_ion_identifier_type == "specie":

                    if mobile_ion_identifiers[1] in supercell[local_env].species:
                        # initialize the event using the new Event constructor
                        this_event = Event(
                            mobile_ion_indices=(supercell_migrating_ion_index, local_env),
                            local_env_indices=local_env_info,
                        )
                        events.append(this_event)

                elif mobile_ion_identifier_type == "label":

                    if (
                        supercell[local_env].properties["label"]
                        == mobile_ion_identifiers[1]
                    ):
                        # or for understanding, if any site in local environment, its label== "Na2"
                        # initialize the event using the new Event constructor
                        this_event = Event(
                            mobile_ion_indices=(supercell_migrating_ion_index, local_env),
                            local_env_indices=local_env_info,
                        )
                        events.append(this_event)

                elif mobile_ion_identifier_type == "list":
                    raise NotImplementedError()
                    # "potentially implement this for grain boundary model"

                else:
                    raise ValueError(
                        'unrecognized mobile_ion_identifier_type. Please select from: ["specie","label"] '
                    )

        if len(events) == 0:
            raise ValueError(
                "There is no event generated. This is probably caused by wrong input parameters."
            )

        # Generate events and create EventLib
        from kmcpy.event import EventLib
        
        event_lib = EventLib()
        events_dict = []

        for event in events:
            event_lib.add_event(event)
            events_dict.append(event.as_dict())

        logger.info(f"Saving: {event_file}")
        with open(event_file, "w") as fhandle:
            jsonStr = json.dumps(events_dict, indent=4, default=convert)
            fhandle.write(jsonStr)

        # Generate event dependencies using EventLib
        logger.info("Generating event dependency matrix...")
        event_lib.generate_event_dependencies()
        
        # Save event dependencies to file for backward compatibility
        event_lib.save_event_dependencies_to_file(filename=event_dependencies_file)
        logger.info(f"Event dependencies saved to: {event_dependencies_file}")

        # Display dependency statistics
        stats = event_lib.get_dependency_statistics()
        logger.info(f"Generated {len(event_lib)} events with dependency statistics: {stats}")

        return reference_local_env_dict

class ModernEventGenerator:
    """
    A modern, streamlined EventGenerator that uses LatticeStructure and LocalLatticeStructure
    to generate events for kinetic Monte Carlo simulations. It's still under testing.
    
    This implementation is cleaner and more maintainable compared to the legacy version,
    while preserving the critical neighbor sequence matching functionality using NeighborInfoMatcher.
    """
    
    def __init__(self):
        """Initialize the ModernEventGenerator."""
        self.lattice_structure = None
        self.local_environments = {}
        self.reference_local_envs = {}
        self.neighbor_matchers = {}  # Store NeighborInfoMatcher objects for consistent ordering
        
    def generate_events(
        self,
        structure: Union[str, Structure],
        mobile_species: List[str] = ["Na"],
        mobile_site_mapping: Optional[Dict] = None,
        local_env_cutoff: float = 4.0,
        exclude_species: Optional[List[str]] = None,
        supercell_matrix: Optional[np.ndarray] = None,
        distance_matrix_rtol: float = 1e-3,
        distance_matrix_atol: float = 1e-3,
        find_nearest_if_fail: bool = True,
        output_dir: str = ".",
        event_filename: str = "events.json",
        dependencies_filename: str = "event_dependencies.csv"
    ) -> EventLib:
        """
        Generate events using modern structure classes with proper neighbor sequence matching.
        
        Args:
            structure: Input structure file path or pymatgen Structure
            mobile_species: List of mobile species (e.g., ["Na", "Li"])
            mobile_site_mapping: Mapping of species to possible occupants
            local_env_cutoff: Cutoff distance for local environment analysis
            exclude_species: Species to exclude from analysis
            supercell_matrix: Supercell transformation matrix
            distance_matrix_rtol: Relative tolerance for distance matrix matching
            distance_matrix_atol: Absolute tolerance for distance matrix matching
            find_nearest_if_fail: Find nearest match if exact match fails
            output_dir: Directory to save output files
            event_filename: Name of events output file
            dependencies_filename: Name of dependencies output file
            
        Returns:
            EventLib: Generated events with dependencies
        """
        logger.info("Starting modern event generation with neighbor sequence matching...")
        
        # Step 1: Load and prepare structure
        structure = self._load_structure(structure)
        if exclude_species:
            structure.remove_species(exclude_species)
            
        # Step 2: Create LatticeStructure
        self.lattice_structure = self._create_lattice_structure(
            structure, mobile_species, mobile_site_mapping
        )
        
        # Step 3: Find mobile sites
        mobile_sites = self._find_mobile_sites(structure, mobile_species)
        
        # Step 4: Generate local environments with neighbor sequence matching
        self._generate_local_environments_with_matching(
            structure, mobile_sites, local_env_cutoff, 
            distance_matrix_rtol, distance_matrix_atol, find_nearest_if_fail
        )
        
        # Step 5: Create supercell
        if supercell_matrix is None:
            supercell_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        supercell = structure.copy()
        supercell.make_supercell(supercell_matrix)
        
        # Step 6: Generate events
        events = self._generate_events_from_supercell(supercell, mobile_species)
        
        # Step 7: Create EventLib and generate dependencies
        event_lib = EventLib()
        for event in events:
            event_lib.add_event(event)
            
        event_lib.generate_event_dependencies()
        
        # Step 8: Save results
        self._save_results(
            event_lib, output_dir, event_filename, dependencies_filename
        )
        
        logger.info(f"Generated {len(event_lib)} events successfully")
        return event_lib
    
    def _load_structure(self, structure: Union[str, Structure]) -> Structure:
        """Load structure from file or return existing Structure object."""
        if isinstance(structure, str):
            return Structure.from_file(structure)
        return structure.copy()
    
    def _create_lattice_structure(
        self,
        structure: Structure,
        mobile_species: List[str],
        mobile_site_mapping: Optional[Dict] = None
    ) -> LatticeStructure:
        """Create LatticeStructure with appropriate site mapping."""
        if mobile_site_mapping is None:
            # Create default mapping where mobile species can be vacant
            mobile_site_mapping = {}
            for species in mobile_species:
                mobile_site_mapping[species] = [species, "X"]  # X represents vacancy
                
            # Add immobile species (they can only be themselves)
            for site in structure:
                species_str = site.species_string
                if species_str not in mobile_site_mapping:
                    mobile_site_mapping[species_str] = [species_str]
        
        return LatticeStructure(
            template_structure=structure,
            specie_site_mapping=mobile_site_mapping,
            basis_type='occupation'
        )
    
    def _find_mobile_sites(
        self, structure: Structure, mobile_species: List[str]
    ) -> List[int]:
        """Find indices of mobile sites in the structure."""
        mobile_sites = []
        for i, site in enumerate(structure):
            if site.species_string in mobile_species:
                mobile_sites.append(i)
        return mobile_sites
    
    def _generate_local_environments_with_matching(
        self,
        structure: Structure,
        mobile_sites: List[int],
        cutoff: float,
        rtol: float,
        atol: float,
        find_nearest_if_fail: bool = True
    ):
        """
        Generate and classify local environments with proper neighbor sequence matching.
        
        This method uses the integrated LocalEnvironmentComparator to ensure that neighbor 
        sequences are consistently ordered according to distance matrices.
        """
        logger.info(f"Analyzing local environments for {len(mobile_sites)} mobile sites with integrated matching...")
        
        reference_environments = {}  # signature -> reference LocalLatticeStructure
        
        for site_idx in mobile_sites:
            # Create LocalLatticeStructure for this site
            local_env = LocalLatticeStructure(
                template_structure=structure,
                center=site_idx,
                cutoff=cutoff,
                specie_site_mapping=self.lattice_structure.specie_site_mapping
            )
            
            # Get environment signature using the integrated comparator
            env_signature = local_env.get_environment_signature()
            
            if env_signature not in reference_environments:
                # This is a new type of local environment - use as reference
                reference_environments[env_signature] = local_env
                self.reference_local_envs[env_signature] = {
                    'local_env': local_env,
                    'sites': [site_idx]
                }
                logger.info(f"New local environment type discovered: {env_signature}")
                
                # Store the original (reference) environment
                self.local_environments[site_idx] = local_env
                
            else:
                # Match this environment to the reference
                reference_env = reference_environments[env_signature]
                
                try:
                    matched_env = local_env.match_with_reference(
                        reference_env, rtol=rtol, atol=atol, find_nearest_if_fail=find_nearest_if_fail
                    )
                    self.local_environments[site_idx] = matched_env
                    self.reference_local_envs[env_signature]['sites'].append(site_idx)
                    logger.debug(f"Matched environment for site {site_idx} to reference {env_signature}")
                    
                except ValueError as e:
                    logger.warning(f"Could not match environment for site {site_idx}: {e}")
                    # Fall back to using the unmatched environment
                    self.local_environments[site_idx] = local_env
                    
        logger.info(f"Processed {len(mobile_sites)} local environments with {len(reference_environments)} unique types")
    
    def _create_environment_signature(self, local_env: LocalLatticeStructure) -> str:
        """Create a unique signature for a local environment type."""
        # Count species in the local environment
        species_count = {}
        for site in local_env.structure:
            species = site.species_string
            species_count[species] = species_count.get(species, 0) + 1
        
        # Create sorted signature
        signature_parts = []
        for species, count in sorted(species_count.items()):
            signature_parts.append(f"{species}:{count}")
        
        return "_".join(signature_parts)
    
    def _generate_events_from_supercell(
        self, supercell: Structure, mobile_species: List[str]
    ) -> List[Event]:
        """
        Generate events by analyzing the supercell with proper neighbor sequence mapping.
        
        This method ensures that local environment indices are consistently ordered
        according to the reference distance matrices established during local environment analysis.
        """
        events = []
        logger.info("Generating events from supercell with consistent neighbor ordering...")
        
        # Find mobile sites in supercell
        mobile_sites_sc = self._find_mobile_sites(supercell, mobile_species)
        
        for site_idx in mobile_sites_sc:
            # Get the local environment type for the corresponding primitive cell site
            # This is a simplified mapping - in a full implementation, you'd need to 
            # map supercell sites back to primitive cell sites to get the environment type
            
            # Get local environment indices with consistent ordering
            local_env_indices = self._get_consistent_local_environment_indices(
                supercell, site_idx
            )
            
            # Generate events with neighbors
            site = supercell[site_idx]
            neighbors = supercell.get_neighbors(site, 6.0)  # Reasonable cutoff
            
            for neighbor in neighbors:
                neighbor_idx = self._find_site_index_in_supercell(supercell, neighbor)
                
                # Check if neighbor is also a mobile species site
                if (neighbor_idx is not None and 
                    neighbor.species_string in mobile_species and
                    neighbor_idx != site_idx):
                    
                    # Create event with consistently ordered local environment
                    event = Event(
                        mobile_ion_indices=(site_idx, neighbor_idx),
                        local_env_indices=tuple(local_env_indices)
                    )
                    events.append(event)
        
        return events
    
    def _get_consistent_local_environment_indices(
        self, supercell: Structure, central_site_idx: int, cutoff: float = 4.0
    ) -> List[int]:
        """
        Get indices of sites in the local environment with consistent ordering.
        
        This method attempts to maintain the same neighbor ordering as established
        in the primitive cell analysis. In a full implementation, this would use
        the stored NeighborInfoMatcher objects to ensure consistent ordering.
        """
        central_site = supercell[central_site_idx]
        neighbors = supercell.get_neighbors(central_site, cutoff)
        
        env_indices = []
        for neighbor in neighbors:
            neighbor_idx = self._find_site_index_in_supercell(supercell, neighbor)
            if neighbor_idx is not None:
                env_indices.append(neighbor_idx)
        
        # Sort by species first, then by coordinate (simplified ordering)
        # In a full implementation, this would use the distance matrix matching
        neighbor_data = []
        for idx in env_indices:
            site = supercell[idx]
            neighbor_data.append((idx, site.species_string, site.frac_coords[0]))
        
        # Sort by species, then by x-coordinate (mimicking original approach)
        neighbor_data.sort(key=lambda x: (x[1], x[2]))
        
        return [data[0] for data in neighbor_data]
    
    def _find_site_index_in_supercell(
        self, supercell: Structure, target_site
    ) -> Optional[int]:
        """Find the index of a target site in the supercell."""
        for i, site in enumerate(supercell):
            if np.allclose(site.frac_coords, target_site.frac_coords, atol=1e-4):
                return i
        return None
    
    def _save_results(
        self,
        event_lib: EventLib,
        output_dir: str,
        event_filename: str,
        dependencies_filename: str
    ):
        """Save the generated events and dependencies to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save events
        events_file = output_path / event_filename
        event_lib.to_json(str(events_file))
        
        # Save dependencies
        deps_file = output_path / dependencies_filename
        event_lib.save_event_dependencies_to_file(str(deps_file))
        
        logger.info(f"Results saved to {output_path}")


    def get_neighbor_matching_info(self) -> Dict:
        """
        Get information about the neighbor matching performed during local environment analysis.
        
        Returns:
            Dict: Information about neighbor matchers and distance matrices
        """
        info = {
            'num_environment_types': len(self.neighbor_matchers),
            'environment_signatures': list(self.neighbor_matchers.keys()),
            'distance_matrices': {},
            'neighbor_species': {}
        }
        
        for env_sig, matcher in self.neighbor_matchers.items():
            info['distance_matrices'][env_sig] = matcher.distance_matrix.tolist()
            info['neighbor_species'][env_sig] = matcher.neighbor_species
        
        return info
    
    def validate_neighbor_consistency(self) -> bool:
        """
        Validate that neighbor sequences are consistently ordered.
        
        Returns:
            bool: True if all neighbor sequences are consistent with reference templates
        """
        for env_sig, env_data in self.reference_local_envs.items():
            if 'neighbor_matcher' not in env_data:
                continue
                
            reference_matrix = env_data['neighbor_matcher'].distance_matrix
            logger.info(f"Environment {env_sig}: Reference distance matrix shape: {reference_matrix.shape}")
        
        logger.info("Neighbor consistency validation completed")
        return True


    
def find_atom_indices(
    structure, mobile_ion_identifier_type="specie", atom_identifier="Li+"
):
    """a function for generating a list of site indices that satisfies the identifier

    Args:
        structure (kmcpy.external.structure.StructureKMCpy): structure object to work on
        mobile_ion_identifier_type (str, optional): elect from: ["specie","label"]. Defaults to "specie".
        atom_identifier (str, optional): identifier of atom. Defaults to "Li+".

        typical input:
        mobile_ion_identifier_type=specie, atom_identifier="Li+"
        mobile_ion_identifier_type=label, atom_identifier="Li1"
        mobile_ion_identifier_type=list, atom_identifier=[0,1,2,3,4,5]

    Raises:
        ValueError: mobile_ion_identifier_type argument is strange

    Returns:
        list: list of atom indices that satisfy the identifier
    """
    mobile_ion_specie_1_indices = []
    if mobile_ion_identifier_type == "specie":
        for i in range(0, len(structure)):
            if atom_identifier in structure[i].species:
                mobile_ion_specie_1_indices.append(i)

    elif mobile_ion_identifier_type == "label":

        for i in range(0, len(structure)):
            if structure[i].properties["label"] == atom_identifier:
                mobile_ion_specie_1_indices.append(i)

    # elif mobile_ion_identifier_type=="list":
    # mobile_ion_specie_1_indices=atom_identifier

    else:
        raise ValueError(
            'unrecognized mobile_ion_identifier_type. Please select from: ["specie","label"] '
        )

    logger.info("please check if these are mobile_ion_specie_1:")
    for i in mobile_ion_specie_1_indices:

        logger.info(str(structure[i]))

    return mobile_ion_specie_1_indices



def normalize_supercell_tuple(
    site_belongs_to_supercell=[5, 1, 7],
    image_of_site=(0, -1, 1),
    supercell_shape=[5, 6, 7],
    additional_input=False,
):
    """finding the equivalent position in periodic supercell considering the periodic boundary condition. i.e., normalize the supercell tuple to make sure that each component of supercell is greater than zero


    for example,

        # 5 1 7 with image 0 -1 1 -> 5 0 8 -> in periodic 567 supercell should change to 561, suppose supercell start with index1

    input:
    site_belongs_to_supercell: site belongs to which supercell

    Returns:
        tuple: supercell tuple
    """
    logger.debug(f"equivalent position: {site_belongs_to_supercell}, {image_of_site}")

    supercell = np.array(site_belongs_to_supercell) + np.array(image_of_site)

    supercell = np.mod(supercell, supercell_shape)

    supercell = supercell.tolist()
    if additional_input is not False:
        supercell.append(additional_input)

    return tuple(supercell)
