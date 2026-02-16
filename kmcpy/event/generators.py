#!/usr/bin/env python
"""
This module provides tools for generating and matching local atomic environments and events for kinetic Monte Carlo (kMC) simulations, particularly for ionic diffusion in crystalline solids. It includes utilities for neighbor environment matching, event generation, and supercell normalization, with support for structures parsed by pymatgen.
"""

import itertools
import json
import logging
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymatgen.util.coord import get_angle

from kmcpy.event.base import Event, EventLib
from kmcpy.structure.local_lattice_structure import LocalLatticeStructure

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
    """
    Generate migration events by combining point+cutoff environment detection with
    primitive-template expansion into a supercell.

    This keeps neighbor searches in the primitive cell only, then maps pre-ordered
    templates to each supercell image via dictionary lookups for performance.
    """

    def __init__(self):
        self.reference_local_env_dict: Dict = {}
        self.local_env_info_dict: Dict[int, List[Dict]] = {}

    @staticmethod
    def _to_identifier_list(identifier: Any) -> List[str]:
        if isinstance(identifier, str):
            return [identifier]
        if isinstance(identifier, Iterable):
            return [str(value) for value in identifier]
        return [str(identifier)]

    @classmethod
    def _site_matches_species_identifier(cls, site, identifier: Any) -> bool:
        candidate_tokens = set()

        species_string = getattr(site, "species_string", None)
        if species_string is not None:
            candidate_tokens.add(str(species_string))

        specie = getattr(site, "specie", None)
        if specie is not None:
            candidate_tokens.add(str(specie))
            symbol = getattr(specie, "symbol", None)
            if symbol is not None:
                candidate_tokens.add(str(symbol))
            element = getattr(specie, "element", None)
            if element is not None:
                candidate_tokens.add(str(element))

        for specie_obj in getattr(site, "species", {}).keys():
            candidate_tokens.add(str(specie_obj))
            symbol = getattr(specie_obj, "symbol", None)
            if symbol is not None:
                candidate_tokens.add(str(symbol))
            element = getattr(specie_obj, "element", None)
            if element is not None:
                candidate_tokens.add(str(element))

        identifier_tokens = set(cls._to_identifier_list(identifier))
        return not candidate_tokens.isdisjoint(identifier_tokens)

    @classmethod
    def _site_matches_label_identifier(cls, site, identifier: Any) -> bool:
        site_label = site.properties.get("label")
        return str(site_label) in set(cls._to_identifier_list(identifier))

    @classmethod
    def _build_uniform_cutoff_dict(
        cls,
        structure,
        mobile_species: List[str],
        local_env_cutoff: float,
    ) -> Dict[Tuple[str, str], float]:
        all_species = sorted({site.species_string for site in structure})
        mobile_species_in_structure = sorted(
            {
                site.species_string
                for site in structure
                if cls._site_matches_species_identifier(site, mobile_species)
            }
        )
        if not mobile_species_in_structure:
            raise ValueError(
                f"No mobile species found in structure for requested mobile_species={mobile_species}"
            )

        cutoff_dict: Dict[Tuple[str, str], float] = {}
        for mobile_sp in mobile_species_in_structure:
            for neighbor_sp in all_species:
                cutoff_dict[(mobile_sp, neighbor_sp)] = float(local_env_cutoff)
        return cutoff_dict

    def _normalize_generate_events_inputs(
        self,
        mobile_ion_identifier_type: str,
        mobile_ion_identifiers,
        species_to_be_removed: Optional[List[str]],
        distance_matrix_rtol: float,
        distance_matrix_atol: float,
        supercell_shape: Optional[List[int]],
        local_env_cutoff_dict: Optional[Dict[Tuple[str, str], float]],
        mobile_species: Optional[List[str]],
        mobile_site_mapping: Optional[Dict],
        local_env_cutoff: Optional[float],
        exclude_species: Optional[List[str]],
        rtol: Optional[float],
        atol: Optional[float],
    ):
        new_style_requested = any(
            value is not None
            for value in (
                mobile_species,
                mobile_site_mapping,
                local_env_cutoff,
                exclude_species,
                rtol,
                atol,
            )
        )

        if rtol is not None:
            distance_matrix_rtol = rtol
        if atol is not None:
            distance_matrix_atol = atol

        if supercell_shape is None:
            supercell_shape = [2, 1, 1]

        if new_style_requested:
            if mobile_species is None:
                mobile_species = []
                if mobile_site_mapping:
                    for key, value in mobile_site_mapping.items():
                        values = value if isinstance(value, list) else [value]
                        value_tokens = {str(v) for v in values}
                        if "X" in value_tokens:
                            mobile_species.append(str(key))
                if not mobile_species:
                    mobile_species = ["Na"]

            if species_to_be_removed is None:
                species_to_be_removed = list(exclude_species or [])
            elif exclude_species:
                species_to_be_removed = list(
                    dict.fromkeys(list(species_to_be_removed) + list(exclude_species))
                )

            if (
                mobile_ion_identifier_type == "label"
                and mobile_ion_identifiers == ("Na1", "Na2")
            ):
                mobile_ion_identifier_type = "specie"
                mobile_ion_identifiers = (mobile_species, mobile_species)

            if local_env_cutoff is None and local_env_cutoff_dict is None:
                local_env_cutoff = 4.0

        else:
            if species_to_be_removed is None:
                species_to_be_removed = ["O2-", "O"]
            if local_env_cutoff_dict is None:
                local_env_cutoff_dict = {("Li+", "Cl-"): 4.0, ("Li+", "Li+"): 3.0}

        return (
            new_style_requested,
            mobile_ion_identifier_type,
            mobile_ion_identifiers,
            species_to_be_removed,
            distance_matrix_rtol,
            distance_matrix_atol,
            supercell_shape,
            local_env_cutoff_dict,
            mobile_species,
            local_env_cutoff,
        )

    def _match_or_register_local_environment(
        self,
        primitive_cell,
        migrating_ion_index: int,
        unsorted_neighbor_sequence: List[Dict],
        reference_local_env_dict: Dict,
        local_env_info_dict: Dict[int, List[Dict]],
        distance_matrix_rtol: float,
        distance_matrix_atol: float,
        find_nearest_if_fail: bool,
    ):
        """Register a new local environment type or match to an existing reference."""
        this_nninfo = NeighborInfoMatcher.from_neighbor_sequences(
            unsorted_neighbor_sequence
        )
        local_index = primitive_cell[migrating_ion_index].properties["local_index"]

        if this_nninfo.neighbor_species not in reference_local_env_dict:
            reference_local_env_dict[this_nninfo.neighbor_species] = this_nninfo
            local_env_info_dict[local_index] = this_nninfo.neighbor_sequence
            return True, this_nninfo

        sorted_neighbor_sequence = reference_local_env_dict[
            this_nninfo.neighbor_species
        ].brutal_match(
            this_nninfo.neighbor_sequence,
            rtol=distance_matrix_rtol,
            atol=distance_matrix_atol,
            find_nearest_if_fail=find_nearest_if_fail,
        )
        local_env_info_dict[local_index] = sorted_neighbor_sequence
        return False, this_nninfo

    def _export_reference_local_environment(
        self,
        structure_cls,
        primitive_cell,
        migrating_ion_index: int,
        unsorted_neighbor_sequence: List[Dict],
        reference_local_env_type: int,
    ) -> int:
        """Export one reference local environment to CIF for inspection."""
        reference_local_env_sites = [primitive_cell[migrating_ion_index]]
        reference_local_env_sites.extend(
            [neighbor["site"] for neighbor in unsorted_neighbor_sequence]
        )

        reference_local_env_structure = structure_cls.from_sites(
            sites=reference_local_env_sites
        )
        reference_local_env_structure.to(
            fmt="cif",
            filename=f"{reference_local_env_type}th_reference_local_env.cif",
        )
        logger.info(
            "%sth type of reference local_env structure cif file is created. please check",
            reference_local_env_type + 1,
        )
        return reference_local_env_type + 1

    def _is_valid_target_site(
        self,
        site,
        mobile_ion_identifier_type: str,
        target_identifier: Any,
    ) -> bool:
        if mobile_ion_identifier_type == "specie":
            return self._site_matches_species_identifier(site, target_identifier)
        if mobile_ion_identifier_type == "label":
            return self._site_matches_label_identifier(site, target_identifier)
        raise ValueError(
            'unrecognized mobile_ion_identifier_type. Please select from: ["specie","label"] '
        )

    def generate_events(
        self,
        structure_file: str = "210.cif",
        convert_to_primitive_cell: bool = False,
        local_env_cutoff_dict: Optional[Dict[Tuple[str, str], float]] = None,
        mobile_ion_identifier_type: str = "label",
        mobile_ion_identifiers: Tuple[str, str] = ("Na1", "Na2"),
        species_to_be_removed: Optional[List[str]] = None,
        distance_matrix_rtol: float = 0.01,
        distance_matrix_atol: float = 0.01,
        find_nearest_if_fail: bool = True,
        export_local_env_structure: bool = False,
        supercell_shape: Optional[List[int]] = None,
        event_file: str = "events.json",
        event_dependencies_file: str = "event_dependencies.csv",
        mobile_species: Optional[List[str]] = None,
        mobile_site_mapping: Optional[Dict] = None,
        local_env_cutoff: Optional[float] = None,
        exclude_species: Optional[List[str]] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ) -> Dict:
        """
        Generate migration events and event dependencies from a CIF structure.

        Event generation always follows the performant primitive-template -> supercell
        expansion workflow.

        This method accepts both:
        1. The legacy argument style (label/specie identifiers and cutoff dict).
        2. The newer argument style (mobile_species/local_env_cutoff/exclude_species/rtol/atol).

        New-style arguments are normalized into the same internal legacy backend.
        """
        from kmcpy.external.local_env import CutOffDictNNKMCpy
        from kmcpy.external.structure import StructureKMCpy
        from kmcpy.io import convert
        import kmcpy

        (
            new_style_requested,
            mobile_ion_identifier_type,
            mobile_ion_identifiers,
            species_to_be_removed,
            distance_matrix_rtol,
            distance_matrix_atol,
            supercell_shape,
            local_env_cutoff_dict,
            mobile_species,
            local_env_cutoff,
        ) = self._normalize_generate_events_inputs(
            mobile_ion_identifier_type=mobile_ion_identifier_type,
            mobile_ion_identifiers=mobile_ion_identifiers,
            species_to_be_removed=species_to_be_removed,
            distance_matrix_rtol=distance_matrix_rtol,
            distance_matrix_atol=distance_matrix_atol,
            supercell_shape=supercell_shape,
            local_env_cutoff_dict=local_env_cutoff_dict,
            mobile_species=mobile_species,
            mobile_site_mapping=mobile_site_mapping,
            local_env_cutoff=local_env_cutoff,
            exclude_species=exclude_species,
            rtol=rtol,
            atol=atol,
        )

        if len(mobile_ion_identifiers) != 2:
            raise ValueError(
                "mobile_ion_identifiers must contain two identifiers: (initial, target)."
            )

        logger.info(kmcpy.get_logo())
        primitive_cell = StructureKMCpy.from_cif(
            structure_file, primitive=convert_to_primitive_cell
        )
        primitive_cell.add_oxidation_state_by_guess()
        if species_to_be_removed:
            primitive_cell.remove_species(species_to_be_removed)

        if local_env_cutoff_dict is None:
            if new_style_requested:
                local_env_cutoff_dict = self._build_uniform_cutoff_dict(
                    structure=primitive_cell,
                    mobile_species=mobile_species or ["Na"],
                    local_env_cutoff=local_env_cutoff or 4.0,
                )
            else:
                local_env_cutoff_dict = {("Li+", "Cl-"): 4.0, ("Li+", "Li+"): 3.0}

        logger.info(
            "primitive cell composition after adding oxidation state and removing uninvolved species: %s",
            primitive_cell.composition,
        )
        logger.info("building migrating_ion index list")

        migrating_ion_indices = find_atom_indices(
            primitive_cell,
            mobile_ion_identifier_type=mobile_ion_identifier_type,
            atom_identifier=mobile_ion_identifiers[0],
        )

        local_env_finder = CutOffDictNNKMCpy(local_env_cutoff_dict)
        reference_local_env_dict: Dict = {}
        local_env_info_dict: Dict[int, List[Dict]] = {}

        logger.info("start finding the neighboring sequence of migrating_ions")
        logger.info("total number of migrating_ions:%s", len(migrating_ion_indices))

        reference_local_env_type = 0
        for index, migrating_ion_index in enumerate(migrating_ion_indices, start=1):
            unsorted_neighbor_sequences = (
                LocalLatticeStructure.ordered_neighbor_info_from_finder(
                    primitive_cell, migrating_ion_index, local_env_finder
                )
            )

            is_new_type, this_nninfo = self._match_or_register_local_environment(
                primitive_cell=primitive_cell,
                migrating_ion_index=migrating_ion_index,
                unsorted_neighbor_sequence=unsorted_neighbor_sequences,
                reference_local_env_dict=reference_local_env_dict,
                local_env_info_dict=local_env_info_dict,
                distance_matrix_rtol=distance_matrix_rtol,
                distance_matrix_atol=distance_matrix_atol,
                find_nearest_if_fail=find_nearest_if_fail,
            )

            if is_new_type and export_local_env_structure:
                reference_local_env_type = self._export_reference_local_environment(
                    StructureKMCpy,
                    primitive_cell,
                    migrating_ion_index,
                    unsorted_neighbor_sequences,
                    reference_local_env_type,
                )

            logger.info(
                "local environment %s with species %s has distance matrix:\n%s",
                "registered" if is_new_type else "matched",
                this_nninfo.neighbor_species,
                this_nninfo.distance_matrix,
            )
            logger.info(
                "%s out of %s neighboring sequence has been found",
                index,
                len(migrating_ion_indices),
            )

        supercell = primitive_cell.make_kmc_supercell(supercell_shape)
        logger.info("supercell is created")
        logger.info(str(supercell))

        supercell_migrating_ion_indices = find_atom_indices(
            supercell,
            mobile_ion_identifier_type=mobile_ion_identifier_type,
            atom_identifier=mobile_ion_identifiers[0],
        )
        indices_dict_from_identifier = supercell.kmc_build_dict(skip_check=False)

        events: List[Event] = []
        for supercell_migrating_ion_index in supercell_migrating_ion_indices:
            supercell_tuple = supercell[supercell_migrating_ion_index].properties[
                "supercell"
            ]
            local_index_of_this_migrating_ion = supercell[
                supercell_migrating_ion_index
            ].properties["local_index"]

            local_env_info: List[int] = []
            for neighbor_site_in_primitive_cell in local_env_info_dict[
                local_index_of_this_migrating_ion
            ]:
                normalized_supercell_tuple = normalize_supercell_tuple(
                    site_belongs_to_supercell=supercell_tuple,
                    image_of_site=neighbor_site_in_primitive_cell["image"],
                    supercell_shape=supercell_shape,
                )
                tuple_key_of_neighbor_site = supercell.site_index_vector(
                    local_index=neighbor_site_in_primitive_cell["local_index"],
                    label=neighbor_site_in_primitive_cell["label"],
                    supercell=normalized_supercell_tuple,
                )
                local_env_info.append(
                    indices_dict_from_identifier[tuple_key_of_neighbor_site]
                )

            local_env_indices = tuple(local_env_info)
            for target_site_index in local_env_info:
                if self._is_valid_target_site(
                    supercell[target_site_index],
                    mobile_ion_identifier_type,
                    mobile_ion_identifiers[1],
                ):
                    events.append(
                        Event(
                            mobile_ion_indices=(
                                supercell_migrating_ion_index,
                                target_site_index,
                            ),
                            local_env_indices=local_env_indices,
                        )
                    )

        if not events:
            raise ValueError(
                "There is no event generated. This is probably caused by wrong input parameters."
            )

        event_lib = EventLib()
        events_dict = []
        for event in events:
            event_lib.add_event(event)
            events_dict.append(event.as_dict())

        logger.info("Saving: %s", event_file)
        with open(event_file, "w") as fhandle:
            json.dump(events_dict, fhandle, indent=4, default=convert)

        logger.info("Generating event dependency matrix...")
        event_lib.generate_event_dependencies()
        event_lib.save_event_dependencies_to_file(filename=event_dependencies_file)
        logger.info("Event dependencies saved to: %s", event_dependencies_file)

        stats = event_lib.get_dependency_statistics()
        logger.info(
            "Generated %s events with dependency statistics: %s",
            len(event_lib),
            stats,
        )

        self.reference_local_env_dict = reference_local_env_dict
        self.local_env_info_dict = local_env_info_dict
        return reference_local_env_dict


    
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
            if EventGenerator._site_matches_species_identifier(
                structure[i], atom_identifier
            ):
                mobile_ion_specie_1_indices.append(i)

    elif mobile_ion_identifier_type == "label":

        for i in range(0, len(structure)):
            if EventGenerator._site_matches_label_identifier(
                structure[i], atom_identifier
            ):
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
