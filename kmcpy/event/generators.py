#!/usr/bin/env python
"""
This module provides tools for generating and matching local atomic environments and events for kinetic Monte Carlo (kMC) simulations, particularly for ionic diffusion in crystalline solids. It includes utilities for neighbor environment matching, event generation, and supercell normalization, with support for structures parsed by pymatgen.
"""

import logging
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from kmcpy.event.base import Event, EventLib
from kmcpy.io.cif import load_labeled_structure_from_cif
from kmcpy.structure.active_site_index_map import ActiveSiteIndexMap
from kmcpy.structure.local_lattice_structure import LocalLatticeStructure
from kmcpy.structure.cluster import Cluster, ClusterMatcher
from kmcpy.structure.neighbors import (
    get_cutoff_neighbor_info,
    prepare_cutoff_neighbor_lookup,
)
from kmcpy.structure.sites import (
    build_site_index,
    make_kmc_supercell,
    site_index_key,
    structure_from_sites,
)

logger = logging.getLogger(__name__) 

def print_divider():
    logger.info("\n\n-------------------------------------------\n\n")

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
    def _neighbor_species_signature(
        neighbor_sequence: List[Dict],
    ) -> tuple[tuple[str, int], ...]:
        species_counts: dict[str, int] = {}
        for neighbor in neighbor_sequence:
            species = neighbor["site"].species_string
            species_counts[species] = species_counts.get(species, 0) + 1
        return tuple(sorted(species_counts.items(), key=lambda item: item[0]))

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
        mobile_ion_identifier_type: Optional[str],
        mobile_ion_identifiers,
        species_to_be_removed: Optional[List[str]],
        distance_matrix_rtol: float,
        distance_matrix_atol: float,
        supercell_shape: Optional[List[int]],
        local_env_cutoff_dict: Optional[Dict[Tuple[str, str], float]],
        mobile_species: Optional[List[str]],
        site_mapping: Optional[Dict],
        local_env_cutoff: Optional[float],
        exclude_species: Optional[List[str]],
        rtol: Optional[float],
        atol: Optional[float],
    ):
        new_style_requested = any(
            value is not None
            for value in (
                mobile_species,
                site_mapping,
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

        if species_to_be_removed is not None or exclude_species is not None:
            raise ValueError(
                "species_to_be_removed and exclude_species are no longer supported; "
                "use site_mapping to define active and fixed sites."
            )
        if site_mapping is None:
            raise ValueError(
                "site_mapping is required to generate events in the "
                "active-site index space."
            )

        if new_style_requested:
            if mobile_species is None:
                mobile_species = []
                if site_mapping:
                    for key, value in site_mapping.items():
                        values = value if isinstance(value, list) else [value]
                        value_tokens = {str(v) for v in values}
                        if "X" in value_tokens:
                            mobile_species.append(str(key))
                if not mobile_species:
                    mobile_species = ["Na"]

            if mobile_ion_identifier_type is None:
                mobile_ion_identifier_type = "specie"
            if mobile_ion_identifiers is None:
                if mobile_ion_identifier_type == "specie":
                    mobile_ion_identifiers = (mobile_species, mobile_species)
                else:
                    mobile_ion_identifiers = ("Na1", "Na2")

            if local_env_cutoff is None and local_env_cutoff_dict is None:
                local_env_cutoff = 4.0

        else:
            if mobile_ion_identifier_type is None:
                mobile_ion_identifier_type = "label"
            if mobile_ion_identifiers is None:
                mobile_ion_identifiers = ("Na1", "Na2")
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
        """Register a primitive-cell environment or map it to a reference order."""
        neighbor_species = self._neighbor_species_signature(unsorted_neighbor_sequence)
        candidate_cluster = Cluster.from_neighbor_info(
            unsorted_neighbor_sequence
        )
        local_index = primitive_cell[migrating_ion_index].properties["local_index"]

        if neighbor_species not in reference_local_env_dict:
            reference_local_env_dict[neighbor_species] = {
                "cluster": candidate_cluster,
                "neighbor_sequence": list(unsorted_neighbor_sequence),
            }
            local_env_info_dict[local_index] = list(unsorted_neighbor_sequence)
            return True, neighbor_species, candidate_cluster

        reference_cluster = reference_local_env_dict[neighbor_species]["cluster"]
        match = ClusterMatcher(
            reference_cluster,
            rtol=distance_matrix_rtol,
            atol=distance_matrix_atol,
        ).match(
            candidate_cluster,
            find_nearest_if_fail=find_nearest_if_fail,
        )
        local_env_info_dict[local_index] = [
            unsorted_neighbor_sequence[index]
            for index in match.reference_to_candidate
        ]
        return False, neighbor_species, candidate_cluster

    def _export_reference_local_environment(
        self,
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

        reference_local_env_structure = structure_from_sites(reference_local_env_sites)
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
        mobile_ion_identifier_type: Optional[str] = None,
        mobile_ion_identifiers: Optional[Tuple[str, str]] = None,
        species_to_be_removed: Optional[List[str]] = None,
        distance_matrix_rtol: float = 0.01,
        distance_matrix_atol: float = 0.01,
        find_nearest_if_fail: bool = True,
        export_local_env_structure: bool = False,
        supercell_shape: Optional[List[int]] = None,
        event_file: str = "events.json",
        mobile_species: Optional[List[str]] = None,
        site_mapping: Optional[Dict] = None,
        local_env_cutoff: Optional[float] = None,
        exclude_species: Optional[List[str]] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ) -> Dict:
        """
        Generate migration events and save as bundled event library.

        Events are saved with embedded dependencies in a single JSON file.
        The dependency matrix is computed and stored as an attribute of EventLib.

        Event generation always follows the performant primitive-template -> supercell
        expansion workflow.

        This method accepts label/specie hop identifiers or mobile_species, but
        site_mapping is always required because generated events are stored in the
        active-site index space.
        """
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
            site_mapping=site_mapping,
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
        full_primitive_cell = load_labeled_structure_from_cif(
            structure_file, primitive=convert_to_primitive_cell
        )
        full_primitive_cell.add_oxidation_state_by_guess()
        primitive_active_site_index_map = ActiveSiteIndexMap.from_structure_and_mapping(
            full_primitive_cell, site_mapping
        )
        event_active_site_index_map = ActiveSiteIndexMap.from_structure_and_mapping(
            full_primitive_cell,
            site_mapping,
            supercell_shape=supercell_shape,
        )
        primitive_cell = structure_from_sites(
            primitive_active_site_index_map.active_structure().sites
        )

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

        local_env_cutoff_lookup = prepare_cutoff_neighbor_lookup(local_env_cutoff_dict)
        reference_local_env_dict: Dict = {}
        local_env_info_dict: Dict[int, List[Dict]] = {}

        logger.info("start finding the neighboring sequence of migrating_ions")
        logger.info("total number of migrating_ions:%s", len(migrating_ion_indices))

        reference_local_env_type = 0
        for index, migrating_ion_index in enumerate(migrating_ion_indices, start=1):
            unsorted_neighbor_sequences = LocalLatticeStructure.sort_neighbor_info(
                get_cutoff_neighbor_info(
                    primitive_cell,
                    migrating_ion_index,
                    local_env_cutoff_lookup,
                )
            )

            is_new_type, neighbor_species, cluster = self._match_or_register_local_environment(
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
                    primitive_cell,
                    migrating_ion_index,
                    unsorted_neighbor_sequences,
                    reference_local_env_type,
                )

            logger.info(
                "local environment %s with species %s has distance matrix:\n%s",
                "registered" if is_new_type else "matched",
                neighbor_species,
                cluster.distance_matrix,
            )
            logger.info(
                "%s out of %s neighboring sequence has been found",
                index,
                len(migrating_ion_indices),
            )

        supercell = make_kmc_supercell(primitive_cell, supercell_shape)
        logger.info("supercell is created")
        logger.info(str(supercell))

        supercell_migrating_ion_indices = find_atom_indices(
            supercell,
            mobile_ion_identifier_type=mobile_ion_identifier_type,
            atom_identifier=mobile_ion_identifiers[0],
        )
        indices_dict_from_identifier = build_site_index(supercell, skip_check=False)

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
                tuple_key_of_neighbor_site = site_index_key(
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
        for event in events:
            event_lib.add_event(event)
        event_lib.set_index_metadata(event_active_site_index_map)

        # Generate dependencies before saving
        logger.info("Generating event dependency matrix...")
        event_lib.generate_event_dependencies()

        # Save in bundled format (events + dependencies in single file)
        logger.info("Saving bundled event library to: %s", event_file)
        event_lib.to_json(event_file)

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
        structure (Structure): structure object to work on
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
