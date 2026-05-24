"""Helpers for loading occupation vectors from simulation state files."""

from __future__ import annotations

import json
import logging

import numpy as np

logger = logging.getLogger(__name__)


def load_occupation_data(
    initial_state_file: str,
    supercell_shape: list,
    select_sites: list | None = None,
    active_site_index_map=None,
) -> list:
    """
    Load occupation data from an initial state JSON file.

    The returned values use the Chebyshev basis expected by the simulator:
    site-state index ``0`` maps to ``-1`` and index ``1`` maps to ``+1``.
    """
    with open(initial_state_file, "r") as f:
        occupation_raw_data = np.array(json.load(f)["occupation"])

    supercell_size = supercell_shape[0] * supercell_shape[1] * supercell_shape[2]
    if len(occupation_raw_data) % supercell_size != 0:
        logger.error(
            f"The length of occupation data {len(occupation_raw_data)} is incompatible with the supercell shape"
        )
        raise ValueError(
            f"The length of occupation data {len(occupation_raw_data)} is incompatible with the supercell shape"
        )

    site_nums = int(len(occupation_raw_data) / supercell_size)
    convert_to_dimension = (
        site_nums,
        supercell_shape[0],
        supercell_shape[1],
        supercell_shape[2],
    )

    if active_site_index_map is not None:
        if len(occupation_raw_data) == active_site_index_map.active_site_count:
            occupation = occupation_raw_data
        elif site_nums == active_site_index_map.primitive_site_count:
            occupation = occupation_raw_data.reshape(convert_to_dimension)[
                list(active_site_index_map.primitive_active_indices)
            ].flatten("C")
        elif len(occupation_raw_data) == active_site_index_map.original_site_count:
            occupation = occupation_raw_data[list(active_site_index_map.active_to_original)]
        else:
            raise ValueError(
                "Initial state length does not match the active-site index map."
            )
        selected = list(active_site_index_map.primitive_active_indices)
    else:
        if select_sites is None:
            raise ValueError(
                "select_sites or active_site_index_map is required to load occupations"
            )
        occupation = occupation_raw_data.reshape(convert_to_dimension)[select_sites].flatten("C")
        selected = select_sites

    occupation_chebyshev = np.where(occupation == 0, -1, 1)

    logger.debug(f"Selected active sites are {selected}")
    logger.debug(f"Converting the occupation raw data to dimension: {convert_to_dimension}")
    logger.debug(f"Occupation in compact active-site basis: {occupation_chebyshev}")

    return occupation_chebyshev.tolist()
