"""Fast endpoint-state helpers for KMC hops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_HOP_STATE_CODES = (0, 1, 1, 0)
INVALID_STATE = -1


@dataclass(frozen=True)
class HopStateLookup:
    """Precomputed mobile/vacancy state codes for each active site."""

    mobile_state_by_site: np.ndarray
    vacancy_state_by_site: np.ndarray

    @classmethod
    def from_active_site_index_map(
        cls,
        active_site_index_map,
        mobile_ion_specie: str,
    ) -> "HopStateLookup":
        """Build site-indexed state-code arrays from active-site metadata."""
        mobile = np.full(
            active_site_index_map.active_site_count,
            INVALID_STATE,
            dtype=np.int64,
        )
        vacancy = np.full(
            active_site_index_map.active_site_count,
            INVALID_STATE,
            dtype=np.int64,
        )

        for active_index, primitive_index in enumerate(
            active_site_index_map.active_to_primitive
        ):
            allowed = active_site_index_map.allowed_species_by_primitive_site[
                int(primitive_index)
            ]
            for state_index, specie_label in enumerate(allowed):
                if _is_mobile_label(specie_label, mobile_ion_specie):
                    mobile[active_index] = int(state_index)
                if _is_vacancy_label(specie_label):
                    vacancy[active_index] = int(state_index)

        return cls(mobile_state_by_site=mobile, vacancy_state_by_site=vacancy)

    def annotate_event(self, event: Any) -> None:
        """Attach hot-loop state codes to an event in compact active-site space."""
        from_site, to_site = event.mobile_ion_indices
        event.hop_state_codes = (
            int(self.mobile_state_by_site[int(from_site)]),
            int(self.vacancy_state_by_site[int(to_site)]),
            int(self.vacancy_state_by_site[int(from_site)]),
            int(self.mobile_state_by_site[int(to_site)]),
        )

    def annotate_event_lib(self, event_lib: Any) -> None:
        """Attach precomputed state codes to every event in an event library."""
        for event in event_lib.events:
            self.annotate_event(event)


def event_direction(occupations: Any, event: Any) -> int:
    """Return +1, -1, or 0 using only endpoint integer state comparisons."""
    from_site, to_site = event.mobile_ion_indices
    codes = getattr(event, "hop_state_codes", DEFAULT_HOP_STATE_CODES)
    from_occ = int(occupations[int(from_site)])
    to_occ = int(occupations[int(to_site)])
    return endpoint_direction_from_codes(from_occ, to_occ, codes)


def endpoint_direction_from_codes(
    from_occ: int,
    to_occ: int,
    codes: tuple[int, int, int, int],
) -> int:
    """Return direction from endpoint states and precomputed hop state codes."""
    if from_occ == codes[0] and to_occ == codes[1]:
        return 1
    if from_occ == codes[2] and to_occ == codes[3]:
        return -1
    return 0


def _is_mobile_label(specie_label: str, mobile_ion_specie: str) -> bool:
    return str(specie_label) == str(mobile_ion_specie)


def _is_vacancy_label(specie_label: str) -> bool:
    return str(specie_label) in {"X", "Vacancy", "vacancy", "Va", "VA"}
