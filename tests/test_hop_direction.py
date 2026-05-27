import types

import numpy as np
from pymatgen.core import Lattice, Structure

from kmcpy.event import Event, EventLib
from kmcpy.simulator.hop import (
    HopStateLookup,
    endpoint_direction_from_codes,
    event_direction,
)
from kmcpy.structure.active_site_order import ActiveSiteOrder


def test_endpoint_direction_uses_precomputed_state_codes():
    codes = (2, 1, 1, 2)

    assert endpoint_direction_from_codes(2, 1, codes) == 1
    assert endpoint_direction_from_codes(1, 2, codes) == -1
    assert endpoint_direction_from_codes(0, 1, codes) == 0
    assert endpoint_direction_from_codes(2, 0, codes) == 0


def test_event_direction_defaults_to_binary_state_indices():
    event = Event(mobile_ion_indices=(0, 1), local_env_indices=())

    assert event_direction([0, 1], event) == 1
    assert event_direction([1, 0], event) == -1
    assert event_direction([0, 0], event) == 0


def test_hop_state_lookup_annotates_multistate_mobile_sites():
    lattice = Lattice.cubic(4.0)
    structure = Structure(
        lattice,
        ["Na", "Na"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    index_map = ActiveSiteOrder.from_structure_and_mapping(
        structure,
        {"Na": ["Mg", "X", "Na"]},
    )
    lookup = HopStateLookup.from_active_site_order(index_map, "Na")
    event = Event(mobile_ion_indices=(0, 1), local_env_indices=())

    lookup.annotate_event(event)

    assert event.hop_state_codes == (2, 1, 1, 2)
    assert event_direction([2, 1], event) == 1
    assert event_direction([1, 2], event) == -1
    assert event_direction([0, 1], event) == 0


def test_hop_state_lookup_annotates_event_libraries():
    event_lib = EventLib()
    event_lib.events = [
        Event(mobile_ion_indices=(0, 1), local_env_indices=()),
        Event(mobile_ion_indices=(1, 0), local_env_indices=()),
    ]
    lookup = HopStateLookup(
        mobile_state_by_site=np.array([2, 2], dtype=np.int64),
        vacancy_state_by_site=np.array([1, 1], dtype=np.int64),
    )

    lookup.annotate_event_lib(event_lib)

    assert event_lib.events[0].hop_state_codes == (2, 1, 1, 2)
    assert event_lib.events[1].hop_state_codes == (2, 1, 1, 2)


def test_hop_state_lookup_marks_non_mobile_endpoints_unavailable():
    event = types.SimpleNamespace(mobile_ion_indices=(0, 1))
    lookup = HopStateLookup(
        mobile_state_by_site=np.array([-1, 2], dtype=np.int64),
        vacancy_state_by_site=np.array([1, 1], dtype=np.int64),
    )

    lookup.annotate_event(event)

    assert event_direction([0, 1], event) == 0
