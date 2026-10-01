"""Tests for the mutable simulation State."""

import pytest
from monty.serialization import dumpfn

from kmcpy.event import Event
from kmcpy.simulator.state import State


@pytest.mark.unit
def test_apply_event_swaps_endpoints_and_advances_counters():
    state = State(occupations=[0, 1, 0])
    state.apply_event(Event(mobile_ion_indices=(0, 1), local_env_indices=(2,)), dt=0.5)

    assert state.occupations == [1, 0, 0]
    assert state.time == 0.5
    assert state.step == 1


@pytest.mark.unit
def test_copy_and_constructor_do_not_share_occupations():
    initial = [0, 1]
    state = State(occupations=initial, time=1.0, step=3)
    copy = state.copy()

    initial[0] = 9
    copy.occupations[1] = 7
    assert state.occupations == [0, 1]
    assert (copy.time, copy.step) == (1.0, 3)


@pytest.mark.unit
def test_dict_and_json_checkpoint_round_trip(tmp_path):
    state = State(occupations=[1, 0, 2], time=2.5, step=4)

    restored = State.from_dict(state.as_dict())
    assert (restored.occupations, restored.time, restored.step) == ([1, 0, 2], 2.5, 4)

    checkpoint = tmp_path / "state.json"
    state.save_checkpoint(str(checkpoint))
    loaded = State.load_checkpoint(str(checkpoint))
    assert (loaded.occupations, loaded.time, loaded.step) == ([1, 0, 2], 2.5, 4)


@pytest.mark.unit
def test_from_file_reads_initial_state_occupation_payload(tmp_path):
    initial_state = tmp_path / "initial_state.json"
    # Two primitive sites in a (2, 1, 1) supercell, stored site-major.
    dumpfn({"occupation": [0, 1, 1, 0]}, initial_state)

    state = State.from_file(str(initial_state), supercell_shape=(2, 1, 1), select_sites=[1])
    assert state.occupations == [1, 0]

    with pytest.raises(ValueError, match="select_sites or active_site_order"):
        State.from_file(str(initial_state))


@pytest.mark.unit
def test_from_file_rejects_unknown_formats(tmp_path):
    with pytest.raises(ValueError, match=".json or .h5"):
        State.from_file(str(tmp_path / "state.txt"))

    payload = tmp_path / "state.json"
    dumpfn({"something": 1}, payload)
    with pytest.raises(ValueError, match="either 'occupations' or initial-state 'occupation'"):
        State.from_file(str(payload))
