"""Tests for shared callable helpers."""

import pytest

from kmcpy.callables import (
    accepts_keyword,
    call_with_supported_keywords,
    resolve_callable_reference,
    supported_keyword_names,
)


def _event_only(event):
    return ("event", event)


def _keyword_only(*, event, state=None):
    return ("keyword_only", event, state)


def _any_keywords(**kwargs):
    return ("kwargs", sorted(kwargs))


class _UnhashableCallable:
    __hash__ = None

    def __call__(self, changes):
        return ("unhashable", changes)


@pytest.mark.unit
def test_accepts_keyword():
    assert accepts_keyword(_event_only, "event")
    assert not accepts_keyword(_event_only, "state")
    assert accepts_keyword(_keyword_only, "state")
    assert accepts_keyword(_any_keywords, "anything")
    assert not accepts_keyword(object(), "event")


@pytest.mark.unit
def test_supported_keyword_names_and_filtered_calls():
    kwargs = {"event": 1, "state": 2, "changes": 3}

    assert supported_keyword_names(_event_only) == {"event"}
    assert supported_keyword_names(_any_keywords) is None
    assert call_with_supported_keywords(_event_only, kwargs) == ("event", 1)
    assert call_with_supported_keywords(_keyword_only, kwargs) == ("keyword_only", 1, 2)
    assert call_with_supported_keywords(_any_keywords, kwargs) == (
        "kwargs", ["changes", "event", "state"]
    )
    assert call_with_supported_keywords(_UnhashableCallable(), kwargs) == ("unhashable", 3)


@pytest.mark.unit
def test_resolve_callable_reference():
    assert resolve_callable_reference("tests.test_callables:_event_only") is _event_only
    assert resolve_callable_reference("tests.test_callables._event_only") is _event_only
    with pytest.raises(ValueError, match="Invalid callable reference"):
        resolve_callable_reference("no_module_path")
    with pytest.raises(ValueError, match="Invalid callable reference"):
        resolve_callable_reference("tests.test_callables:")
    with pytest.raises(TypeError, match="is not callable"):
        resolve_callable_reference("tests.test_callables:pytest.__name__")
