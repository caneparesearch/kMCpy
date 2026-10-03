"""Plain JSON/YAML input-file loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from monty.serialization import loadfn


def load_raw_data(filename: str | Path) -> Any:
    """Load a JSON/YAML file as plain data, without decoding ``@module`` objects.

    Files written by kMCpy contain ``@module`` and ``@class`` keys. Newer monty
    releases decode those into objects for YAML unless ``cls=None`` is passed;
    older releases reject ``cls`` for YAML and never decode it.
    """
    try:
        return loadfn(str(filename), cls=None)
    except TypeError:
        return loadfn(str(filename))
