"""Species normalization helpers for kMCpy structure code."""

from __future__ import annotations

from typing import Any

from pymatgen.core import DummySpecies, Species


VACANCY_LABELS = frozenset({"X", "Vacancy", "vacancy", "Va", "VA"})


def vacancy_species() -> DummySpecies:
    """Return the internal pymatgen species used for vacancy sites."""
    return DummySpecies("X", 0)


def is_vacancy_species(specie: Any) -> bool:
    """Return whether a species-like value represents a vacancy."""
    if isinstance(specie, str):
        return specie in VACANCY_LABELS
    symbol = getattr(specie, "symbol", None)
    return symbol in VACANCY_LABELS


def normalize_species(value: Any) -> Any:
    """Normalize strings to pymatgen species and vacancy labels to ``X``."""
    if is_vacancy_species(value):
        return vacancy_species()
    if isinstance(value, str):
        return Species(value)
    return value


def species_tokens(specie: Any) -> set[str]:
    """Return comparable string tokens for a species-like value."""
    if is_vacancy_species(specie):
        return {"X", "Vacancy"}
    tokens = {str(specie)}
    symbol = getattr(specie, "symbol", None)
    if symbol is not None:
        tokens.add(str(symbol))
    element = getattr(specie, "element", None)
    if element is not None:
        tokens.add(str(element))
    return tokens


def species_label(specie: Any) -> str:
    """Return a compact serialized label for a species-like value."""
    if is_vacancy_species(specie):
        return "X"
    if isinstance(specie, str):
        return specie
    symbol = getattr(specie, "symbol", None)
    if symbol is not None:
        return str(symbol)
    return str(specie)


def species_equivalent(left: Any, right: Any) -> bool:
    """Return whether two species-like values share any comparable token."""
    return bool(species_tokens(left).intersection(species_tokens(right)))
