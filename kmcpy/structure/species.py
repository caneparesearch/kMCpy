"""Species normalization and ``site_mapping`` handling for kMCpy structure code.

This module is the single owner of what counts as a vacancy and of how a
``site_mapping`` (template species -> allowed species) is normalized and
matched to template sites.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from pymatgen.core import DummySpecies, Species


# Compared case-insensitively, e.g. "X", "Va", "VA", "vacancy", "VACANCY".
VACANCY_LABELS = frozenset({"x", "va", "vacancy"})


def vacancy_species() -> DummySpecies:
    """Return the internal pymatgen species used for vacancy sites."""
    return DummySpecies("X", 0)


def is_vacancy_species(specie: Any) -> bool:
    """Return whether a species-like value represents a vacancy."""
    if isinstance(specie, str):
        return specie.lower() in VACANCY_LABELS
    symbol = getattr(specie, "symbol", None)
    return isinstance(symbol, str) and symbol.lower() in VACANCY_LABELS


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


class SiteMapping:
    """Normalized ``site_mapping``: template species -> allowed species.

    The allowed species of a template site are its occupation states, in
    order. A site with one allowed species is fixed; a site whose allowed
    species include a vacancy hosts a mobile species.

    This is the physical species mapping used by ``LatticeStructure``,
    ``ActiveSiteOrder``, and ``EventGenerator``. It is unrelated to the
    external-site ``site_mapping`` of ``SiteEnergyModel``.
    """

    def __init__(self, site_mapping: "Mapping[Any, Any] | SiteMapping"):
        if isinstance(site_mapping, SiteMapping):
            self._entries = site_mapping._entries
            return
        entries = []
        for key, value in site_mapping.items():
            values = value if isinstance(value, (list, tuple)) else [value]
            entries.append(
                (
                    str(key),
                    normalize_species(key),
                    tuple(normalize_species(item) for item in values),
                )
            )
        self._entries = tuple(entries)

    def as_dict(self) -> dict[Any, list[Any]]:
        """Return ``{normalized template species: [normalized allowed species]}``."""
        return {species: list(allowed) for _, species, allowed in self._entries}

    def allowed_species_for(self, specie: Any) -> tuple[Any, ...] | None:
        """Return the allowed species of the first entry matching ``specie``."""
        for _, mapped_species, allowed in self._entries:
            if species_equivalent(specie, mapped_species):
                return allowed
        return None

    def allowed_species_by_site(
        self,
        structure: Sequence[Any],
        strict: bool = True,
    ) -> list[tuple[Any, ...] | None]:
        """Return the allowed species for every site of ``structure``.

        With ``strict=True`` a site without a matching entry raises
        ``ValueError``; otherwise its entry is ``None``.
        """
        allowed_species = []
        for index, site in enumerate(structure):
            allowed = self.allowed_species_for(site.specie)
            if allowed is None and strict:
                raise ValueError(
                    "No site_mapping entry found for template site "
                    f"{index} with species {site.species_string}."
                )
            allowed_species.append(allowed)
        return allowed_species

    def mobile_species(self) -> list[str]:
        """Return template species labels whose allowed states include a vacancy."""
        return [
            label
            for label, _, allowed in self._entries
            if any(is_vacancy_species(specie) for specie in allowed)
        ]

