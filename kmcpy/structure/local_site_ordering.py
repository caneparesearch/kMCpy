"""Ordering conventions for local-environment occupation vectors."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Sequence


BUILTIN_ORDERING_CONVENTIONS = {
    "kmcpy_default": {
        "sort_keys": ("species_string",),
        "exclude_center_site": False,
    },
    "nasicon_nat_commun_2022": {
        "sort_keys": ("species", "cartesian_x"),
        "exclude_center_site": True,
    },
}

LEGACY_ORDERING_CONVENTION_NAMES = {
    "kmcpy_default_v1": "kmcpy_default",
    "nasicon_publication_v1": "nasicon_nat_commun_2022",
}


@dataclass(frozen=True)
class LocalSiteOrderingConvention:
    """Rules that define the order of sites in a local occupation vector."""

    name: str
    sort_keys: tuple[str, ...] = ("species_string",)
    exclude_center_site: bool = False
    center_match_tolerance: float = 1e-3

    @classmethod
    def from_name(cls, name: str) -> "LocalSiteOrderingConvention":
        """Create a built-in ordering convention by name."""
        canonical_name = LEGACY_ORDERING_CONVENTION_NAMES.get(name, name)
        convention = BUILTIN_ORDERING_CONVENTIONS.get(canonical_name)
        if convention is not None:
            return cls(
                name=canonical_name,
                sort_keys=convention["sort_keys"],
                exclude_center_site=convention["exclude_center_site"],
            )
        available = sorted(BUILTIN_ORDERING_CONVENTIONS)
        raise ValueError(
            f"Unknown local site ordering convention '{name}'. "
            f"Available: {available}"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LocalSiteOrderingConvention":
        """Create an ordering convention from serialized metadata."""
        if "name" not in data:
            raise ValueError("Local site ordering convention requires a 'name'")
        name = str(data["name"])
        try:
            base = cls.from_name(name)
            resolved_name = base.name
        except ValueError:
            base = cls(name=name)
            resolved_name = name
        return cls(
            name=resolved_name,
            sort_keys=tuple(data.get("sort_keys", base.sort_keys)),
            exclude_center_site=bool(
                data.get("exclude_center_site", base.exclude_center_site)
            ),
            center_match_tolerance=float(
                data.get("center_match_tolerance", base.center_match_tolerance)
            ),
        )

    @classmethod
    def resolve(
        cls,
        convention: str | dict[str, Any] | "LocalSiteOrderingConvention" | None,
    ) -> "LocalSiteOrderingConvention":
        """Normalize user input into an ordering convention."""
        if convention is None:
            return cls.from_name("kmcpy_default")
        if isinstance(convention, cls):
            return convention
        if isinstance(convention, str):
            return cls.from_name(convention)
        if isinstance(convention, dict):
            return cls.from_dict(convention)
        raise TypeError(
            "ordering_convention must be None, a name, a dict, "
            "or LocalSiteOrderingConvention"
        )

    def with_exclude_center_site(
        self, exclude_center_site: bool
    ) -> "LocalSiteOrderingConvention":
        """Return a copy with an explicit center-site exclusion policy."""
        return LocalSiteOrderingConvention(
            name=self.name,
            sort_keys=self.sort_keys,
            exclude_center_site=exclude_center_site,
            center_match_tolerance=self.center_match_tolerance,
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialize the ordering convention."""
        return {
            "name": self.name,
            "sort_keys": list(self.sort_keys),
            "exclude_center_site": self.exclude_center_site,
            "center_match_tolerance": self.center_match_tolerance,
        }

    def sort_local_env_sites(self, local_env_sites: list[Any]) -> list[Any]:
        """Sort ``get_sites_in_sphere`` results according to this convention."""
        return sorted(local_env_sites, key=lambda item: self._sort_key(item[0]))

    def _sort_key(self, site: Any) -> tuple[Any, ...]:
        values: list[Any] = []
        for key in self.sort_keys:
            if key == "species_string":
                values.append(site.species_string)
            elif key == "species":
                values.append(site.specie)
            elif key == "cartesian_x":
                values.append(float(site.coords[0]))
            elif key == "cartesian_y":
                values.append(float(site.coords[1]))
            elif key == "cartesian_z":
                values.append(float(site.coords[2]))
            else:
                raise ValueError(f"Unsupported local site ordering sort key: {key}")
        return tuple(values)


def ordered_site_signature(
    sites: Iterable[Any], *, decimals: int = 8
) -> list[dict[str, Any]]:
    """Return an order-sensitive, JSON-safe signature for local sites."""
    signature = []
    for site in sites:
        signature.append(
            {
                "species": site.species_string,
                "cartesian_coords": [
                    round(float(coord), decimals) for coord in site.coords
                ],
            }
        )
    return signature


def ordered_site_hash(signature: Sequence[dict[str, Any]]) -> str:
    """Hash an ordered local-site signature for compatibility checks."""
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
