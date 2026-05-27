"""Ordering rules for local-environment occupation vectors.

``LocalSiteOrder`` defines only how already selected local sites are ordered and
whether a real center site is included in the occupation vector. The center
position itself is chosen by ``LocalLatticeStructure`` or the local-environment
enumeration helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Sequence

from monty.json import MSONable


BUILTIN_LOCAL_SITE_ORDERS = {
    "kmcpy_default": {
        "sort_keys": ("species_string",),
        "exclude_center_site": False,
    },
    "nasicon_nat_commun_2022": {
        "sort_keys": ("species", "cartesian_x"),
        "exclude_center_site": True,
    },
}

@dataclass(frozen=True)
class LocalSiteOrder(MSONable):
    """Rules that define a local occupation-vector sequence.

    This object is deliberately narrow: it does not choose the local-environment
    center and it does not decide whether the center is a real atom or an
    abstract coordinate. It only controls:

    - sort keys for sites returned by the local cutoff search;
    - whether a site matching the supplied center is removed from the vector.

    If the center is an active-site index, ``exclude_center_site=True`` removes
    that exact site. If the center is a fractional coordinate, it removes a real
    site only when one lies within ``center_match_tolerance`` of that coordinate.
    """

    name: str
    sort_keys: tuple[str, ...] = ("species_string",)
    exclude_center_site: bool = False
    center_match_tolerance: float = 1e-3

    @classmethod
    def from_name(cls, name: str) -> "LocalSiteOrder":
        """Create a built-in local site order by name."""
        order = BUILTIN_LOCAL_SITE_ORDERS.get(name)
        if order is not None:
            return cls(
                name=name,
                sort_keys=order["sort_keys"],
                exclude_center_site=order["exclude_center_site"],
            )
        available = sorted(BUILTIN_LOCAL_SITE_ORDERS)
        raise ValueError(
            f"Unknown local site order '{name}'. "
            f"Available: {available}"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LocalSiteOrder":
        """Create a local site order from serialized metadata."""
        if "name" not in data:
            raise ValueError("Local site order requires a 'name'")
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
        order: str | dict[str, Any] | "LocalSiteOrder" | None,
    ) -> "LocalSiteOrder":
        """Normalize user input into a local site order."""
        if order is None:
            return cls.from_name("kmcpy_default")
        if isinstance(order, cls):
            return order
        if isinstance(order, str):
            return cls.from_name(order)
        if isinstance(order, dict):
            return cls.from_dict(order)
        raise TypeError(
            "local_site_order must be None, a name, a dict, "
            "or LocalSiteOrder"
        )

    def with_exclude_center_site(
        self, exclude_center_site: bool
    ) -> "LocalSiteOrder":
        """Return a copy with an explicit center-site exclusion policy."""
        return LocalSiteOrder(
            name=self.name,
            sort_keys=self.sort_keys,
            exclude_center_site=exclude_center_site,
            center_match_tolerance=self.center_match_tolerance,
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialize the local site order."""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "name": self.name,
            "sort_keys": list(self.sort_keys),
            "exclude_center_site": self.exclude_center_site,
            "center_match_tolerance": self.center_match_tolerance,
        }

    def sort_local_env_sites(self, local_env_sites: list[Any]) -> list[Any]:
        """Sort ``get_sites_in_sphere`` results according to this order."""
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
                raise ValueError(f"Unsupported local site order sort key: {key}")
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
    """Hash an ordered local-site signature for consistency checks."""
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
