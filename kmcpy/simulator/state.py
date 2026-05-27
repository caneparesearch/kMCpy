"""Mutable simulation state for kMC execution."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from monty.json import MSONable
from monty.serialization import dumpfn, loadfn

from ..event.base import Event

logger = logging.getLogger(__name__)


class State(MSONable):
    """Mutable simulation state: occupations, simulation time, and step count."""

    def __init__(
        self,
        occupations: list[int],
        time: float = 0.0,
        step: int = 0,
    ):
        """Initialize state from occupations and optional time/step counters."""
        self.occupations = list(occupations)
        self.time = float(time)
        self.step = int(step)

    @classmethod
    def from_occupations(
        cls,
        occupations: Sequence[int],
        active_site_order=None,
        time: float = 0.0,
        step: int = 0,
    ) -> "State":
        """Create state from active-site or full-structure occupations."""
        values = list(occupations)
        if active_site_order is not None:
            values = active_site_order.select_active_values(values)
        return cls(occupations=values, time=time, step=step)

    @staticmethod
    def _initial_occupation_from_payload(
        occupation_raw_data: Sequence[int],
        supercell_shape: Sequence[int] | None = None,
        select_sites: Sequence[int] | None = None,
        active_site_order=None,
    ) -> list[int]:
        """
        Convert an initial-state ``occupation`` payload into active-site values.

        Initial-state files use site-state indices directly.
        """
        occupation_raw_data = np.array(occupation_raw_data)

        if active_site_order is not None:
            if len(occupation_raw_data) == active_site_order.active_site_count:
                occupation = occupation_raw_data
                selected = list(range(active_site_order.active_site_count))
            else:
                occupation = np.array(
                    active_site_order.select_active_values(occupation_raw_data.tolist())
                )
                selected = list(active_site_order.active_to_original)
        else:
            if select_sites is None:
                raise ValueError(
                    "select_sites or active_site_order is required to load occupations"
                )
            if supercell_shape is None:
                raise ValueError(
                    "supercell_shape is required when loading occupations with select_sites"
                )

            supercell_shape = tuple(int(value) for value in supercell_shape)
            if len(supercell_shape) != 3:
                raise ValueError("supercell_shape must have three components")

            supercell_size = supercell_shape[0] * supercell_shape[1] * supercell_shape[2]
            if len(occupation_raw_data) % supercell_size != 0:
                raise ValueError(
                    "The length of occupation data "
                    f"{len(occupation_raw_data)} is incompatible with the supercell shape"
                )

            site_nums = int(len(occupation_raw_data) / supercell_size)
            convert_to_dimension = (site_nums, *supercell_shape)
            occupation = occupation_raw_data.reshape(convert_to_dimension)[
                list(select_sites)
            ].flatten("C")
            selected = list(select_sites)

        logger.debug("Selected active sites are %s", selected)
        logger.debug(
            "Occupation in compact active-site basis: %s", occupation
        )

        return occupation.tolist()

    @classmethod
    def from_file(
        cls,
        filepath: str,
        supercell_shape: Sequence[int] | None = None,
        select_sites: Sequence[int] | None = None,
        active_site_order=None,
        time: float = 0.0,
        step: int = 0,
    ) -> "State":
        """Load state from a checkpoint or initial-state JSON file."""
        suffix = Path(filepath).suffix.lower()
        if suffix == ".h5":
            return cls.load_checkpoint(filepath)
        if suffix != ".json":
            raise ValueError("State file format must be .json or .h5")

        payload = loadfn(filepath, cls=None)
        if not isinstance(payload, dict):
            raise ValueError("State JSON file must contain an object")

        if "occupations" in payload:
            return cls.from_occupations(
                payload["occupations"],
                active_site_order=active_site_order,
                time=payload.get("time", time),
                step=payload.get("step", step),
            )

        if "occupation" in payload:
            occupations = cls._initial_occupation_from_payload(
                payload["occupation"],
                supercell_shape=supercell_shape,
                select_sites=select_sites,
                active_site_order=active_site_order,
            )
            return cls(occupations=occupations, time=time, step=step)

        raise ValueError(
            "State JSON file must contain either 'occupations' or initial-state 'occupation'"
        )

    def apply_event(self, event: Event, dt: float) -> None:
        """Apply one event transition and advance simulation counters."""
        from_site, to_site = event.mobile_ion_indices
        self.occupations[from_site], self.occupations[to_site] = (
            self.occupations[to_site],
            self.occupations[from_site],
        )
        self.time += dt
        self.step += 1

    def copy(self) -> "State":
        """Return a deep copy of mutable state."""
        return State(
            occupations=self.occupations,
            time=self.time,
            step=self.step,
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialize core mutable state to a plain dictionary."""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "occupations": list(self.occupations),
            "time": float(self.time),
            "step": int(self.step),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "State":
        """Create state from dictionary payload."""
        return cls(
            occupations=data["occupations"],
            time=data["time"],
            step=data["step"],
        )

    def save_checkpoint(self, filepath: str) -> None:
        """Persist core mutable state to a checkpoint file (.json or .h5)."""
        data = self.as_dict()

        if filepath.endswith(".json"):
            dumpfn(data, filepath, indent=2)
            return

        if filepath.endswith(".h5"):
            try:
                import h5py
            except ImportError as exc:
                raise ImportError("h5py required for HDF5 checkpoints") from exc

            with h5py.File(filepath, "w") as fhandle:
                for key, value in data.items():
                    if key.startswith("@"):
                        continue
                    fhandle.create_dataset(key, data=value)
            return

        raise ValueError("Checkpoint format must be .json or .h5")

    @classmethod
    def load_checkpoint(cls, filepath: str) -> "State":
        """Restore core mutable state from a checkpoint file (.json or .h5)."""
        if filepath.endswith(".json"):
            return cls.from_file(filepath)

        if filepath.endswith(".h5"):
            try:
                import h5py
            except ImportError as exc:
                raise ImportError("h5py required for HDF5 checkpoints") from exc

            with h5py.File(filepath, "r") as fhandle:
                data = {
                    "occupations": fhandle["occupations"][()].tolist(),
                    "time": float(fhandle["time"][()]),
                    "step": int(fhandle["step"][()]),
                }
            return cls.from_dict(data)

        raise ValueError("Checkpoint format must be .json or .h5")

    def __repr__(self) -> str:
        """Return compact debug representation of simulation state."""
        return f"State(time={self.time:.2e}, step={self.step}, n_sites={len(self.occupations)})"
