"""Mutable simulation state for kMC execution."""

from __future__ import annotations

import json
from typing import Any

from ..event.base import Event


class SimulationState:
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

    def apply_event(self, event: Event, dt: float) -> None:
        """Apply one event transition and advance simulation counters."""
        from_site, to_site = event.mobile_ion_indices
        self.occupations[from_site] *= -1
        self.occupations[to_site] *= -1
        self.time += dt
        self.step += 1

    def copy(self) -> "SimulationState":
        """Return a deep copy of mutable state."""
        return SimulationState(
            occupations=self.occupations,
            time=self.time,
            step=self.step,
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialize core mutable state to a plain dictionary."""
        return {
            "occupations": list(self.occupations),
            "time": float(self.time),
            "step": int(self.step),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationState":
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
            with open(filepath, "w") as fhandle:
                json.dump(data, fhandle, indent=2)
            return

        if filepath.endswith(".h5"):
            try:
                import h5py
            except ImportError as exc:
                raise ImportError("h5py required for HDF5 checkpoints") from exc

            with h5py.File(filepath, "w") as fhandle:
                for key, value in data.items():
                    fhandle.create_dataset(key, data=value)
            return

        raise ValueError("Checkpoint format must be .json or .h5")

    @classmethod
    def load_checkpoint(cls, filepath: str) -> "SimulationState":
        """Restore core mutable state from a checkpoint file (.json or .h5)."""
        if filepath.endswith(".json"):
            with open(filepath, "r") as fhandle:
                data = json.load(fhandle)
            return cls.from_dict(data)

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
        return f"SimulationState(time={self.time:.2e}, step={self.step}, n_sites={len(self.occupations)})"
