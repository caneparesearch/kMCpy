"""Public high-level API helpers for kMCpy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kmcpy.simulator.config import SimulationConfig
    from kmcpy.simulator.tracker import Tracker


def run(config: "SimulationConfig", label: str | None = None) -> "Tracker":
    """Run a KMC simulation from a ``SimulationConfig`` in one call.

    Args:
        config: Simulation configuration.
        label: Optional output label. If ``None``, ``config.name`` is used.

    Returns:
        Tracker with simulation results.
    """
    from kmcpy.simulator.config import SimulationConfig
    from kmcpy.simulator.kmc import KMC

    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig instance")

    kmc = KMC.from_config(config)
    return kmc.run(config=config, label=label)
