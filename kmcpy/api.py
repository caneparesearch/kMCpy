"""Public high-level API helpers for kMCpy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kmcpy.simulator.config import Configuration
    from kmcpy.simulator.tracker import Tracker


def run(config: "Configuration", label: str | None = None) -> "Tracker":
    """Run a KMC simulation from a ``Configuration`` in one call.

    Args:
        config: Simulation configuration.
        label: Optional output label. If ``None``, ``config.name`` is used.

    Returns:
        Tracker with simulation results.
    """
    from kmcpy.simulator.config import Configuration
    from kmcpy.simulator.kmc import KMC

    if not isinstance(config, Configuration):
        raise TypeError("config must be a Configuration/SimulationConfig instance")

    kmc = KMC.from_config(config)
    return kmc.run(config=config, label=label)
