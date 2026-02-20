"""Built-in transport property calculations for kMC trajectories."""

from __future__ import annotations

from typing import Mapping

import numpy as np

BUILTIN_PROPERTY_FIELDS = (
    "msd",
    "jump_diffusivity",
    "tracer_diffusivity",
    "conductivity",
    "havens_ratio",
    "correlation_factor",
)



def _is_enabled(enabled: Mapping[str, bool] | None, name: str) -> bool:
    """Return True when a metric is enabled under the provided toggle map."""
    if enabled is None:
        return True
    return bool(enabled.get(name, True))



def compute_transport_properties(
    displacement: np.ndarray,
    hop_counter: np.ndarray,
    *,
    sim_time: float,
    dimension: int,
    n_mobile_ion_specie: int,
    elementary_hop_distance: float,
    volume: float,
    mobile_ion_charge: float,
    temperature: float,
    enabled: Mapping[str, bool] | None = None,
) -> dict[str, float]:
    """Compute built-in transport properties from trajectory state."""
    nan = float("nan")

    displacement_norm_sq = np.linalg.norm(displacement, axis=1) ** 2
    msd_internal = float(np.mean(displacement_norm_sq))

    if sim_time > 0 and n_mobile_ion_specie > 0:
        displacement_vector_tot = np.linalg.norm(np.sum(displacement, axis=0))
        jump_diffusivity_internal = (
            displacement_vector_tot**2
            / (2 * dimension * sim_time * n_mobile_ion_specie)
            * 10 ** (-16)
        )
        tracer_diffusivity_internal = (
            msd_internal
            / (2 * dimension * sim_time)
            * 10 ** (-16)
        )
    else:
        jump_diffusivity_internal = nan
        tracer_diffusivity_internal = nan

    conductivity_internal = nan
    if np.isfinite(jump_diffusivity_internal):
        k = 8.617333262145 * 10 ** (-2)  # meV/K
        n_carrier = n_mobile_ion_specie / volume
        conductivity_internal = (
            jump_diffusivity_internal
            * n_carrier
            * mobile_ion_charge**2
            / (k * temperature)
            * 1.602
            * 10**11
        )

    havens_ratio_internal = nan
    if (
        np.isfinite(jump_diffusivity_internal)
        and np.isfinite(tracer_diffusivity_internal)
        and jump_diffusivity_internal != 0
    ):
        havens_ratio_internal = tracer_diffusivity_internal / jump_diffusivity_internal

    hop_counter_safe = np.where(hop_counter == 0, 1, hop_counter)
    correlation_factor_internal = displacement_norm_sq / (
        hop_counter_safe * elementary_hop_distance**2
    )
    correlation_factor_internal[hop_counter == 0] = 0
    correlation_factor_internal = float(np.mean(correlation_factor_internal))

    return {
        "msd": msd_internal if _is_enabled(enabled, "msd") else nan,
        "jump_diffusivity": (
            jump_diffusivity_internal if _is_enabled(enabled, "jump_diffusivity") else nan
        ),
        "tracer_diffusivity": (
            tracer_diffusivity_internal if _is_enabled(enabled, "tracer_diffusivity") else nan
        ),
        "conductivity": (
            conductivity_internal if _is_enabled(enabled, "conductivity") else nan
        ),
        "havens_ratio": (
            havens_ratio_internal if _is_enabled(enabled, "havens_ratio") else nan
        ),
        "correlation_factor": (
            correlation_factor_internal if _is_enabled(enabled, "correlation_factor") else nan
        ),
    }
