"""Unit conventions and conversion constants used by kMCpy.

kMCpy keeps the public API numeric and lightweight, but the numeric values have
fixed units. This module makes those conventions explicit so calculations,
output metadata, and documentation all refer to the same source.
"""

from __future__ import annotations


# Energy and rate conventions.
BOLTZMANN_CONSTANT_MEV_PER_K: float = 8.617333262145e-2


# Length and volume conventions.
ANGSTROM_TO_CM: float = 1.0e-8
ANGSTROM_SQUARED_TO_CM_SQUARED: float = ANGSTROM_TO_CM**2
ANGSTROM_CUBED_TO_CM_CUBED: float = ANGSTROM_TO_CM**3


# Conductivity convention.
#
# compute_transport_properties uses:
#   D_J in cm^2/s
#   carrier concentration in 1/Angstrom^3
#   mobile ion charge in units of |e|
#   k_B T in meV
# This factor converts the Nernst-Einstein expression to mS/cm. The 1.602
# coefficient preserves the historical kMCpy convention used by existing tests
# and reference outputs.
CONDUCTIVITY_MS_PER_CM_FACTOR: float = 1.602e11


UNIT_CONVENTIONS: dict[str, str] = {
    "barrier": "meV",
    "energy": "meV",
    "e_kra": "meV",
    "e_site": "meV",
    "site_energy": "meV",
    "probability": "Hz",
    "rate": "Hz",
    "event_rate": "Hz",
    "attempt_frequency": "Hz",
    "temperature": "K",
    "time": "s",
    "sim_time": "s",
    "property_sampling_time_interval": "s",
    "length": "Angstrom",
    "elementary_hop_distance": "Angstrom",
    "displacement": "Angstrom",
    "volume": "Angstrom^3",
    "carrier_concentration": "1/Angstrom^3",
    "mobile_ion_charge": "|e|",
    "msd": "Angstrom^2",
    "jump_diffusivity": "cm^2/s",
    "tracer_diffusivity": "cm^2/s",
    "conductivity": "mS/cm",
    "havens_ratio": "dimensionless",
    "correlation_factor": "dimensionless",
    "dimension": "dimensionless",
    "occupation": "dimensionless",
}


TRANSPORT_PROPERTY_UNITS: dict[str, str] = {
    "time": UNIT_CONVENTIONS["time"],
    "msd": UNIT_CONVENTIONS["msd"],
    "jump_diffusivity": UNIT_CONVENTIONS["jump_diffusivity"],
    "tracer_diffusivity": UNIT_CONVENTIONS["tracer_diffusivity"],
    "conductivity": UNIT_CONVENTIONS["conductivity"],
    "havens_ratio": UNIT_CONVENTIONS["havens_ratio"],
    "correlation_factor": UNIT_CONVENTIONS["correlation_factor"],
}


MODEL_PROPERTY_UNITS: dict[str, str] = {
    "barrier": UNIT_CONVENTIONS["barrier"],
    "probability": UNIT_CONVENTIONS["probability"],
}


def unit_for(quantity: str, default: str | None = None) -> str | None:
    """Return the configured unit label for a named quantity."""
    return UNIT_CONVENTIONS.get(quantity, default)
