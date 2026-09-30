"""Numba kernels for local cluster expansion correlations and composite rates.

Cluster indices are passed as flat CSR-style arrays built by
:func:`flatten_cluster_indices`, because passing nested ``numba.typed.List``
objects into ``njit`` functions costs more than the calculation itself.
:class:`LCEKernelInputs` bundles one model's arrays for the batched composite
rate kernel.
"""

from __future__ import annotations

from typing import NamedTuple

import numba as nb
import numpy as np


class LCEKernelInputs(NamedTuple):
    """Arrays describing one ``LocalClusterExpansion`` for the rate kernel."""

    decorated: bool
    correlation_count: int
    orbit_offsets: np.ndarray
    cluster_offsets: np.ndarray
    sites: np.ndarray
    basis: np.ndarray
    site_basis_values: np.ndarray
    keci: np.ndarray
    empty_cluster: float

    @classmethod
    def empty(cls) -> "LCEKernelInputs":
        """Placeholder inputs for an absent submodel (evaluates to zero terms)."""
        return cls(
            False,
            0,
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            EMPTY_SITE_BASIS_VALUES,
            np.zeros(0, dtype=np.float64),
            0.0,
        )


# Placeholder ``site_basis_values`` for undecorated (plain) models.
EMPTY_SITE_BASIS_VALUES = np.zeros((1, 1, 1), dtype=np.float64)


def flatten_cluster_indices(cluster_site_indices, cluster_basis_indices=None):
    """Flatten nested ``[orbit][cluster][site]`` indices into CSR-style arrays.

    Returns ``(orbit_offsets, cluster_offsets, sites, basis)``. Clusters of
    orbit ``i`` are ``orbit_offsets[i]:orbit_offsets[i + 1]`` and sites of
    cluster ``j`` are ``sites[cluster_offsets[j]:cluster_offsets[j + 1]]``.
    ``basis`` is aligned with ``sites`` (all zeros when no basis indices are
    given).
    """
    orbit_offsets = [0]
    cluster_offsets = [0]
    sites = []
    basis = []
    for orbit_index, orbit in enumerate(cluster_site_indices):
        basis_orbit = (
            cluster_basis_indices[orbit_index]
            if cluster_basis_indices is not None
            else None
        )
        for cluster_index, cluster in enumerate(orbit):
            basis_cluster = (
                basis_orbit[cluster_index] if basis_orbit is not None else None
            )
            for site_position, site_index in enumerate(cluster):
                sites.append(int(site_index))
                basis.append(
                    int(basis_cluster[site_position])
                    if basis_cluster is not None
                    else 0
                )
            cluster_offsets.append(len(sites))
        orbit_offsets.append(len(cluster_offsets) - 1)
    return (
        np.asarray(orbit_offsets, dtype=np.int64),
        np.asarray(cluster_offsets, dtype=np.int64),
        np.asarray(sites, dtype=np.int64),
        np.asarray(basis, dtype=np.int64),
    )


@nb.njit
def correlation(corr, occ_latt, orbit_offsets, cluster_offsets, sites):
    """
    Calculate correlation function for cluster expansion.
    
    Args:
        corr: Output correlation array
        occ_latt: Occupation array for the lattice
        orbit_offsets, cluster_offsets, sites: Flat cluster indices from
            ``flatten_cluster_indices``
    """
    for i in range(len(orbit_offsets) - 1): # loop through orbits
        corr[i] = 0
        for cluster in range(orbit_offsets[i], orbit_offsets[i + 1]): # loop through clusters in the orbit
            corr_cluster = 1
            for position in range(cluster_offsets[cluster], cluster_offsets[cluster + 1]):
                corr_cluster *= occ_latt[sites[position]]
            corr[i] += corr_cluster


@nb.njit
def decorated_correlation(
    corr,
    occ_latt,
    orbit_offsets,
    cluster_offsets,
    sites,
    basis,
    site_basis_values,
):
    """
    Calculate decorated multicomponent correlation functions.

    ``occ_latt`` stores species-state indices. ``site_basis_values`` maps
    ``[local_site, state_index, basis_index]`` to the scalar basis value.
    """
    for i in range(len(orbit_offsets) - 1):
        corr[i] = 0.0
        for cluster in range(orbit_offsets[i], orbit_offsets[i + 1]):
            corr_cluster = 1.0
            for position in range(cluster_offsets[cluster], cluster_offsets[cluster + 1]):
                occ_site = sites[position]
                state_index = int(occ_latt[occ_site])
                basis_index = int(basis[position])
                corr_cluster *= site_basis_values[occ_site, state_index, basis_index]
            corr[i] += corr_cluster


@nb.njit
def _lce_value(
    local_occupation,
    decorated,
    correlation_count,
    orbit_offsets,
    cluster_offsets,
    sites,
    basis,
    site_basis_values,
    keci,
    empty_cluster,
):
    corr = np.empty(correlation_count)
    if decorated:
        decorated_correlation(
            corr,
            local_occupation,
            orbit_offsets,
            cluster_offsets,
            sites,
            basis,
            site_basis_values,
        )
    else:
        correlation(corr, local_occupation, orbit_offsets, cluster_offsets, sites)
    return np.dot(corr, keci) + empty_cluster


@nb.njit
def composite_lce_rates(
    rates,
    event_indices,
    occupations,
    from_sites,
    to_sites,
    hop_codes,
    env_offsets,
    env_sites,
    kra_decorated,
    kra_correlation_count,
    kra_orbit_offsets,
    kra_cluster_offsets,
    kra_sites,
    kra_basis,
    kra_site_basis_values,
    kra_keci,
    kra_empty_cluster,
    has_site_model,
    site_decorated,
    site_correlation_count,
    site_orbit_offsets,
    site_cluster_offsets,
    site_sites,
    site_basis,
    site_site_basis_values,
    site_keci,
    site_empty_cluster,
    attempt_frequency,
    k_times_temperature,
):
    """Evaluate ``CompositeLCEModel.compute_probability`` for many events.

    The arithmetic mirrors the scalar path operation by operation so that
    batched and per-event rates are identical.
    """
    for position in range(len(event_indices)):
        event_index = event_indices[position]
        from_occ = occupations[from_sites[event_index]]
        to_occ = occupations[to_sites[event_index]]
        if from_occ == hop_codes[event_index, 0] and to_occ == hop_codes[event_index, 1]:
            direction = 1
        elif from_occ == hop_codes[event_index, 2] and to_occ == hop_codes[event_index, 3]:
            direction = -1
        else:
            rates[position] = 0.0
            continue

        start = env_offsets[event_index]
        stop = env_offsets[event_index + 1]
        local_occupation = np.empty(stop - start, dtype=np.int64)
        for offset in range(stop - start):
            local_occupation[offset] = occupations[env_sites[start + offset]]

        e_kra = _lce_value(
            local_occupation,
            kra_decorated,
            kra_correlation_count,
            kra_orbit_offsets,
            kra_cluster_offsets,
            kra_sites,
            kra_basis,
            kra_site_basis_values,
            kra_keci,
            kra_empty_cluster,
        )
        delta_e_site = 0.0
        if has_site_model:
            delta_e_site = direction * _lce_value(
                local_occupation,
                site_decorated,
                site_correlation_count,
                site_orbit_offsets,
                site_cluster_offsets,
                site_sites,
                site_basis,
                site_site_basis_values,
                site_keci,
                site_empty_cluster,
            )
        e_barrier = e_kra + delta_e_site / 2
        rates[position] = attempt_frequency * np.exp(-e_barrier / k_times_temperature)
