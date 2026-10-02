#!/usr/bin/env python3
"""NASICON tutorial: plug different rate models and initial states into one setup.

Uses the bundled data in ``tests/files``; run from a source checkout::

    uv run python example/tutorial_nasicon.py                    # fitted LCE model
    uv run python example/tutorial_nasicon.py --model constant   # constant barrier
    uv run python example/tutorial_nasicon.py --na-fraction 0.5  # random start
"""

from __future__ import annotations

import argparse
from pathlib import Path

import kmcpy

DATA = Path(__file__).resolve().parent.parent / "tests" / "files"
SITE_MAPPING = {"Na": ["Na", "X"], "Si": ["Si", "P"]}


def build_model(name: str):
    """The model slot: any rate model fits; only this function changes."""
    if name == "lce":
        # Fitted KRA + site-energy local cluster expansions.
        return DATA / "input" / "model.json"
    if name == "constant":
        # Every Na hop has the same 300 meV barrier.
        return kmcpy.LocalBarrierModel.constant_barrier(300.0)
    raise ValueError(f"Unknown model {name!r}")


def build_state(na_fraction: float | None, seed: int):
    """The state slot: a stored initial state or a random occupation."""
    if na_fraction is None:
        return DATA / "input" / "initial_state.json"
    return kmcpy.RandomOccupation({"Na": na_fraction}, seed=seed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=["lce", "constant"], default="lce")
    parser.add_argument("--na-fraction", type=float, default=None,
                        help="Start from a random occupation with this Na fraction.")
    parser.add_argument("--temperature", type=float, default=298.0, help="K")
    parser.add_argument("--kmc-passes", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument("--output-dir", default="example/output/tutorial")
    args = parser.parse_args(argv)

    lattice = kmcpy.LatticeStructure.from_cif(
        DATA / "EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",
        site_mapping=SITE_MAPPING,
        primitive=True,
    ).make_supercell((2, 1, 1))
    print(f"{lattice.n_active_sites} active sites, mobile species: {lattice.mobile_species}")

    simulation = kmcpy.Simulation(
        lattice,
        events=kmcpy.HopEvents(
            cutoffs={("Na+", "Na+"): 4.0, ("Na+", "Si4+"): 4.0},
            labels=("Na1", "Na2"),
        ),
        model=build_model(args.model),
        state=build_state(args.na_fraction, args.random_seed),
        temperature=args.temperature,
        attempt_frequency=5e12,
        equilibration_passes=1,
        kmc_passes=args.kmc_passes,
        random_seed=args.random_seed,
        name=f"NASICON_{args.model}",
    )

    # Optional: record an extra property during the run.
    allowed = lattice.active_site_order.allowed_species_by_active_site
    na_sites = [site for site, species in enumerate(allowed) if "X" in species]

    def na_fraction(state, step, sim_time):
        return sum(state.occupations[site] == 0 for site in na_sites) / len(na_sites)

    simulation.attach(na_fraction, interval=50, name="na_fraction")

    tracker = simulation.run(output_dir=args.output_dir)
    time, msd, d_jump, d_tracer, conductivity, haven_ratio, f = tracker.return_current_info()
    print(f"Model: {args.model}")
    print(f"Tracer diffusivity: {d_tracer:.3e} cm^2/s")
    print(f"Conductivity:       {conductivity:.3f} mS/cm")
    print(f"Na fraction:        {tracker.get_property_records('na_fraction')[-1]['value']:.3f}")
    print(f"Results written to: {Path(args.output_dir).resolve()}")
    return tracker


if __name__ == "__main__":
    main()
