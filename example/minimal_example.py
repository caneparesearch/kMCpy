#!/usr/bin/env python3
"""Minimal kMCpy run: Na diffusion in NASICON with a fitted LCE model.

Uses the bundled data in ``tests/files``; run from a source checkout::

    uv run python example/minimal_example.py
"""

from __future__ import annotations

from pathlib import Path

import kmcpy

DATA = Path(__file__).resolve().parent.parent / "tests" / "files"


def main(output_dir: str | Path = "example/output/minimal", kmc_passes: int = 100):
    # 1. The lattice: every site and what it may hold (Na or vacancy, Si or
    #    P; the rest is fixed), made into the simulated supercell.
    lattice = kmcpy.LatticeStructure.from_cif(
        DATA / "EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",
        site_mapping={"Na": ["Na", "X"], "Si": ["Si", "P"]},
        primitive=True,
    ).make_supercell((2, 1, 1))

    # 2. Plug in events, a rate model, and an initial state.
    simulation = kmcpy.Simulation(
        lattice,
        # Na1 <-> Na2 hops; the local environment (Na and Si neighbors within
        # 4 Angstrom) must match the one the LCE model was fitted on.
        events=kmcpy.HopEvents(
            cutoffs={("Na+", "Na+"): 4.0, ("Na+", "Si4+"): 4.0},
            labels=("Na1", "Na2"),
        ),
        model=DATA / "input" / "model.json",
        state=DATA / "input" / "initial_state.json",
        # 3. Run settings. The mobile ion's charge and hop length are taken
        #    from the structure and the events.
        temperature=298.0,
        attempt_frequency=5e12,
        equilibration_passes=1,
        kmc_passes=kmc_passes,
        random_seed=12345,
        name="MinimalExample",
    )

    tracker = simulation.run(output_dir=output_dir)
    time, msd, d_jump, d_tracer, conductivity, haven_ratio, f = tracker.return_current_info()
    print(f"Tracer diffusivity: {d_tracer:.3e} cm^2/s")
    print(f"Conductivity:       {conductivity:.3f} mS/cm")
    print(f"Results written to: {Path(output_dir).resolve()}")
    return tracker


if __name__ == "__main__":
    main()
