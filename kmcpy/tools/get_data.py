#!/usr/bin/env python
"""Post-process migration outputs and summarize transport properties."""

import glob

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from kmcpy.simulator.built_in_properties import compute_transport_properties


results = glob.glob("comp_0.1_1")



def compute_chemical_diffusivity(jump_diffusivity: float, theta: float) -> float:
    return jump_diffusivity * theta



def fit_theta():
    x, y = np.loadtxt("theta_575.txt")  # x, theta
    from scipy.interpolate import interp1d

    return interp1d(x, y, kind="cubic")



def calc_transport_properties(
    displacement,
    hop_counter,
    n_mobile_ion,
    sim_time,
    *,
    dimension=3,
    elementary_hop_distance=3.47782,
    volume=1.0,
    charge=1,
    temperature=300,
):
    """Calculate transport metrics from trajectory arrays using shared core logic."""
    return compute_transport_properties(
        np.asarray(displacement),
        np.asarray(hop_counter),
        sim_time=sim_time,
        dimension=dimension,
        n_mobile_ion_specie=n_mobile_ion,
        elementary_hop_distance=elementary_hop_distance,
        volume=volume,
        mobile_ion_charge=charge,
        temperature=temperature,
    )


spl = fit_theta()
data = []
for result in results:
    for n_pass in np.arange(0, 2000, 1, dtype="int"):
        print(result)
        x_value = 3 - 3 * float(result.strip().split("_")[1])
        structure_id = int(result.strip().split("_")[2])
        jump_diffusivity, tracer_diffusivity, correlation_factor, conductivity = np.loadtxt(
            result + "/migration_" + str(n_pass) + ".txt"
        )

        theta = spl(x_value)
        chemical_diffusivity = compute_chemical_diffusivity(jump_diffusivity, theta)
        print("D_c = ", chemical_diffusivity)

        data.append(
            [
                x_value,
                n_pass,
                structure_id,
                jump_diffusivity,
                tracer_diffusivity,
                chemical_diffusivity,
                theta,
                correlation_factor,
                conductivity,
                result,
            ]
        )


if __name__ == "__main__":
    df = pd.DataFrame(
        data,
        columns=[
            "x",
            "n_pass",
            "structure_id",
            "jump_diffusivity",
            "tracer_diffusivity",
            "chemical_diffusivity",
            "theta",
            "correlation_factor",
            "conductivity",
            "location",
        ],
    ).sort_values(by=["x", "n_pass"])

    df.to_csv("data_all.csv")
    df2 = df.groupby("x").mean().reset_index().sort_values(by="x")
    print(df2)
    df2.to_csv("data.csv")

    df2 = pd.read_csv("data.csv")
    fig, axes = plt.subplots(2, 2, figsize=(5, 4), sharex="col")
    axes[0, 0].plot(df2.x, df2.jump_diffusivity, ".")
    axes[0, 0].set_ylabel(r"$D_{J}$ (cm$^2$/s)")
    axes[0, 0].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2e"))

    axes[0, 1].plot(df2.x, df2.tracer_diffusivity, ".")
    axes[0, 1].set_ylabel(r"$D^{*}$ (cm$^2$/s)")
    axes[0, 1].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2e"))

    axes[1, 0].plot(df2.x, df2.correlation_factor, ".")
    axes[1, 0].set_ylabel(r"Correlation factor")
    axes[1, 0].set_xlabel(
        r"${\rm x}$ in Na$_{\rm 1+x}$Zr$_{\rm 2}$Si$_{\rm x}$P$_{\rm 3-x}$O$_{\rm 12}$"
    )
    axes[1, 0].set_ylim((0, 1))
    axes[1, 0].set_xlim((0, 3))

    axes[1, 1].plot(df2.x, df2.conductivity, ".")
    axes[1, 1].set_ylabel(r"$\sigma$ (mS/cm)")
    axes[1, 1].set_xlabel(
        r"${\rm x}$ in Na$_{\rm 1+x}$Zr$_{\rm 2}$Si$_{\rm x}$P$_{\rm 3-x}$O$_{\rm 12}$"
    )
    axes[1, 1].set_xlim((0, 3))

    fig.tight_layout()
    fig.savefig("plot.pdf")
