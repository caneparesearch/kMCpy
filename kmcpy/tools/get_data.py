#!/usr/bin/env python
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


results = glob.glob("comp_0.1_1")


def calc_D_J(displacement, n_na, time, d=3):
    displacement_vector_tot = np.linalg.norm(np.sum(displacement, axis=0))
    D_J = displacement_vector_tot**2 / (2 * d * time * n_na) * 10 ** (-16)  # to cm^2/s
    print("log10(10^13/v* D_J) = ", np.log10(D_J), "cm^2/s")
    print("D_J = ", D_J, "cm^2/s")
    return D_J


def calc_D_tracer(displacement, n_na, time, d=3):
    displacement_vector_tot = np.sum((np.linalg.norm(displacement, axis=1)) ** 2)
    D_tracer = (
        displacement_vector_tot / (2 * d * time * n_na) * 10 ** (-16)
    )  # to cm^2/s
    print("log10(10^13/v* D_tracer) =", np.log10(D_tracer), "cm^2/s")
    print("D_tracer =", D_tracer, "cm^2/s")
    return D_tracer


def calc_corr_factor(
    displacement, n_na, hop_counter, a=3.47782
):  # a is the hopping distance in Angstrom
    n = np.average(hop_counter)
    # distance_squared_avg = np.average((self.hop_counter*a)**2)
    corr_factor = np.sum(np.linalg.norm(displacement) ** 2) / (n_na * n * a**2)
    # corr_factor = distance_squared_avg/(self.n_na*n*a**2)
    print("Correlation factor f = ", corr_factor)
    return corr_factor


def calc_D_c(D_J, theta):
    D_c = D_J * theta
    return D_c


def fit_theta():
    x, y = np.loadtxt("theta_575.txt")  # x,theta
    from scipy.interpolate import interp1d

    spl = interp1d(x, y, kind="cubic")
    return spl


def calc_conductivity(D_J, D_tracer, n_na, volume, q=1, T=300):
    H_R = D_tracer / D_J
    n = (n_na) / volume  # e per Angst^3 vacancy is the carrier
    k = 8.617333262145 * 10 ** (-2)  # unit in meV/K
    conductivity = D_J * n * q**2 / (k * T) * 1.602 * 10**11  # to mS/cm
    print("Haven's ratio H_R =", H_R)
    print("Conductivty: sigma = ", conductivity, "mS/cm")
    return conductivity


def search_string(fname, word, location):
    with open(fname) as f:
        for l in f.readlines():
            if word in l:
                res = l.strip().split()[location]
    return float(res)


spl = fit_theta()
data = []
for result in results:
    for n_pass in np.arange(0, 2000, 1, dtype="int"):
        print(result)
        x = 3 - 3 * float(result.strip().split("_")[1])
        structure_id = int(result.strip().split("_")[2])
        D_J, D_tracer, f, conductivity = np.loadtxt(
            result + "/migration_" + str(n_pass) + ".txt"
        )
        # displacement = np.loadtxt(result+'/displacement.txt')
        # hop_counter = np.loadtxt(result+'/hop_count.txt')
        # time = search_string(result+'/kmc_stdout.txt','Time',2)
        # n_na = search_string(result+'/kmc_stdout.txt','n_Na_sites',2)
        # volume = 2213256.28246698
        # D_J = calc_D_J(displacement,n_na,time,d=3)
        # D_tracer = calc_D_tracer(displacement,n_na,time,d=3)
        theta = spl(x)
        D_c = calc_D_c(D_J, theta)
        print("D_c = ", D_c)
        # f = calc_corr_factor(displacement,n_na,hop_counter)
        # conductivity = calc_conductivity(D_J,D_tracer,n_na,volume,q=1,T=572)
        data.append(
            [
                x,
                n_pass,
                structure_id,
                D_J,
                D_tracer,
                D_c,
                theta,
                f,
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
            "D_J",
            "D_tracer",
            "D_c",
            "Theta",
            "f",
            "Conductivity",
            "location",
        ],
    ).sort_values(by=["x", "n_pass"])

    df.to_csv("data_all.csv")
    df2 = df.groupby("x").mean().reset_index().sort_values(by="x")
    print(df2)
    df2.to_csv("data.csv")

    df2 = pd.read_csv("data.csv")
    fig, axes = plt.subplots(2, 2, figsize=(5, 4), sharex="col")
    axes[0, 0].plot(df2.x, df2.D_J, ".")
    axes[0, 0].set_ylabel(r"$D_{J}$ (cm$^2$/s)")
    axes[0, 0].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2e"))

    axes[0, 1].plot(df2.x, df2.D_tracer, ".")
    axes[0, 1].set_ylabel(r"$D^{*}$ (cm$^2$/s)")
    axes[0, 1].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2e"))

    axes[1, 0].plot(df2.x, df2.f, ".")
    axes[1, 0].set_ylabel(r"Correlation factor")
    axes[1, 0].set_xlabel(
        r"${\rm x}$ in Na$_{\rm 1+x}$Zr$_{\rm 2}$Si$_{\rm x}$P$_{\rm 3-x}$O$_{\rm 12}$"
    )
    axes[1, 0].set_ylim((0, 1))
    axes[1, 0].set_xlim((0, 3))

    axes[1, 1].plot(df2.x, df2.Conductivity, ".")
    axes[1, 1].set_ylabel(r"$\sigma$ (mS/cm)")
    axes[1, 1].set_xlabel(
        r"${\rm x}$ in Na$_{\rm 1+x}$Zr$_{\rm 2}$Si$_{\rm x}$P$_{\rm 3-x}$O$_{\rm 12}$"
    )
    axes[1, 1].set_xlim((0, 3))

    fig.tight_layout()

    fig.savefig("plot.pdf")
