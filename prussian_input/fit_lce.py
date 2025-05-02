from kmcpy.fitting import Fitting
import numpy as np

local_cluster_expansion_fit = Fitting()

print("initializing")

y_pred, y_true = local_cluster_expansion_fit.fit(
    alpha=1.5,
    max_iter=1000000,
    ekra_fname="E_kra_middle.txt",
    keci_fname="keci.txt",
    weight_fname="weights.txt",
    corr_fname="correlation_matrix.txt",
    fit_results_fname="fitting_results.json",
)
print("fitting", y_pred, y_true)
local_cluster_expansion_fit = Fitting()

y_pred, y_true = local_cluster_expansion_fit.fit(
    alpha=1.5,
    max_iter=1000000,
    ekra_fname="E_kra_middle.txt",
    keci_fname="keci.txt",
    weight_fname="weights.txt",
    corr_fname="correlation_matrix.txt",
    fit_results_fname="fitting_results_site.json",
)
print("fitting", y_pred, y_true)
