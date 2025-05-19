from kmcpy.fitting import Fitting
import numpy as np

local_cluster_expansion_fit = Fitting()

print("initializing")

y_pred, y_true = local_cluster_expansion_fit.fit(
    alpha=1.5,
    max_iter=1000000,
    ekra_fname="fit_input_files/E_kra_middle.txt",
    keci_fname="fit_output_files/keci.txt",
    weight_fname="fit_input_files/weights.txt",
    corr_fname="lce_output_files/correlation_matrix.txt",
    fit_results_fname="fit_output_files/fitting_results.json",
)
print("fitting", y_pred, y_true)
local_cluster_expansion_fit = Fitting()

y_pred, y_true = local_cluster_expansion_fit.fit(
    alpha=1.5,
    max_iter=1000000,
    ekra_fname="fit_input_files/E_kra_middle.txt",
    keci_fname="fit_output_files/keci.txt",
    weight_fname="fit_input_files/weights.txt",
    corr_fname="lce_output_files/correlation_matrix.txt",
    fit_results_fname="fit_output_files/fitting_results.json"
)
print("fitting", y_pred, y_true)
