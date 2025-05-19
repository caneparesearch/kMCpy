from kmcpy.io import InputSet, load_occ
from kmcpy.kmc import KMC
import numpy as np

api = 3
inputset = InputSet.from_json("test_input_v3.json", api=3)

print(inputset._parameters.keys())
print(inputset._parameters["mc_results"])
inputset.parameter_checker()

inputset.set_parameter(
    "occ",
    load_occ(
        fname=inputset._parameters["mc_results"],
        shape=inputset._parameters["supercell_shape"],
        select_sites=inputset._parameters["select_sites"],
        api=inputset.api,
        verbose=True,
    ),
)
inputset.set_parameter("use_numpy_random_kernel", True)
kmc = KMC(api=api)

events_initialized = kmc.initialization(**inputset._parameters)  # v in 10^13 hz

# # step 2 compute the site kernal (used for kmc run)
kmc.load_site_event_list(inputset._parameters["event_kernel"])

# # step 3 run kmc
kmc_tracker = kmc.run_from_database(events=events_initialized, **inputset._parameters)
# print(kmc_tracker.return_current_info())
