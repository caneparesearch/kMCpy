from kmcpy.io import InputSet
from kmcpy.kmc import KMC
import numpy as np

inputset = InputSet.from_json("kmc_input_files/kmc_input.json")

print(inputset._parameters.keys())
print(inputset._parameters["mc_results"])
inputset.parameter_checker()

inputset.load_occ(verbose=True)

inputset.set_parameter("use_numpy_random_kernel", True)
kmc = KMC()

events_initialized = kmc.initialization(**inputset._parameters)  # v in 10^13 hz

# # step 2 compute the site kernal (used for kmc run)
kmc.load_site_event_list(inputset._parameters["event_kernel"])

# # step 3 run kmc
kmc_tracker = kmc.run_from_database(events=events_initialized, **inputset._parameters)
# print(kmc_tracker.return_current_info())
