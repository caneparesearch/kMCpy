from kmcpy.io import InputSet
from kmcpy.kmc import KMC

import logging

## for debugging purposes, we set the logging level to DEBUG
logging.basicConfig(
    level=logging.DEBUG, # Set to DEBUG to see everything
    format='%(asctime)s - %(name)-28s - %(levelname)-8s - %(message)s',
    datefmt='%Y-%m-%d %H:%M',
    filename='run.log',
    filemode='w',)  # Log to a file named debug.log


inputset = InputSet.from_json("kmc_input_files/kmc_input_test.json")

print(inputset._parameters.keys())
print(inputset._parameters["initial_state"])

kmc = KMC.from_inputset(inputset)

kmc_tracker = kmc.run(inputset)
# print(kmc_tracker.return_current_info())
