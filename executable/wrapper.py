#!/usr/bin/env python
"""a Wrapper for running the KMC, receive 1 standard input of file path, read the file as the incar and run KMC

Raises:
    NotImplementedError: _description_
"""



from kmcpy.io import InputSet,load_occ
from kmcpy.kmc import KMC
import os
import sys



input_json_path= sys.argv[1]
inputset=InputSet.from_json(input_json_path)

print(inputset._parameters.keys())

def main(api=1,**kwargs):
    inputset.parameter_checker(api)
    if api==1:


        inputset.set_parameter("occ",load_occ(inputset._parameters["mc_results"],inputset._parameters["supercell_shape"]))
        # step 1 initialize global occupation and conditions
        kmc = KMC()
        events_initialized = kmc.initialization(**inputset._parameters) # v in 10^13 hz

        # # step 2 compute the site kernal (used for kmc run)
        kmc.load_site_event_list(inputset._parameters["event_kernel"])

        # # step 3 run kmc
        kmc.run_from_database(events=events_initialized,**inputset._parameters)
    if api>1:
        raise NotImplementedError
    pass


if __name__ == "__main__":

    main()
