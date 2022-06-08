#!/usr/bin/env python
"""a Wrapper for running the KMC, receive 1 standard input of file path, read the file as the incar and run KMC

Usage:

./wrapper.py ../examples/test_input.json

Raises:
    NotImplementedError: _description_
"""



from kmcpy.io import InputSet,load_occ
from kmcpy.kmc import KMC

import argparse






def main(api=3,**kwargs):
    """
    This is the wrapper for executing KMC
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('incar', metavar='N', type=str,help='path to the input.json')
    args = parser.parse_args()
    inputset=InputSet.from_json(args.incar,api=api)
    inputset.parameter_checker()
    # check if the parameter is good

    inputset.set_parameter("occ",load_occ(inputset._parameters["mc_results"],inputset._parameters["supercell_shape"],api=inputset.api))
    # step 1 initialize global occupation and conditions
    kmc = KMC()
    events_initialized = kmc.initialization(**inputset._parameters) # v in 10^13 hz

    # # step 2 compute the site kernal (used for kmc run)
    kmc.load_site_event_list(inputset._parameters["event_kernel"])

    # # step 3 run kmc
    kmc.run_from_database(events=events_initialized,**inputset._parameters)

    pass


if __name__ == "__main__":

    main()
