# just want to use the debug function in vscode to see the variables
from kmcpy.io import InputSet,load_occ
from kmcpy.kmc import KMC
import os
import sys



def main(api=1,**kwargs):
    print(os.getcwd())
    os.chdir("./examples")
    inputset=InputSet.from_json("./test_input.json")

    print(inputset._parameters.keys())
    inputset.parameter_checker()

    if api==1:


        inputset.set_parameter("occ",load_occ(inputset._parameters["mc_results"],inputset._parameters["supercell_shape"],api=inputset.api))
        # step 1 initialize global occupation and conditions
        kmc = KMC()
        events_initialized = kmc.initialization(**inputset._parameters) # v in 10^13 hz

        # # step 2 compute the site kernal (used for kmc run)
        kmc.load_site_event_list(inputset._parameters["event_kernel"])

        # # step 3 run kmc
        kmc.run_from_database(events=events_initialized,**inputset._parameters)
    if api==2:
        raise NotImplementedError
    pass


if __name__ == "__main__":

    main(api=2)