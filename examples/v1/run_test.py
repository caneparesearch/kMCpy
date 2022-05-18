import unittest

class TestStringMethods(unittest.TestCase):
    from kmcpy.model import LocalClusterExpansion

    a=LocalClusterExpansion(api=1)
    a.initialization1(center_Na1_index=0,cutoff_cluster=[6,6,0],cutoff_region=4,template_cif_fname='./EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif')
    a.to_json("lce.json")
    from kmcpy.event_generator import generate_events


    from kmcpy.io import InputSet,load_occ
    from kmcpy.kmc import KMC
    import os
    import sys


    inputset=InputSet.from_json("./test_input.json")

    print(inputset._parameters.keys())
    inputset.parameter_checker()



    inputset.set_parameter("occ",load_occ(inputset._parameters["mc_results"],inputset._parameters["supercell_shape"],api=inputset.api))
    # step 1 initialize global occupation and conditions
    kmc = KMC()
    events_initialized = kmc.initialization(**inputset._parameters) # v in 10^13 hz

    # # step 2 compute the site kernal (used for kmc run)
    kmc.load_site_event_list(inputset._parameters["event_kernel"])

    # # step 3 run kmc
    kmc.run_from_database(events=events_initialized,**inputset._parameters)

    pass
if __name__ == '__main__':
    unittest.main()