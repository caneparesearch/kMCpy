import unittest

class TestStringMethods(unittest.TestCase):
    from kmcpy.event_generator import generate_events2

    convert_to_primitive_cell=True
    # event kernal
    generate_events2(prim_cif_name="input/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",convert_to_primitive_cell=convert_to_primitive_cell,supercell_shape=[2,1,1],local_env_cutoff_dict = {('Na+','Na+'):4,('Na+','Si4+'):4},event_fname="input/events.json",event_kernal_fname='input/event_kernal.csv',center_atom_label_or_indices="Na1",species_to_be_removed=['Zr4+','O2-','O','Zr'],diffuse_to_atom_label="Na2",verbose=True,hacking_arg={1: [18, 20, 19, 22, 21, 23, 108, 110, 109, 112, 111, 113]})

    # model kernal

    from kmcpy.model import LocalClusterExpansion

    a=LocalClusterExpansion(api=2)
    a.initialization(center_atom_index="Na1",cutoff_cluster=[6,6,0],cutoff_region=4,template_cif_fname='./input/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif',species_to_be_removed=['Zr4+','O2-','O','Zr'],convert_to_primitive_cell=convert_to_primitive_cell)
    a.to_json("input/lce.json")
    a.to_json("input/lce_site.json")

    from kmcpy.io import InputSet,load_occ
    from kmcpy.kmc import KMC

    api=2
    inputset=InputSet.from_json("input/test_input_v2.json",api=2)

    print(inputset._parameters.keys())
    print(inputset._parameters["mc_results"])
    inputset.parameter_checker()

    inputset.set_parameter("occ",load_occ(fname=inputset._parameters["mc_results"],shape=inputset._parameters["supercell_shape"],select_sites=inputset._parameters["select_sites"],api=inputset.api,verbose=True))
    
    
    

    # step 1 initialize global occupation and conditions
    
    
    
    kmc = KMC(api=api)
    events_initialized = kmc.initialization(**inputset._parameters) # v in 10^13 hz

    # # step 2 compute the site kernal (used for kmc run)
    kmc.load_site_event_list(inputset._parameters["event_kernel"])

    # # step 3 run kmc
    kmc.run_from_database(events=events_initialized,**inputset._parameters)




if __name__ == '__main__':
    unittest.main()