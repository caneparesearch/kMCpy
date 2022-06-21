

import unittest
import pytest
class Test_version3(unittest.TestCase):
    

    @pytest.mark.order(2)        
    def test_generate_events(self):
        mobile_ion_identifier_type="label"
        mobile_ion_specie_1_identifier="Na0"
        mobile_ion_specie_2_identifier="Li1"
        prim_cif_name="LiCoO2_dummy_added.cif"
        local_env_cutoff_dict={('Na+','Li+'):4.31}
        from kmcpy.event_generator import generate_events3

        
        reference_local_env_dict=generate_events3(prim_cif_name=prim_cif_name,local_env_cutoff_dict=local_env_cutoff_dict,mobile_ion_identifier_type=mobile_ion_identifier_type,mobile_ion_specie_1_identifier=mobile_ion_specie_1_identifier,mobile_ion_specie_2_identifier=mobile_ion_specie_2_identifier,species_to_be_removed=["O2-","O","Co+","Co"],distance_matrix_rtol=0.01,distance_matrix_atol=0.01,find_nearest_if_fail=False,convert_to_primitive_cell=False,export_local_env_structure=True,supercell_shape=[2,2,2],event_fname="events.json",event_kernal_fname='event_kernal.csv',verbosity="INFO")
        
        
        
        self.assertEqual(len(reference_local_env_dict),1)  # only one type of local environment should be found. If more than 1, raise error.


    @pytest.mark.order(3)
    def test_generate_local_cluster_exapnsion(self):
        from pathlib import Path
        import os
        current_dir= Path(__file__).absolute().parent
        os.chdir(current_dir)
        from kmcpy.model import LocalClusterExpansion
        mobile_ion_identifier_type="label"
        mobile_ion_specie_1_identifier="Na0"
        mobile_ion_specie_2_identifier="Li1"
        prim_cif_name="LiCoO2_dummy_added.cif"
        local_env_cutoff_dict={('Na+','Li+'):4.31}
        species_to_be_removed=["O2-","O","Co+","Co"]
        a=LocalClusterExpansion(api=3)
        a.initialization3(mobile_ion_identifier_type=mobile_ion_identifier_type,mobile_ion_specie_1_identifier=mobile_ion_specie_1_identifier,cutoff_cluster=[6,6,0],cutoff_region=4.31,template_cif_fname="LiCoO2_dummy_added.cif",convert_to_primitive_cell=False,species_to_be_removed=species_to_be_removed,exclude_species=["Na","Na+"])
        a.to_json("./input/lce.json")
        self.assertEqual(1,1)
        
    def test_fitting(self):
        from pathlib import Path
        import os
        current_dir= Path(__file__).absolute().parent
        os.chdir(current_dir)


        from kmcpy.fitting import Fitting
        import numpy as np

        local_cluster_expansion_fit = Fitting()

        y_pred, y_true = local_cluster_expansion_fit.fit(alpha=1.5,max_iter=1000000,ekra_fname='fitting/local_cluster_expansion/e_kra.txt',keci_fname='keci.txt',
            weight_fname='fitting/local_cluster_expansion/weight.txt',corr_fname='fitting/local_cluster_expansion/correlation_matrix.txt',
            fit_results_fname='fitting_results.json')
        print("fitting",y_pred, y_true )
        self.assertTrue(np.allclose(y_pred, y_true,rtol=0.3,atol=10.0))
        
    @pytest.mark.order(4)    
    def test_kmc_main_function(self):
        from pathlib import Path
        import os
        current_dir= Path(__file__).absolute().parent
        os.chdir(current_dir)
        from kmcpy.io import InputSet,load_occ
        from kmcpy.kmc import KMC
        import numpy as np
        api=3
        inputset=InputSet.from_json("input/test_input_v3.json",api=api)

        print(inputset._parameters.keys())
        print(inputset._parameters["mc_results"])
        inputset.parameter_checker()

        inputset.set_parameter("occ",load_occ(fname=inputset._parameters["mc_results"],shape=inputset._parameters["supercell_shape"],select_sites=inputset._parameters["select_sites"],api=inputset.api,verbose=True))
        kmc = KMC(api=api)
        events_initialized = kmc.initialization(**inputset._parameters) # v in 10^13 hz

        # # step 2 compute the site kernal (used for kmc run)
        kmc.load_site_event_list(inputset._parameters["event_kernel"])

        # # step 3 run kmc
        kmc_tracker=kmc.run_from_database(events=events_initialized,**inputset._parameters)
        print(kmc_tracker.return_current_info())
        self.assertTrue(np.allclose(np.array(kmc_tracker.return_current_info()),np.array((3.508959816621752e-06, 101.40523388197452, 1.8796478422471864e-09, 4.816490697215713e-10, 0.15190642462810805, 0.2562443128419963, 0.24387012300994337)),rtol=0.01,atol=0.01))
        
        
        # np.array((3.517242770690013e-06, 26.978226076495748, 3.187544456106211e-10, 1.2783794881088614e-10, 0.025760595723683707, 0.4010546380490277, 0.04309185078659044)) this is run from the given random number kernal and random number seed. This is a very strict criteria to see if the behavior of KMC is correct
    @pytest.mark.order(5)
    def test_kmc_main_function_randomized(self):
        from pathlib import Path
        import os
        current_dir= Path(__file__).absolute().parent
        os.chdir(current_dir)
        from kmcpy.io import InputSet,load_occ
        from kmcpy.kmc import KMC
        import numpy as np
        api=3
        inputset=InputSet.from_json("input/test_input_v3.json",api=3)

        print(inputset._parameters.keys())
        print(inputset._parameters["mc_results"])
        inputset.parameter_checker()

        inputset.set_parameter("occ",load_occ(fname=inputset._parameters["mc_results"],shape=inputset._parameters["supercell_shape"],select_sites=inputset._parameters["select_sites"],api=inputset.api,verbose=True))
        inputset.set_parameter("use_numpy_random_kernel",False)
        kmc = KMC(api=api)
        events_initialized = kmc.initialization(**inputset._parameters) # v in 10^13 hz

        # # step 2 compute the site kernal (used for kmc run)
        kmc.load_site_event_list(inputset._parameters["event_kernel"])

        # # step 3 run kmc
        kmc_tracker=kmc.run_from_database(events=events_initialized,**inputset._parameters)
        print(kmc_tracker.return_current_info())
        self.assertFalse(np.allclose(np.array(kmc_tracker.return_current_info()),np.array((3.508959816621752e-06, 101.40523388197452, 1.8796478422471864e-09, 4.816490697215713e-10, 0.15190642462810805, 0.2562443128419963, 0.24387012300994337)),rtol=0.01,atol=0.01))
        
        
        # np.array((3.517242770690013e-06, 26.978226076495748, 3.187544456106211e-10, 1.2783794881088614e-10, 0.025760595723683707, 0.4010546380490277, 0.04309185078659044)) this is run from the given random number kernal and random number seed. This is a very strict criteria to see if the behavior of KMC is correct
        
        
     
if __name__ == '__main__':
    unittest.main()