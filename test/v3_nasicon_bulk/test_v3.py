import unittest

class Test_version3(unittest.TestCase):
    def neighbor_info_matcher(self):
        from pathlib import Path
        import os
        current_dir= Path(__file__).absolute().parent
        os.chdir(current_dir) 
        from kmcpy.event_generator import generate_events3,neighbor_info_matcher
        import numpy as np

        from kmcpy.external.pymatgen_structure import Structure
        from kmcpy.external.pymatgen_local_env import CutOffDictNN

        nasicon=Structure.from_cif("EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",primitive=True)

        center_Na1=[0,1]
        local_env_finder = CutOffDictNN({('Na+','Na+'):4,('Na+','Si4+'):4})


        reference_neighbor_sequences=sorted(sorted(local_env_finder.get_nn_info(nasicon,center_Na1[0]),key=lambda x:x["wyckoff_sequence"]),key = lambda x:x["label"])     

        np.set_printoptions(precision=2,suppress=True)
        a=neighbor_info_matcher.from_neighbor_sequences(neighbor_sequences=reference_neighbor_sequences)


        print(a.neighbor_species)
        print(a.distance_matrix)
        print(a.neighbor_species_respective_distance_matrix_dict)
        print(a.neighbor_species_respective_neighbor_sequence_dict)


        wrong_sequence_neighbor=sorted(sorted(local_env_finder.get_nn_info(nasicon,center_Na1[1]),key=lambda x:x["wyckoff_sequence"]),key = lambda x:x["label"]) 

        a.brutal_match(wrong_sequence_neighbor,rtol=0.01)
        
    def test_generate_events(self):
        from pathlib import Path
        import os
        current_dir= Path(__file__).absolute().parent
        os.chdir(current_dir)
        atom_identifier_type="label"
        center_atom_identifier="Na1"
        diffuse_to_atom_identifier="Na2"
        prim_cif_name="EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif"
        local_env_cutoff_dict={('Na+','Na+'):4,('Na+','Si4+'):4}
        from kmcpy.event_generator import generate_events3
        generate_events3(prim_cif_name=prim_cif_name,local_env_cutoff_dict=local_env_cutoff_dict,atom_identifier_type=atom_identifier_type,center_atom_identifier=center_atom_identifier,diffuse_to_atom_identifier=diffuse_to_atom_identifier,species_to_be_removed=["O2-","O","Zr4+","Zr"],distance_matrix_rtol=0.01,distance_matrix_atol=0.01,find_nearest_if_fail=False,convert_to_primitive_cell=False,export_local_env_structure=True,supercell_shape=[2,1,1],event_fname="events.json",event_kernal_fname='event_kernal.csv',verbosity="INFO")

        generate_events3(prim_cif_name=prim_cif_name,local_env_cutoff_dict=local_env_cutoff_dict,atom_identifier_type=atom_identifier_type,center_atom_identifier=center_atom_identifier,diffuse_to_atom_identifier=diffuse_to_atom_identifier,species_to_be_removed=["O2-","O","Zr4+","Zr"],distance_matrix_rtol=0.01,distance_matrix_atol=0.01,find_nearest_if_fail=False,convert_to_primitive_cell=True,export_local_env_structure=True,supercell_shape=[2,1,1],event_fname="events.json",event_kernal_fname='event_kernal.csv',verbosity="INFO")
        
        self.assertEqual(1,1)



    def test_generate_local_cluster_exapnsion(self):
        from pathlib import Path
        import os
        current_dir= Path(__file__).absolute().parent
        os.chdir(current_dir)
        from kmcpy.model import LocalClusterExpansion
        atom_identifier_type="label"
        center_atom_identifier="Na1"
        a=LocalClusterExpansion(api=3)
        a.initialization3(atom_identifier_type=atom_identifier_type,center_atom_identifier=center_atom_identifier,cutoff_cluster=[6,6,0],cutoff_region=4,template_cif_fname='./EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif')
        a.to_json("lce.json")
        self.assertEqual(1,1)
    
    def test_kmc_main_function(self):
        from pathlib import Path
        import os
        current_dir= Path(__file__).absolute().parent
        os.chdir(current_dir)
        from kmcpy.io import InputSet,load_occ
        from kmcpy.kmc import KMC
        api=2
        inputset=InputSet.from_json("input/test_input_v3.json",api=2)

        print(inputset._parameters.keys())
        print(inputset._parameters["mc_results"])
        inputset.parameter_checker()

        inputset.set_parameter("occ",load_occ(fname=inputset._parameters["mc_results"],shape=inputset._parameters["supercell_shape"],select_sites=inputset._parameters["select_sites"],api=inputset.api,verbose=True))
        kmc = KMC(api=api)
        events_initialized = kmc.initialization(**inputset._parameters) # v in 10^13 hz

        # # step 2 compute the site kernal (used for kmc run)
        kmc.load_site_event_list(inputset._parameters["event_kernel"])

        # # step 3 run kmc
        kmc.run_from_database(events=events_initialized,**inputset._parameters)
        self.assertEqual(1,1)

if __name__ == '__main__':
    unittest.main()