import time
class Test_version3():
    def __init__(self,cutoff1=4.0) -> None:
        self.cutoff=cutoff1
        self.supercell=[3,3,3]
    
    def time_test(self):
        tick=time.time()
        with open("./input/initial_state.json","w") as f:
            occupation=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]*self.supercell[0]*self.supercell[1]*self.supercell[2]
            write_str="""{
    "occupation" :  """+str(occupation)+""" 
  }"""
            f.write(write_str)

        reference_local_env_dict=self.test_generate_events()
        #self.test_kmc_main_function()
        tock=time.time()
        print("elapsed time for ",reference_local_env_dict, tock-tick)
        return (reference_local_env_dict, tock-tick)    
    
    def test_generate_events(self):
        from pathlib import Path
        import os
        current_dir= Path(__file__).absolute().parent
        os.chdir(current_dir)
        mobile_ion_identifier_type="label"
        mobile_ion_specie_1_identifier="Na1"
        mobile_ion_specie_2_identifier="Na2"
        prim_cif_name="EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif"
        local_env_cutoff_dict={('Na+','Na+'):self.cutoff,('Na+','Si4+'):self.cutoff}
        from kmcpy.event_generator import generate_events3

        reference_local_env_dict=generate_events3(prim_cif_name=prim_cif_name,local_env_cutoff_dict=local_env_cutoff_dict,mobile_ion_identifier_type=mobile_ion_identifier_type,mobile_ion_specie_1_identifier=mobile_ion_specie_1_identifier,mobile_ion_specie_2_identifier=mobile_ion_specie_2_identifier,species_to_be_removed=["O2-","O","Zr4+","Zr"],distance_matrix_rtol=0.01,distance_matrix_atol=0.01,find_nearest_if_fail=False,convert_to_primitive_cell=True,export_local_env_structure=True,supercell_shape=self.supercell,event_fname="./input/events.json",event_kernal_fname='./input/event_kernal.csv',verbosity="INFO")
        for i in reference_local_env_dict:
            return i
        #return reference_local_env_dict[0]

        
  

  
        
if __name__ == '__main__':
    with open("cutoff_scalability.txt","w") as t:
        content=""
        for i in range(4,10):
            a=Test_version3(cutoff1=i)
            b=a.time_test()
            content+=str(b)
        t.write(content)
        
