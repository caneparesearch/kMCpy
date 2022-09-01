import time
class Test_version3():
    def __init__(self,cutoff1=4.0) -> None:
        self.cutoff=cutoff1
        self.supercell=[8,8,8]
    
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
        
        time2=self.test_kmc_main_function()
        print("elapsed time for ",reference_local_env_dict, time2)
        return (self.cutoff, time2)    
    def test_generate_local_cluster_exapnsion(self):
        from pathlib import Path
        import os
        current_dir= Path(__file__).absolute().parent
        os.chdir(current_dir)
        from kmcpy.model import LocalClusterExpansion
        mobile_ion_identifier_type="label"
        mobile_ion_specie_1_identifier="Na1"
        tick=time.perf_counter()
        a=LocalClusterExpansion(api=3)
        a.initialization3(mobile_ion_identifier_type=mobile_ion_identifier_type,mobile_ion_specie_1_identifier=mobile_ion_specie_1_identifier,cutoff_cluster=[6,6,0],cutoff_region=4,template_cif_fname='./EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif',convert_to_primitive_cell=True)
        self.orbits=len(a.orbits)
        self.clusters=len(a.clusters)
        a.to_json("./input/lce.json")
        a.to_json("./input/lce_site.json")
        tock=time.perf_counter()
        print("time for lce: ",tock-tick)
            
    def test_generate_events(self):
        from pathlib import Path
        import os
        current_dir= Path(__file__).absolute().parent
        os.chdir(current_dir)
        mobile_ion_identifier_type="label"
        mobile_ion_specie_1_identifier="Na1"
        mobile_ion_specie_2_identifier="Na2"
        prim_cif_name="EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif"
        local_env_cutoff_dict={('Na+','Na+'):4,('Na+','Si4+'):4}
        from kmcpy.event_generator import generate_events3

        reference_local_env_dict=generate_events3(prim_cif_name=prim_cif_name,local_env_cutoff_dict=local_env_cutoff_dict,mobile_ion_identifier_type=mobile_ion_identifier_type,mobile_ion_specie_1_identifier=mobile_ion_specie_1_identifier,mobile_ion_specie_2_identifier=mobile_ion_specie_2_identifier,species_to_be_removed=["O2-","O","Zr4+","Zr"],distance_matrix_rtol=0.01,distance_matrix_atol=0.01,find_nearest_if_fail=False,convert_to_primitive_cell=True,export_local_env_structure=True,supercell_shape=self.supercell,event_fname="./input/events.json",event_kernal_fname='./input/event_kernal.csv',verbosity="INFO")
        
        site_event_list = []
        
        
        extended_dummy=[]
        for i in range(0,int(self.cutoff)):
            for j in range(0,12):
                extended_dummy.append(12*i+j)
        
        with open('./input/event_kernal.csv') as f:
            data = f.readlines()
        for x in data:
            if len(x.strip()) == 0:
                site_event_list.append([])
            else:
                print([int(y) for y in x.strip().split()])
                print(extended_dummy)
                print([int(y) for y in x.strip().split()]+(extended_dummy))
                site_event_list.append([int(y) for y in x.strip().split()]+extended_dummy)
        with open('./input/event_kernal.csv', 'w') as f:
  
            for row in site_event_list:
                for item in row:
                    f.write('%5d ' % item)
                f.write('\n')
            
        
        for i in reference_local_env_dict:
            return i
        #return reference_local_env_dict[0]


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

        inputset.parameter_checker()
        inputset.set_parameter("supercell_shape",self.supercell)
        inputset.set_parameter("occ",load_occ(fname=inputset._parameters["mc_results"],shape=inputset._parameters["supercell_shape"],select_sites=inputset._parameters["select_sites"],api=inputset.api,verbose=True))

        
        kmc = KMC(api=api)

        events_initialized = kmc.initialization(**inputset._parameters) # v in 10^13 hz


        # # step 2 compute the site kernal (used for kmc run)
        kmc.load_site_event_list(inputset._parameters["event_kernel"])

        
        tick=time.time()
        # # step 3 run kmc
        kmc_tracker=kmc.run_from_database(events=events_initialized,**inputset._parameters)
        tock=time.time()
        return (tock-tick)/3


if __name__ == '__main__':
    with open("cutoff_scalability.txt","w") as t:
        content=""
        
        basis_set=[]
        run_time=[]
        
        for i in range(0,30):
            a=Test_version3(cutoff1=i)
            b=a.time_test()
            basis_set=i*12+12
            run_time.append(b[1])
            content+=f"{12*i+12},{b[1]}"
        t.write(content)



import matplotlib.pyplot as plt
import numpy as np



from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def eq1(x,c1):
    return c1**x

def eq2(x,k1,k2):
    return k2*k1**x

def eq3(x,z1,z2,z3):
    return z2*z1**x + z3


popt3,_=curve_fit(eq3,basis_set,run_time,maxfev=5000)
z1,z2,z3=popt3
fit3=eq3(basis_set,z1,z2,z3)
r2_3=r2_score(run_time,fit3)

plt.scatter(basis_set,run_time)

plt.plot(fit3,label=r2_3,color="green")
plt.legend(loc="upper left")
plt.xlabel("basis")
plt.ylabel("run_time")
plt.show()