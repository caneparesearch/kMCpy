import time
class Test_version3():
    def __init__(self,supercell=[2,2,2]) -> None:
        self.supercell=supercell
    
    def time_test(self):
        tick=time.time()
        with open("./input/initial_state.json","w") as f:
            occupation=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]*self.supercell[0]*self.supercell[1]*self.supercell[2]
            write_str="""{
    "occupation" :  """+str(occupation)+""" 
  }"""
            f.write(write_str)

        self.test_generate_events()
        self.test_kmc_main_function()
        tock=time.time()
        print("elapsed time for ",self.supercell, tock-tick)
        return (self.supercell, tock-tick)    
    
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

        # # step 3 run kmc
        kmc_tracker=kmc.run_from_database(events=events_initialized,**inputset._parameters)



        

        
if __name__ == '__main__':
    with open("supercell_scalability_log.txt","w") as t:
        content=""
        data=[]
        for i in range(1,5):
            a=Test_version3(supercell=[i,i,i])
            b=a.time_test()
            data.append(b)
            content+=str(b)
        t.write(content)
        
import matplotlib.pyplot as plt
import numpy as np
cell_size=[]
run_time=[]
predict=[]
for j in range(0,len(data)):
    cell_size.append(str(data[j][0]))
    run_time.append(data[j][1])
    predict.append(j)
predict=np.array(predict)

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def eq1(x,c1):
    return c1**x

def eq2(x,k1,k2):
    return k2*k1**x

def eq3(x,z1,z2,z3):
    return z2*z1**x + z3

#def eq4(x,a1,a2,a3):
#    return a1*x**2 + a2*x + a3

#def eq5(x,b1,b2,b3,b4):
#    return b1*x**3 + b2*x**2 + b3*x + b4

#def eq6(x,p):
#    return x**p

popt1,_=curve_fit(eq1,predict,run_time)
c1=popt1
fit1=eq1(predict,c1)
r2_1=r2_score(run_time,fit1)

popt2,_=curve_fit(eq2,predict,run_time)
k1,k2=popt2
fit2=eq2(predict,k1,k2)
r2_2=r2_score(run_time,fit2)

popt3,_=curve_fit(eq3,predict,run_time)
z1,z2,z3=popt3
fit3=eq3(predict,z1,z2,z3)
r2_3=r2_score(run_time,fit3)

#popt4,_=curve_fit(eq4,predict,run_time)
#a1,a2,a3=popt4
#fit4=eq4(predict,a1,a2,a3)
#r2_4=r2_score(run_time,fit4)

#popt5,_=curve_fit(eq5,predict,run_time)
#b1,b2,b3,b4=popt5
#fit5=eq5(predict,b1,b2,b3,b4)
#r2_5=r2_score(run_time,fit5)

#popt6,_=curve_fit(eq6,predict,run_time)
#p=popt6
#fit6=eq6(predict,p)
#r2_6=r2_score(run_time,fit6)

plt.scatter(cell_size,run_time)
plt.plot(fit1,label=r2_1,color="red")
plt.plot(fit2,label=r2_2,color="blue")
plt.plot(fit3,label=r2_3,color="green")
plt.legend(loc="upper left")
plt.xlabel("cell_size")
plt.ylabel("run_time (sec)")
plt.show()