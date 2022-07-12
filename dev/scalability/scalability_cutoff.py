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
        data=[]
        for i in range(4,11):
            a=Test_version3(cutoff1=i)
            b=a.time_test()
            data.append(b)
            content+=str(b)
        t.write(content)

del data[0]
        
import matplotlib.pyplot as plt
import numpy as np
var=[]
run_time=[]
predict=[]
for j in range(0,len(data)):
    var.append(str(data[j][0]))
    run_time.append(data[j][1])
    predict.append(j)

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def eq1(x,c1):
    return c1**x

def eq2(x,k1,k2):
    return k2*k1**x

def eq3(x,z1,z2,z3):
    return z2*z1**x + z3

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

plt.scatter(var,run_time)
plt.plot(fit1,label=r2_1,color="red")
plt.plot(fit2,label=r2_2,color="blue")
plt.plot(fit3,label=r2_3,color="green")
plt.legend(loc="upper left")
plt.show()