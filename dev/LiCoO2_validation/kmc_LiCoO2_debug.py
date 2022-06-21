mobile_ion_identifier_type="label"
mobile_ion_specie_1_identifier="Na0"
mobile_ion_specie_2_identifier="Li1"
prim_cif_name="LiCoO2_dummy_added.cif"
local_env_cutoff_dict={('Na+','Li+'):4.31}
from kmcpy.event_generator import *
import itertools
import os
os.chdir("/Users/weihangxie/Documents/GitHub/kmcPy_dev/dev/LiCoO2_validation")
  
generate_events3(prim_cif_name=prim_cif_name,local_env_cutoff_dict=local_env_cutoff_dict,mobile_ion_identifier_type=mobile_ion_identifier_type,mobile_ion_specie_1_identifier=mobile_ion_specie_1_identifier,mobile_ion_specie_2_identifier=mobile_ion_specie_2_identifier,species_to_be_removed=["O2-","O","Co+","Co"],distance_matrix_rtol=0.01,distance_matrix_atol=0.01,find_nearest_if_fail=False,convert_to_primitive_cell=False,export_local_env_structure=True,supercell_shape=[2,2,2],event_fname="events.json",event_kernal_fname='event_kernal.csv',verbosity="INFO")