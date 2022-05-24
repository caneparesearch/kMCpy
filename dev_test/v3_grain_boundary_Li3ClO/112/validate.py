# test on 112
from kmcpy.external.pymatgen_structure import Structure
import numpy as np
from kmcpy.event_generator import generate_events3
gb_112=Structure.from_file("112_CONTCAR")
gb_112.to("cif","112.cif")

np.set_printoptions(precision=2,suppress=True)

    
# even with rtol=0.25 it is not working. Trying set rtol=0.35
    
generate_events3(prim_cif_name="112.cif",local_env_cutoff_dict={("Li+","Cl-"):4.0,("Li+","Li+"):3.0},atom_identifier_type="specie",center_atom_identifier="Li+",diffuse_to_atom_identifier="Li+",species_to_be_removed=["O2-","O"],distance_matrix_rtol=0.35,distance_matrix_atol=0.5,find_nearest_if_fail=True,convert_to_primitive_cell=False,create_reference_cluster=True,supercell_shape=[2,1,1],event_fname="events.json",event_kernal_fname='event_kernal.csv',verbosity=0)

