import unittest


class generate_grainboundary_events(unittest.TestCase):
    from kmcpy.event_generator import generate_events3
    import numpy as np
    np.set_printoptions(precision=2,suppress=True)
    
    
    generate_events3(prim_cif_name="210.cif",local_env_cutoff_dict={("Li+","Cl-"):4.0,("Li+","Li+"):3.0},mobile_ion_identifier_type="specie",mobile_ion_specie_1_identifier="Li+",mobile_ion_specie_2_identifier="Li+",species_to_be_removed=["O2-","O"],distance_matrix_rtol=0.25,distance_matrix_atol=0.5,find_nearest_if_fail=True,convert_to_primitive_cell=False,create_reference_cluster=True,supercell_shape=[2,1,1],event_fname="events.json",event_kernal_fname='event_kernal.csv',verbosity=0)


if __name__ == '__main__':
    unittest.main()