import unittest

class cluster_matcher_test(unittest.TestCase):
    from kmcpy.event_generator import generate_events3
    from kmcpy.event_generator import cluster_matcher
    import numpy as np
    import json
    import logging
    from kmcpy.external.pymatgen_structure import Structure
    from kmcpy.external.pymatgen_local_env import CutOffDictNN
    import itertools
    from kmcpy.io import convert
    from kmcpy.event import Event
    
    logging.basicConfig(level=1,handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ])
    nasicon=Structure.from_cif("EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",primitive=True)

    center_Na1=[0,1]
    local_env_finder = CutOffDictNN({('Na+','Na+'):4,('Na+','Si4+'):4})


    reference_neighbor_sequences=sorted(sorted(local_env_finder.get_nn_info(nasicon,center_Na1[0]),key=lambda x:x["wyckoff_sequence"]),key = lambda x:x["label"])     

    np.set_printoptions(precision=2,suppress=True)
    a=cluster_matcher.from_reference_neighbor_sequences(reference_neighbor_sequences=reference_neighbor_sequences)


    print(a.neighbor_species)
    print(a.reference_distance_matrix)
    print(a.neighbor_species_respective_reference_distance_matrix_dict)
    print(a.neighbor_species_respective_reference_neighbor_sequence_dict)


    wrong_sequence_neighbor=sorted(sorted(local_env_finder.get_nn_info(nasicon,center_Na1[1]),key=lambda x:x["wyckoff_sequence"]),key = lambda x:x["label"]) 


    a.brutal_match(wrong_sequence_neighbor,rtol=0.01)

class generate_nasicon_events(unittest.TestCase):
    from kmcpy.event_generator import generate_events3
    generate_events3(prim_cif_name="EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",local_env_cutoff_dict={('Na+','Na+'):4,('Na+','Si4+'):4},atom_identifier_type="label",center_atom_identifier="Na1",diffuse_to_atom_identifier="Na2",species_to_be_removed=["O2-","O","Zr4+","Zr"],distance_matrix_rtol=0.01,distance_matrix_atol=0.01,find_nearest_if_fail=False,convert_to_primitive_cell=False,create_reference_cluster=True,supercell_shape=[2,1,1],event_fname="events.json",event_kernal_fname='event_kernal.csv',verbosity=0)


if __name__ == '__main__':
    unittest.main()