from kmcpy.event_generator import generate_events2
from kmcpy.external.pymatgen_structure import Structure

#psuedocubic=Structure.from_file("Transformed.cif")
#psuedocubic.to("cif","transformed2.cif",symprec=0.001)

generate_events2(prim_cif_name="transformed2.cif",supercell_shape=[2,1,1],local_env_cutoff_dict = {('Li+','Na+'):4},event_fname="events.json",event_kernal_fname='event_kernal.csv',mobile_ion_specie_1_index_label_or_indices="Li1",mobile_ion_specie_2_index_atom_label="Na0",species_to_be_removed=['O2-'],verbose=True,hacking_arg=None)

#generate_events2(prim_cif_name="transformed2.cif",supercell_shape=[2,1,1],local_env_cutoff_dict = {('V2.5+','Na+'):8,('V2.5+','P5+'):8},event_fname="events.json",event_kernal_fname='event_kernal.csv',mobile_ion_specie_1_index_label_or_indices="Na1",mobile_ion_specie_2_index_atom_label="Na0",species_to_be_removed=[],verbose=True,hacking_arg=None)