# script for initializing the local cluster expansion to predict barriers (mean barrier)

from kmcpy.model import LocalClusterExpansion
mobile_ion_identifier_type="specie" #"specie"
mobile_ion_specie_1_identifier="Na"

# cutoff_region = 6.2 # Check to match in Vesta # Seems the Na--Na distance are about 6 Å
# cif_file = "relaxed_plus_U_Na8Fie8N24C24_structure.cif"
#cif_file = "EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif"
cif_file = "full_24d.cif"

cutoff_region = 4.0 #5.1
# cutoff_cluster = [3.5,3.5,3.5] # default
cutoff_cluster = [3.5, 3.5, 3.5]
lce=LocalClusterExpansion(api=2) # api?
lce.initialization3(
    mobile_ion_identifier_type=mobile_ion_identifier_type,
    mobile_ion_specie_1_identifier=mobile_ion_specie_1_identifier,
    cutoff_cluster=cutoff_cluster,
    cutoff_region=cutoff_region,
    template_cif_fname=cif_file,
    convert_to_primitive_cell=True # It is already the primitive cell
    )
lce.to_json("lce.json")
lce.to_json("lce_site.json")

# # print(cif_file)
# # Writing correlation-matrix txt file
lce.get_correlation_matrix_neb_cif("cif_files/*.cif")

print("Suceeded")
