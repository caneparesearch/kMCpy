# script for initializing the local cluster expansion to predict barriers (mean barrier)

from kmcpy.model import LocalClusterExpansion
mobile_ion_identifier_type="specie" #"specie"
mobile_ion_specie_1_identifier="Na"

# cutoff_region = 6.2 # Check to match in Vesta # Seems the Na--Na distance are about 6 Å
cif_file = "lce_input_files/full_24d.cif"

cutoff_region = 5.1 #5.1
# cutoff_cluster = [3.5,3.5,3.5] # default
cutoff_cluster = [3.6, 3.6, 3.6]
cutoff_cluster = [4,4,4]
lce=LocalClusterExpansion() # api?
lce.initialization(
    center_frac_coord=[0.875, 0.25, 0.125],
    mobile_ion_identifier_type=mobile_ion_identifier_type,
    mobile_ion_specie_1_identifier=mobile_ion_specie_1_identifier,
    cutoff_cluster=cutoff_cluster,
    cutoff_region=cutoff_region,
    template_cif_fname=cif_file,
    species_to_be_removed=["Fe", "C", "N"],
    convert_to_primitive_cell=False # It is already the primitive cell
    )
lce.to_json("lce_output_files/lce.json")
lce.to_json("lce_output_files/lce_site.json")

# # print(cif_file)
# # Writing correlation-matrix txt file
lce.get_correlation_matrix_neb_cif("lce_input_files/two_structure_input/*.cif")

print("Suceeded")

print("Moving files")

import os
import shutil

# Define the target directory
target_dir = "lce_output_files"

# Create the target directory if it doesn't exist
# os.makedirs(target_dir, exist_ok=True)½

# List of files to move
files_to_move = ["debug.log", "correlation_matrix.txt", "occupation.txt"]

# Move each file
for file_name in files_to_move:
    if os.path.exists(file_name):
        shutil.move(file_name, os.path.join(target_dir, file_name))
        print(f"Moved {file_name} to {target_dir}/")
    else:
        print(f"File not found: {file_name}")

