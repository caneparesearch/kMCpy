mobile_ion_identifier_type = "specie" # "label"
mobile_ion_specie_1_identifier = "Na+"
mobile_ion_specie_2_identifier = "Na+"
prim_cif_name = "lce_input_files/full_24d.cif"
local_env_cutoff_dict = {
        ("Na+", "Na+"): 3.6
        # ("Na+", "Si4+"): 4
        }
from kmcpy.event_generator import generate_events3

generate_events3(
    prim_cif_name=prim_cif_name,
    local_env_cutoff_dict=local_env_cutoff_dict,
    mobile_ion_identifier_type=mobile_ion_identifier_type,
    mobile_ion_specie_1_identifier=mobile_ion_specie_1_identifier,
    mobile_ion_specie_2_identifier=mobile_ion_specie_2_identifier,
    species_to_be_removed=["Fe", "N", "C", "Fe3+", "N3-", "C+"],
    distance_matrix_rtol=0.01,
    distance_matrix_atol=0.01,
    find_nearest_if_fail=False,
    convert_to_primitive_cell=False,
    export_local_env_structure=True,
    supercell_shape=[1, 1, 1],
    event_fname="events_output_files/events.json",
    event_kernal_fname="events_output_files/event_kernal.csv",
    verbosity="INFO",
)

print("Moving files")

import os
import shutil

# Define the target directory
target_dir = "events_output_files"

# Create the target directory if it doesn't exist
#os.makedirs(target_dir, exist_ok=True)

# List of files to move
files_to_move = ["0th_reference_local_env.cif", "debug.log"]

# Move each file
for file_name in files_to_move:
    if os.path.exists(file_name):
        shutil.move(file_name, os.path.join(target_dir, file_name))
        print(f"Moved {file_name} to {target_dir}/")
    else:
        print(f"File not found: {file_name}")
