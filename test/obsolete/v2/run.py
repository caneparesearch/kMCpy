import unittest
from pathlib import Path
import os


class TestStringMethods(unittest.TestCase):
    from kmcpy.event_generator import generate_events

    convert_to_primitive_cell = True
    current_dir = Path(__file__).absolute().parent
    os.chdir(current_dir)
    # event kernal
    generate_events(
        prim_cif_name="./input/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",
        convert_to_primitive_cell=convert_to_primitive_cell,
        supercell_shape=[2, 1, 1],
        local_env_cutoff_dict={("Na+", "Na+"): 4, ("Na+", "Si4+"): 4},
        event_fname="input/events.json",
        event_kernal_fname="input/event_kernal.csv",
        mobile_ion_specie_1_index_label_or_indices="Na1",
        species_to_be_removed=["Zr4+", "O2-", "O", "Zr"],
        mobile_ion_specie_2_index_atom_label="Na2",
        hacking_arg={1: [18, 20, 19, 22, 21, 23, 108, 110, 109, 112, 111, 113]},
    )

    # model kernal

    from kmcpy.model import LocalClusterExpansion

    a = LocalClusterExpansion()
    a.initialization(
        mobile_ion_specie_1_index="Na1",
        cutoff_cluster=[6, 6, 0],
        cutoff_region=4,
        template_cif_fname="./input/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",
        species_to_be_removed=["Zr4+", "O2-", "O", "Zr"],
        convert_to_primitive_cell=convert_to_primitive_cell,
    )
    a.to_json("input/lce.json")
    a.to_json("input/lce_site.json")

    from kmcpy.io import InputSet
    from kmcpy.kmc import KMC

    inputset = InputSet.from_json("input/test_input_v2.json")

    print(inputset._parameters.keys())
    print(inputset._parameters["initial_state"])

    # initialize global occupation and conditions

    kmc = KMC.from_inputset(inputset=inputset)

    # run kmc
    kmc.run(inputset=inputset)


if __name__ == "__main__":
    unittest.main()
