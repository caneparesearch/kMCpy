import unittest
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = f'{current_dir}/files'

class TestNa3SbS4(unittest.TestCase):

    def test_generate_events(self):
        from pathlib import Path
        import os

        current_dir = Path(__file__).absolute().parent
        os.chdir(current_dir)
        mobile_ion_identifier_type = "label"
        mobile_ion_specie_1_identifier = "Na1"
        mobile_ion_specie_2_identifier = "Na1"
        template_structure_fname = f"{file_path}/Na3SbS4_cubic.cif"
        local_env_cutoff_dict = {("Na+", "Na+"): 5, ("Na+", "Sb5+"): 4}
        from kmcpy.event_generator import generate_events

        reference_local_env_dict = generate_events(
            template_structure_fname=template_structure_fname,
            local_env_cutoff_dict=local_env_cutoff_dict,
            mobile_ion_identifier_type=mobile_ion_identifier_type,
            mobile_ion_specie_1_identifier=mobile_ion_specie_1_identifier,
            mobile_ion_specie_2_identifier=mobile_ion_specie_2_identifier,
            species_to_be_removed=["S2-", "S", "Zr4+", "Zr"],
            distance_matrix_rtol=0.01,
            distance_matrix_atol=0.01,
            find_nearest_if_fail=False,
            convert_to_primitive_cell=False,
            export_local_env_structure=True,
            supercell_shape=[3, 3, 3],
            event_fname=f"{file_path}/events.json",
            event_kernal_fname=f"{file_path}/event_kernal.csv",
        )

        self.assertEqual(
            len(reference_local_env_dict), 1
        )  # only one type of local environment should be found. If more than 1, raise error.

    def test_generate_local_cluster_exapnsion(self):
        from pathlib import Path
        import os

        current_dir = Path(__file__).absolute().parent
        os.chdir(current_dir)
        from kmcpy.model import LocalClusterExpansion

        mobile_ion_identifier_type = "label"
        mobile_ion_specie_1_identifier = "Na1"
        a = LocalClusterExpansion()
        a.initialization(
            mobile_ion_identifier_type=mobile_ion_identifier_type,
            mobile_ion_specie_1_identifier=mobile_ion_specie_1_identifier,
            cutoff_cluster=[6, 6, 0],
            cutoff_region=5,
            template_structure_fname=f"{file_path}/Na3SbS4_cubic.cif",
            convert_to_primitive_cell=True,
            is_write_basis=True,
            species_to_be_removed=["S2-", "S"],
        )
        self.assertEqual(1, 1)


if __name__ == "__main__":
    unittest.main()
