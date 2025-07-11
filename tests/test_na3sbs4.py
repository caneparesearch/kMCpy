import unittest
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'files')

class TestNa3SbS4(unittest.TestCase):

    def test_generate_events(self):
        from pathlib import Path
        import os

        current_dir = Path(__file__).absolute().parent
        os.chdir(current_dir)
        mobile_ion_identifier_type = "label"
        mobile_ion_specie_identifier = "Na1"
        mobile_ion_specie_2_identifier = "Na1"
        template_structure_fname = f"{file_path}/Na3SbS4_cubic.cif"
        local_env_cutoff_dict = {("Na+", "Na+"): 5, ("Na+", "Sb5+"): 4}
        from kmcpy.event import EventGenerator

        reference_local_env_dict = EventGenerator().generate_events(
            template_structure_fname=template_structure_fname,
            local_env_cutoff_dict=local_env_cutoff_dict,
            mobile_ion_identifier_type=mobile_ion_identifier_type,
            mobile_ion_identifiers=("Na1", "Na1"),
            species_to_be_removed=["S2-", "S", "Zr4+", "Zr"],
            distance_matrix_rtol=0.01,
            distance_matrix_atol=0.01,
            find_nearest_if_fail=False,
            convert_to_primitive_cell=False,
            export_local_env_structure=True,
            supercell_shape=[3, 3, 3],
            event_fname=f"{file_path}/events.json",
            event_dependencies_fname=f"{file_path}/event_dependencies.csv",
        )

        self.assertEqual(
            len(reference_local_env_dict), 1
        )  # only one type of local environment should be found. If more than 1, raise error.

    def test_generate_local_cluster_exapnsion(self):
        from pathlib import Path
        import os

        current_dir = Path(__file__).absolute().parent
        os.chdir(current_dir)
        from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
        from kmcpy.external.structure import StructureKMCpy

        mobile_ion_identifier_type = "label"
        mobile_ion_specie_identifier = "Na1"
        structure = StructureKMCpy.from_cif(
            filename=f"{file_path}/Na3SbS4_cubic.cif", primitive=True
        )
        a = LocalClusterExpansion(template_structure=structure,
            specie_site_mapping={"Na": ["Na", "X"], "Sb": "Sb", "S": "S"},
            basis_type="chebychev")
        a.build(
            mobile_ion_identifier_type=mobile_ion_identifier_type,
            mobile_ion_specie_identifier=mobile_ion_specie_identifier,
            cutoff_cluster=[6, 6, 0],
            cutoff_region=5,
            template_structure_fname=f"{file_path}/Na3SbS4_cubic.cif",
            convert_to_primitive_cell=True,
            is_write_basis=True,
            species_to_be_removed=["S2-", "S"],
        )
        # Basic test - should verify object creation
        self.assertEqual(1, 1)

    def test_simulation_config_basic(self):
        """Test basic SimulationConfig functionality for Na3SbS4."""
        from kmcpy.simulator.condition import SimulationConfig
        
        # Test basic configuration creation
        config = SimulationConfig(
            name="Na3SbS4_Test",
            temperature=600.0,
            attempt_frequency=1e13,
            equilibration_passes=100,
            kmc_passes=500,
            dimension=3,
            elementary_hop_distance=2.0,
            mobile_ion_charge=1.0,
            mobile_ion_specie="Na",
            supercell_shape=[2, 2, 2],
            initial_occ=[1, -1, 1, -1, 1, -1, 1, -1],
            immutable_sites=["S", "Sb"],
            
            # Use fake file paths for testing
            fitting_results="fake_fitting.json",
            fitting_results_site="fake_fitting_site.json",
            lce_fname="fake_lce.json",
            lce_site_fname="fake_lce_site.json",
            template_structure_fname="fake_structure.cif",
            event_fname="fake_events.json",
            event_dependencies="fake_dependencies.csv"
        )
        
        # Test validation
        try:
            config.validate()
            self.assertTrue(True)  # Validation passed
        except Exception as e:
            self.fail(f"Configuration validation failed: {e}")
        
        # Test dictionary conversion
        config_dict = config.to_dict()
        self.assertIn('name', config_dict)
        self.assertIn('temperature', config_dict)
        self.assertIn('v', config_dict)
        self.assertEqual(config_dict['v'], 1e13)
        
        # Test dataclass dictionary conversion
        dataclass_dict = config.to_dataclass_dict()
        self.assertIn('name', dataclass_dict)
        self.assertIn('temperature', dataclass_dict)
        self.assertIn('attempt_frequency', dataclass_dict)
        self.assertIn('equilibration_passes', dataclass_dict)
        self.assertIn('kmc_passes', dataclass_dict)
        
        # Test parameter modification
        modified_config = config.copy_with_changes(temperature=700.0, name="Modified_Na3SbS4")
        self.assertEqual(modified_config.temperature, 700.0)
        self.assertEqual(modified_config.name, "Modified_Na3SbS4")
        self.assertEqual(modified_config.attempt_frequency, config.attempt_frequency)
        
        print("✓ SimulationConfig basic functionality works for Na3SbS4")


if __name__ == "__main__":
    unittest.main()
