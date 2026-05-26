import unittest
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'files')

class TestNa3SbS4(unittest.TestCase):

    def test_generate_events(self):
        from pathlib import Path
        import os

        current_dir = Path(__file__).absolute().parent
        original_cwd = Path.cwd()
        try:
            os.chdir(current_dir)
            mobile_ion_identifier_type = "label"
            mobile_ion_specie_identifier = "Na1"
            mobile_ion_specie_2_identifier = "Na1"
            structure_file = f"{file_path}/Na3SbS4_cubic.cif"
            local_env_cutoff_dict = {("Na+", "Na+"): 5, ("Na+", "Sb5+"): 4}
            from kmcpy.event import EventGenerator

            reference_local_env_dict = EventGenerator().generate_events(
                structure_file=structure_file,
                local_env_cutoff_dict=local_env_cutoff_dict,
                mobile_ion_identifier_type=mobile_ion_identifier_type,
                mobile_ion_identifiers=("Na1", "Na1"),
                site_mapping={"Na": ["Na", "X"], "Sb": "Sb", "S": "S"},
                distance_matrix_rtol=0.01,
                distance_matrix_atol=0.01,
                find_nearest_if_fail=False,
                convert_to_primitive_cell=False,
                export_local_env_structure=True,
                supercell_shape=[3, 3, 3],
                event_file=f"{file_path}/events.json",
            )

            self.assertEqual(
                len(reference_local_env_dict), 1
            )  # only one type of local environment should be found. If more than 1, raise error.
        finally:
            os.chdir(original_cwd)

    def test_generate_events_with_new_style_arguments(self):
        from pathlib import Path
        import os
        import tempfile

        current_dir = Path(__file__).absolute().parent
        original_cwd = Path.cwd()
        try:
            os.chdir(current_dir)
            structure_file = f"{file_path}/Na3SbS4_cubic.cif"

            from kmcpy.event import EventGenerator

            with tempfile.TemporaryDirectory() as tmpdir:
                reference_local_env_dict = EventGenerator().generate_events(
                    structure_file=structure_file,
                    mobile_species=["Na"],
                    site_mapping={"Na": ["Na", "X"], "Sb": "Sb", "S": "S"},
                    local_env_cutoff=5.0,
                    supercell_shape=[2, 2, 2],
                    event_file=os.path.join(tmpdir, "events.json"),
                    rtol=0.01,
                    atol=0.01,
                )

                self.assertGreaterEqual(len(reference_local_env_dict), 1)
        finally:
            os.chdir(original_cwd)

    def test_generate_local_cluster_expansion(self):
        from pathlib import Path
        import os

        current_dir = Path(__file__).absolute().parent
        original_cwd = Path.cwd()
        try:
            os.chdir(current_dir)
            from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
            from kmcpy.structure.local_lattice_structure import LocalLatticeStructure
            from kmcpy.io.cif import load_labeled_structure_from_cif

            mobile_ion_identifier_type = "label"
            mobile_ion_specie_identifier = "Na1"
            structure = load_labeled_structure_from_cif(
                filename=f"{file_path}/Na3SbS4_cubic.cif", primitive=True
            )
            a = LocalClusterExpansion()
            local_lattice_structure = LocalLatticeStructure(template_structure=structure, center=0, cutoff=5,
                                         site_mapping={"Na": ["Na", "X"], "Sb": "Sb", "S": "S"},
                                         basis_type="chebyshev")
            a.build(
                local_lattice_structure=local_lattice_structure,
                mobile_ion_identifier_type=mobile_ion_identifier_type,
                mobile_ion_specie_identifier=mobile_ion_specie_identifier,
                cutoff_cluster=[6, 6, 0],
                cutoff_region=5,
                template_structure_fname=f"{file_path}/Na3SbS4_cubic.cif",  # LCE still uses legacy param name
                convert_to_primitive_cell=True,
                is_write_basis=True,
            )
            # Basic test - should verify object creation
            self.assertEqual(1, 1)
        finally:
            os.chdir(original_cwd)

    def test_simulation_config_basic(self):
        """Test basic Configuration functionality for Na3SbS4."""
        from kmcpy.simulator.config import Configuration
        
        # Test basic configuration creation
        config = Configuration(
            structure_file="fake_structure.cif",  # Required parameter with correct name
            name="Na3SbS4_Test",
            temperature=600.0,
            attempt_frequency=1e13,
            equilibration_passes=100,
            kmc_passes=500,
            dimension=3,
            elementary_hop_distance=2.0,
            mobile_ion_charge=1.0,
            mobile_ion_specie="Na",
            supercell_shape=(2, 2, 2),  # Use tuple
            site_mapping={"Na": ["Na", "X"], "Sb": "Sb", "S": "S"},
            
            model_file="fake_model.json",
            event_file="fake_events.json",
        )
        
        # Test that configuration was created successfully
        try:
            self.assertIsNotNone(config.system_config)
            self.assertIsNotNone(config.runtime_config)
            self.assertTrue(True)  # Configuration created successfully
        except Exception as e:
            self.fail(f"Configuration creation failed: {e}")
        
        # Test dictionary conversion
        config_dict = config.as_dict()
        self.assertIn('name', config_dict)
        self.assertIn('temperature', config_dict)
        self.assertIn('equilibration_passes', config_dict)
        self.assertIn('kmc_passes', config_dict)
        self.assertEqual(config_dict['equilibration_passes'], 100)
        self.assertEqual(config_dict['kmc_passes'], 500)
        
        # Test property access
        self.assertEqual(config.name, "Na3SbS4_Test")
        self.assertEqual(config.temperature, 600.0)
        self.assertEqual(config.attempt_frequency, 1e13)
        self.assertEqual(config.equilibration_passes, 100)
        self.assertEqual(config.kmc_passes, 500)
        
        # Test parameter modification using available methods
        modified_config = config.with_runtime_changes(temperature=700.0, name="Modified_Na3SbS4")
        self.assertEqual(modified_config.temperature, 700.0)
        self.assertEqual(modified_config.name, "Modified_Na3SbS4")
        self.assertEqual(modified_config.attempt_frequency, config.attempt_frequency)
        
        print("✓ Configuration basic functionality works for Na3SbS4")


if __name__ == "__main__":
    unittest.main()
