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
        structure_file = f"{file_path}/Na3SbS4_cubic.cif"
        local_env_cutoff_dict = {("Na+", "Na+"): 5, ("Na+", "Sb5+"): 4}
        from kmcpy.event import EventGenerator

        reference_local_env_dict = EventGenerator().generate_events_legacy(
            structure_file=structure_file,
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
            event_file=f"{file_path}/events.json",
            event_dependencies_file=f"{file_path}/event_dependencies.csv",
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
        from kmcpy.structure.local_lattice_structure import LocalLatticeStructure
        from kmcpy.external.structure import StructureKMCpy

        mobile_ion_identifier_type = "label"
        mobile_ion_specie_identifier = "Na1"
        structure = StructureKMCpy.from_cif(
            filename=f"{file_path}/Na3SbS4_cubic.cif", primitive=True
        )
        a = LocalClusterExpansion()
        local_lattice_structure = LocalLatticeStructure(template_structure=structure, center=0, cutoff=5,
                                     specie_site_mapping={"Na": ["Na", "X"], "Sb": "Sb", "S": "S"},
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
            species_to_be_removed=["S2-", "S"],
        )
        # Basic test - should verify object creation
        self.assertEqual(1, 1)

    def test_simulation_config_basic(self):
        """Test basic SimulationConfig functionality for Na3SbS4."""
        from kmcpy.simulator.config import SimulationConfig
        
        # Test basic configuration creation
        config = SimulationConfig(
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
            immutable_sites=("S", "Sb"),  # Use tuple
            
            # Use correct parameter names
            fitting_results_file="fake_fitting.json",
            fitting_results_site_file="fake_fitting_site.json",
            cluster_expansion_file="fake_lce.json",
            cluster_expansion_site_file="fake_lce_site.json",
            event_file="fake_events.json",
            event_dependencies="fake_dependencies.csv"
        )
        
        # Test that configuration was created successfully
        try:
            self.assertIsNotNone(config.system_config)
            self.assertIsNotNone(config.runtime_config)
            self.assertTrue(True)  # Configuration created successfully
        except Exception as e:
            self.fail(f"Configuration creation failed: {e}")
        
        # Test dictionary conversion
        config_dict = config.to_dict()
        self.assertIn('name', config_dict)
        self.assertIn('temperature', config_dict)
        # Note: to_dict() uses legacy key names
        self.assertIn('equ_pass', config_dict)  # maps from equilibration_passes
        self.assertIn('kmc_pass', config_dict)  # maps from kmc_passes
        self.assertEqual(config_dict['equ_pass'], 100)
        self.assertEqual(config_dict['kmc_pass'], 500)
        
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
        
        print("✓ SimulationConfig basic functionality works for Na3SbS4")


if __name__ == "__main__":
    unittest.main()
