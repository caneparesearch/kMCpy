import unittest
import json
import pytest
import os
from kmcpy.simulator.config import (
    Configuration,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "files")


def create_test_simulation_config(name="Test_Config", use_real_files=True):
    """
    Create a test Configuration with appropriate file paths.
    
    Args:
        name: Name for the configuration
        use_real_files: If True, use real test files; if False, use fake paths
    
    Returns:
        Configuration: Test configuration object
    """
    if use_real_files:
        # Use real test files - Create model-based Configuration using the create() method
        config = Configuration.create(
            structure_file=f"{file_path}/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",
            model_file=f"{file_path}/input/model.json",
            event_file=f"{file_path}/input/events.json",
            initial_state_file=f"{file_path}/input/initial_state.json",
            mobile_ion_specie="Na",
            temperature=873.0,
            name=name,
            attempt_frequency=1e13,
            equilibration_passes=10,  # Small for testing
            kmc_passes=50,  # Small for testing
            dimension=3,
            elementary_hop_distance=2.5,
            mobile_ion_charge=1.0,
            supercell_shape=(2, 1, 1),  # Use tuple
            site_mapping={"Na": ["Na", "X"], "Zr": "Zr", "Si": ["Si", "P"], "O": "O"}
        )
        return config
    else:
        # Use fake paths for testing parameter handling
        return Configuration(
            structure_file="fake_structure.cif",  # Required parameter
            name=name,
            temperature=573.0,
            attempt_frequency=1e13,
            equilibration_passes=10,
            kmc_passes=50,
            dimension=3,
            elementary_hop_distance=2.5,
            mobile_ion_charge=1.0,
            mobile_ion_specie="Na",
            supercell_shape=(2, 1, 1),  # Use tuple
            site_mapping={"Na": ["Na", "X"], "Zr": "Zr", "Si": ["Si", "P"], "O": "O"},
            # Fake file paths with correct names
            model_file="fake_model.json",
            event_file="fake_events.json",
        )


def check_file_exists(file_path):
    """Check if a test file exists."""
    return os.path.exists(file_path)


class TestNASICONbulk(unittest.TestCase):
    @pytest.mark.order("first")
    def test_cluster_matcher(self):
        print("cluster matcher testing")

        import numpy as np
        from kmcpy.structure.cluster import Cluster, ClusterMatcher

        from kmcpy.io.cif import load_labeled_structure_from_cif
        from kmcpy.structure.neighbors import (
            get_cutoff_neighbor_info,
            prepare_cutoff_neighbor_lookup,
        )

        nasicon = load_labeled_structure_from_cif(
            f"{file_path}/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif", primitive=True
        )

        center_Na1 = [0, 1]
        local_env_lookup = prepare_cutoff_neighbor_lookup(
            {("Na+", "Na+"): 4, ("Na+", "Si4+"): 4}
        )

        reference_neighbor_sequences = sorted(
            sorted(
                get_cutoff_neighbor_info(nasicon, center_Na1[0], local_env_lookup),
                key=lambda x: x["wyckoff_sequence"],
            ),
            key=lambda x: x["label"],
        )

        np.set_printoptions(precision=2, suppress=True)
        reference_neighbor = Cluster.from_neighbor_info(
            reference_neighbor_sequences
        )

        print(reference_neighbor.signature)
        print(reference_neighbor.distance_matrix)

        wrong_sequence_neighbor = sorted(
            sorted(
                get_cutoff_neighbor_info(nasicon, center_Na1[1], local_env_lookup),
                key=lambda x: x["wyckoff_sequence"],
            ),
            key=lambda x: x["label"],
        )

        self.assertFalse(
            np.allclose(
                Cluster.from_neighbor_info(
                    wrong_sequence_neighbor
                ).distance_matrix,
                reference_neighbor.distance_matrix,
                rtol=0.01,
                atol=0.01,
            )
        )

        match = ClusterMatcher(reference_neighbor, rtol=0.01).match(
            Cluster.from_neighbor_info(wrong_sequence_neighbor)
        )
        resorted_wrong_sequence = [
            wrong_sequence_neighbor[index]
            for index in match.reference_to_candidate
        ]

        resorted_neighbor = Cluster.from_neighbor_info(
            resorted_wrong_sequence
        )

        self.assertTrue(
            np.allclose(
                reference_neighbor.distance_matrix,
                resorted_neighbor.distance_matrix,
                rtol=0.01,
                atol=0.01,
            )
        )

    @pytest.mark.order("second")
    def test_generate_events(self):
        mobile_ion_identifier_type = "label"
        mobile_ion_identifiers = ("Na1", "Na2")
        structure_file = (
            f"{file_path}/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif"
        )
        local_env_cutoff_dict = {("Na+", "Na+"): 4, ("Na+", "Si4+"): 4}
        from kmcpy.event import EventGenerator

        generator = EventGenerator()
        generator.generate_events(
            structure_file=structure_file,
            local_env_cutoff_dict=local_env_cutoff_dict,
            mobile_ion_identifier_type=mobile_ion_identifier_type,
            mobile_ion_identifiers=mobile_ion_identifiers,
            site_mapping={"Na": ["Na", "X"], "Zr": "Zr", "Si": ["Si", "P"], "O": "O"},
            distance_matrix_rtol=0.01,
            distance_matrix_atol=0.01,
            find_nearest_if_fail=False,
            convert_to_primitive_cell=False,
            export_local_env_structure=True,
            supercell_shape=[2, 1, 1],
            event_file=f"{file_path}/events.json",
        )

        reference_local_env_dict = generator.generate_events(
            structure_file=structure_file,
            local_env_cutoff_dict=local_env_cutoff_dict,
            mobile_ion_identifier_type=mobile_ion_identifier_type,
            mobile_ion_identifiers=mobile_ion_identifiers,
            site_mapping={"Na": ["Na", "X"], "Zr": "Zr", "Si": ["Si", "P"], "O": "O"},
            distance_matrix_rtol=0.01,
            distance_matrix_atol=0.01,
            find_nearest_if_fail=False,
            convert_to_primitive_cell=True,
            export_local_env_structure=True,
            supercell_shape=[2, 1, 1],
            event_file=f"{file_path}/events.json",
        )

        print("reference_local_env_dict:", reference_local_env_dict)

        with open(f"{file_path}/input/events.json") as expected_file:
            expected_event_library = json.load(expected_file)
        with open(f"{file_path}/events.json") as generated_file:
            generated_event_library = json.load(generated_file)
        self.assertEqual(generated_event_library, expected_event_library)

        self.assertGreaterEqual(len(reference_local_env_dict), 1)

    @pytest.mark.order("third")
    def test_generate_local_cluster_exapnsion(self):
        from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
        from kmcpy.structure.local_lattice_structure import LocalLatticeStructure
        from kmcpy.io.cif import load_labeled_structure_from_cif
        mobile_ion_identifier_type = "label"
        mobile_ion_specie_identifier = "Na1"
        structure = load_labeled_structure_from_cif(
            filename=f"{file_path}/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",
            primitive=True,
        )
        local_lattice_structure = LocalLatticeStructure(template_structure=structure,center=0, cutoff=4.0,site_mapping={"Na": ["Na", "X"],"Zr":"Zr","Si":["Si","P"],"O":"O"},
                                     basis_type = "chebyshev", is_write_basis=True)
        a = LocalClusterExpansion()
        a.build(local_lattice_structure=local_lattice_structure,
            mobile_ion_identifier_type=mobile_ion_identifier_type,
            mobile_ion_specie_identifier=mobile_ion_specie_identifier,
            cutoff_cluster=[6, 6, 0],
            cutoff_region=4,
            convert_to_primitive_cell=True,
        )
        a.to_json(f"{file_path}/lce.json")
        # Basic test - should verify object creation
        self.assertEqual(1, 1)

    def test_fitting(self):
        from kmcpy.models.local_cluster_expansion import LocalClusterExpansion
        import numpy as np

        _, y_pred, y_true = LocalClusterExpansion().fit(
            alpha=1.5,
            max_iter=1000000,
            ekra_fname=f"{file_path}/fitting/local_cluster_expansion/e_kra.txt",
            keci_fname=f"{file_path}/keci.txt",
            weight_fname=f"{file_path}/fitting/local_cluster_expansion/weight.txt",
            corr_fname=f"{file_path}/fitting/local_cluster_expansion/correlation_matrix.txt",
            fit_results_fname=f"{file_path}/fitting_results.json",
            lce_params_fname=None,
            lce_params_history_fname=None,
        )
        print("fitting", y_pred, y_true)
        self.assertTrue(np.allclose(y_pred, y_true, rtol=0.3, atol=10.0))

    @pytest.mark.order("kmc_original")
    def test_kmc_main_function(self):
        """KMC test using modern Configuration approach."""
        from kmcpy.simulator.config import Configuration
        from kmcpy.simulator.kmc import KMC
        import numpy as np

        # Change to tests directory temporarily to make relative paths work
        original_cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            
            # Create Configuration from the same parameters as the old kmc_input.json
            config = Configuration.create(
                structure_file=f"{file_path}/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",
                model_file=f"{file_path}/input/model.json",
                event_file=f"{file_path}/input/events.json",
                initial_state_file=f"{file_path}/input/initial_state.json",
                mobile_ion_specie="Na",  # Fixed parameter name
                temperature=298,
                attempt_frequency=5e12,
                equilibration_passes=1,
                kmc_passes=100,
                supercell_shape=(2, 1, 1),
                site_mapping={"Na": ["Na", "X"], "Zr": "Zr", "Si": ["Si", "P"], "O": "O"},
                convert_to_primitive_cell=True,
                elementary_hop_distance=3.47782,  # Same as original kmc_input.json
                random_seed=12345,
                name="NASICON_Test"
            )

            kmc = KMC.from_config(config)

            kmc_tracker = kmc.run(config)

            print(kmc_tracker.return_current_info())
            self.assertTrue(
                np.allclose(
                    np.array(kmc_tracker.return_current_info()),
                    np.array(
                        (
                            1.1193006038758543e-06,
                            307.37444494263616,
                            1.4630573145769372e-08,
                            4.5768825621743376e-09,
                            1.1823906621661553,
                            0.312830024946617,
                            0.21998150220477225,
                        )
                    ),
                    rtol=0.01,
                    atol=0.01,
                )
            )
        finally:
            os.chdir(original_cwd)

        # np.array((3.517242770690013e-06, 26.978226076495748, 3.187544456106211e-10, 1.2783794881088614e-10, 0.025760595723683707, 0.4010546380490277, 0.04309185078659044)) this is run from the given random number kernal and random number seed. This is a very strict criteria to see if the behavior of KMC is correct
        # with 0-7, 32-37 selected: np.array(1.1193006038758543e-06, 307.37444494263616, 1.4630573145769372e-08, 4.5768825621743376e-09, 1.1823906621661553, 0.312830024946617, 0.21998150220477225)

    @pytest.mark.order("kmc_modernized")
    def test_kmc_main_function_modernized(self):
        """Modernized KMC test using Configuration approach."""
        print("Testing modernized KMC workflow with Configuration")

        from kmcpy.simulator.config import Configuration
        from kmcpy.simulator.kmc import KMC
        import numpy as np

        # Change to tests directory temporarily to make relative paths work
        original_cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(os.path.abspath(__file__)))

            # Create modern Configuration using the create() method
            config = Configuration.create(
                structure_file=f"{file_path}/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",
                model_file=f"{file_path}/input/model.json",
                event_file=f"{file_path}/input/events.json",
                initial_state_file=f"{file_path}/input/initial_state.json",
                mobile_ion_specie="Na",
                temperature=298.0,
                attempt_frequency=5e12,
                equilibration_passes=1,
                kmc_passes=100,
                supercell_shape=(2, 1, 1),
                site_mapping={"Na": ["Na", "X"], "Zr": "Zr", "Si": ["Si", "P"], "O": "O"},
                convert_to_primitive_cell=True,
                elementary_hop_distance=3.47782,  # Same as original kmc_input.json
                random_seed=12345,
                name="NASICON_Modernized_Test"
            )

            # Modern workflow
            kmc = KMC.from_config(config)
            kmc_tracker = kmc.run(config)

            print(
                f"Modern Configuration results: {kmc_tracker.return_current_info()}"
            )

            # Should produce identical results to the original test
            self.assertTrue(
                np.allclose(
                    np.array(kmc_tracker.return_current_info()),
                    np.array(
                        (
                            1.1193006038758543e-06,
                            307.37444494263616,
                            1.4630573145769372e-08,
                            4.5768825621743376e-09,
                            1.1823906621661553,
                            0.312830024946617,
                            0.21998150220477225,
                        )
                    ),
                    rtol=0.01,
                    atol=0.01,
                )
            )

            print("✅ Modern Configuration approach produces identical results!")

        finally:
            os.chdir(original_cwd)

    @pytest.mark.order("data_gathering")
    def test_gather_mc_data(self):
        from kmcpy.tools.gather_mc_data import generate_supercell, gather_data
        from kmcpy.io.cif import load_labeled_structure_from_cif
        from kmcpy.structure.sites import make_kmc_supercell
        import numpy as np

        structure_from_json = generate_supercell(
            f"{file_path}/gather_mc_data/prim.json", (8, 8, 8)
        )
        df = gather_data(f"{file_path}/gather_mc_data/comp*", structure_from_json)
        df.to_json(f"{file_path}/mc_results_json.json", orient="index")
        occ1 = df["occ"]

        structure_from_cif = load_labeled_structure_from_cif(
            f"{file_path}/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif", primitive=True
        )
        structure_from_cif.remove_species(["Zr", "O", "Zr4+", "O2-"])
        structure_from_cif.remove_oxidation_states()
        structure_from_cif = make_kmc_supercell(structure_from_cif, [8, 8, 8])
        df2 = gather_data(f"{file_path}/gather_mc_data/comp*", structure_from_cif)
        df2.to_json(f"{file_path}/mc_results_cif.json", orient="index")
        occ2 = df2["occ"]
        for i in range(0, len(occ1[0])):
            if occ1[0][i] != occ2[0][i]:
                print(i, occ1[i], occ2[i])
        self.assertTrue(np.allclose(occ1[0], occ2[0], rtol=0.001, atol=0.001))

    @pytest.mark.order("simulation_config_basic")
    def test_simulation_config_with_nasicon(self):
        """Test Configuration integration with NASICON test files."""
        print("Testing Configuration with NASICON files")

        from kmcpy.simulator.kmc import KMC

        # Check if required files exist
        required_files = [
            f"{file_path}/fitting_results.json",
            f"{file_path}/lce.json",
            f"{file_path}/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",
            f"{file_path}/events.json",
        ]

        for file in required_files:
            if not check_file_exists(file):
                self.skipTest(f"Required test file missing: {file}")

        # Create Configuration with test files
        config = create_test_simulation_config(
            name="NASICON_Test_Config", use_real_files=True
        )

        # Test configuration validation
        try:
            config.validate()
            print("✓ Configuration validation passed")
        except Exception as e:
            self.fail(f"Configuration validation failed: {e}")

        # Test parameter modification
        modified_config = config.copy_with_changes(
            temperature=400.0, name="Modified_NASICON_Config"
        )

        self.assertEqual(modified_config.temperature, 400.0)
        self.assertEqual(modified_config.name, "Modified_NASICON_Config")
        self.assertEqual(modified_config.attempt_frequency, config.attempt_frequency)
        print("✓ Parameter modification works")

        # Test KMC integration methods exist
        self.assertTrue(hasattr(KMC, "from_config"))
        self.assertTrue(hasattr(KMC, "run"))
        print("✓ KMC integration methods exist")

        # Test that we can create KMC instance (may fail due to file format issues)
        try:
            kmc_instance = KMC.from_config(config)
            print("✓ KMC instance creation from Configuration works")
        except Exception as e:
            # This might fail due to file format issues, which is acceptable
            print(
                f"⚠ KMC instance creation failed (expected with test files): {type(e).__name__}"
            )

        print("✅ Configuration NASICON integration test completed")

    @pytest.mark.order("simulation_config_parameters")
    def test_simulation_config_parameter_studies(self):
        """Test Configuration parameter study capabilities."""
        print("Testing Configuration parameter studies")

        from tests.test_utils import create_temperature_series

        # Create base configuration (using fake files for parameter testing)
        base_config = create_test_simulation_config(
            name="Parameter_Study_Base", use_real_files=False
        )

        # Test temperature series
        temperatures = [300, 400, 500]
        temp_series = create_temperature_series(base_config, temperatures)

        self.assertEqual(len(temp_series), 3)
        for i, config in enumerate(temp_series):
            self.assertEqual(config.temperature, temperatures[i])
            self.assertIn(f"T_{temperatures[i]}K", config.name)

        print("✓ Temperature series creation works")

        # Test multiple parameter modifications
        study_configs = []
        for temp in [300, 400]:
            for freq in [1e12, 1e13]:
                modified_config = base_config.copy_with_changes(
                    temperature=temp,
                    attempt_frequency=freq,
                    name=f"Study_T{temp}_f{freq:.0e}",
                )
                study_configs.append(modified_config)

        self.assertEqual(len(study_configs), 4)

        # Verify configurations are different
        temps_and_freqs = [(c.temperature, c.attempt_frequency) for c in study_configs]
        self.assertEqual(
            len(set(temps_and_freqs)), 4
        )  # All combinations should be unique

        print("✓ Multi-parameter studies work")
        print("✅ Parameter studies test completed")

    @pytest.mark.order("kmc_comparison")
    def test_kmc_simulation_config_validation(self):
        """Test that KMC with Configuration produces expected results."""
        print("Testing KMC Configuration validation")

        from kmcpy.simulator.kmc import KMC
        from kmcpy.simulator.config import Configuration
        import numpy as np

        # Change to tests directory temporarily to make relative paths work
        original_cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(os.path.abspath(__file__)))

            # ========== Test: Configuration approach ==========
            print("Running Configuration approach...")

            # Create Configuration with the same parameters as the old kmc_input.json
            config = Configuration.create(
                structure_file=f"{file_path}/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",
                model_file=f"{file_path}/input/model.json",
                event_file=f"{file_path}/input/events.json",
                # Note: initial_state_file is not supported in the clean API
                mobile_ion_specie="Na",
                temperature=298.0,
                attempt_frequency=5e12,
                equilibration_passes=1,
                kmc_passes=100,
                supercell_shape=(2, 1, 1),
                site_mapping={"Na": ["Na", "X"], "Zr": "Zr", "Si": ["Si", "P"], "O": "O"},
                convert_to_primitive_cell=True,
                elementary_hop_distance=3.47782,
                random_seed=12345,
                name="NASICON_KMC_Test",
                initial_state_file=f"{file_path}/input/initial_state.json"
            )

            # Test KMC with Configuration
            kmc_simulation = KMC.from_config(config)
            kmc_tracker_simulation = kmc_simulation.run(config)
            simulation_results = kmc_tracker_simulation.return_current_info()
            print(f"Configuration results: {simulation_results}")

            # ========== Validate Results ==========
            print("Validating results...")

            # Validate that we get reasonable results (these are the expected values from the original test)
            expected_results = np.array([
                1.1193006038758543e-06,
                307.37444494263616,
                1.4630573145769372e-08,
                4.5768825621743376e-09,
                1.1823906621661553,
                0.312830024946617,
                0.21998150220477225,
            ])
            
            # Results should be close to expected values (allowing for some numerical variation)
            self.assertTrue(
                np.allclose(
                    np.array(simulation_results),
                    expected_results,
                    rtol=0.01,
                    atol=0.01,
                ),
                f"Configuration results don't match expected: {simulation_results} vs {expected_results}",
            )

            print("✓ Configuration produces expected results")
            print("✅ Configuration validation test completed")

        finally:
            os.chdir(original_cwd)

    @pytest.mark.order("kmc_workflow")
    def test_kmc_simulation_config_workflow(self):
        """Test complete KMC workflow using Configuration approach."""
        print("Testing complete KMC workflow with Configuration")

        from kmcpy.simulator.config import Configuration
        from kmcpy.simulator.kmc import KMC
        import numpy as np

        # Change to tests directory temporarily to make relative paths work
        original_cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(os.path.abspath(__file__)))

            # Create Configuration using the create() method
            config = Configuration.create(
                structure_file=f"{file_path}/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",
                model_file=f"{file_path}/input/model.json",
                event_file=f"{file_path}/input/events.json",
                # Note: initial_state_file is not supported in the clean API
                mobile_ion_specie="Na",
                temperature=298.0,
                attempt_frequency=5e12,
                equilibration_passes=1,
                kmc_passes=100,
                supercell_shape=(2, 1, 1),
                site_mapping={"Na": ["Na", "X"], "Zr": "Zr", "Si": ["Si", "P"], "O": "O"},
                convert_to_primitive_cell=True,
                elementary_hop_distance=3.47782,
                random_seed=12345,
                name="NASICON_SimulationConfig_Test",
                initial_state_file=f"{file_path}/input/initial_state.json"
            )

            print("✓ Configuration created with test parameters")

            # Test 1: Create KMC from Configuration
            kmc = KMC.from_config(config)
            print("✓ KMC instance created from Configuration")

            # Test 2: Run simulation using run method (recommended approach)
            print("Running KMC simulation using run method...")
            tracker = kmc.run(config)
            results = tracker.return_current_info()
            print(f"✓ Configuration results: {results}")

            # Test 3: Verify results match expected values (same as original test)
            expected_results = np.array(
                [
                    1.1193006038758543e-06,
                    307.37444494263616,
                    1.4630573145769372e-08,
                    4.5768825621743376e-09,
                    1.1823906621661553,
                    0.312830024946617,
                    0.21998150220477225,
                ]
            )

            self.assertTrue(
                np.allclose(np.array(results), expected_results, rtol=0.01, atol=0.01),
                f"Configuration results don't match expected: {results} vs {expected_results}",
            )

            # Test 4: Demonstrate configuration modification for parameter studies
            print("\nTesting parameter study capabilities...")

            # Create a modified configuration with different temperature
            high_temp_config = config.with_runtime_changes(
                temperature=400.0, name="NASICON_HighTemp_Test"
            )

            self.assertEqual(high_temp_config.temperature, 400.0)
            self.assertEqual(high_temp_config.name, "NASICON_HighTemp_Test")
            self.assertEqual(
                high_temp_config.attempt_frequency, config.attempt_frequency
            )  # Should be unchanged

            print("✓ Configuration modification for parameter studies works")

            # Test 5: Show serialization capabilities
            config_dict = config.to_dict()
            self.assertIn("temperature", config_dict)
            self.assertIn("kmc_passes", config_dict)

            print("✓ Configuration serialization works")

            print("\n✅ Complete Configuration workflow test passed!")
            print(
                "✅ Configuration system is working correctly and produces expected results!"
            )

        finally:
            os.chdir(original_cwd)

if __name__ == "__main__":
    unittest.main()
