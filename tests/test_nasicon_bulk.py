import unittest
import pytest
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = f'{current_dir}/files'

class TestNASICONbulk(unittest.TestCase):

    @pytest.mark.order(1)
    def test_neighbor_info_matcher(self):
        print("neighbor info matcher testing")
        
        from kmcpy.event_generator import neighbor_info_matcher
        import numpy as np

        from kmcpy.external.structure import StructureKMCpy
        from kmcpy.external.local_env import CutOffDictNNKMCpy

        nasicon = StructureKMCpy.from_cif(
            f"{file_path}/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif", primitive=True
        )

        center_Na1 = [0, 1]
        local_env_finder = CutOffDictNNKMCpy({("Na+", "Na+"): 4, ("Na+", "Si4+"): 4})

        reference_neighbor_sequences = sorted(
            sorted(
                local_env_finder.get_nn_info(nasicon, center_Na1[0]),
                key=lambda x: x["wyckoff_sequence"],
            ),
            key=lambda x: x["label"],
        )

        np.set_printoptions(precision=2, suppress=True)
        reference_neighbor = neighbor_info_matcher.from_neighbor_sequences(
            neighbor_sequences=reference_neighbor_sequences
        )

        print(reference_neighbor.neighbor_species)
        print(reference_neighbor.distance_matrix)

        wrong_sequence_neighbor = sorted(
            sorted(
                local_env_finder.get_nn_info(nasicon, center_Na1[1]),
                key=lambda x: x["wyckoff_sequence"],
            ),
            key=lambda x: x["label"],
        )

        self.assertFalse(
            np.allclose(
                neighbor_info_matcher.from_neighbor_sequences(
                    neighbor_sequences=wrong_sequence_neighbor
                ).distance_matrix,
                reference_neighbor.distance_matrix,
                rtol=0.01,
                atol=0.01,
            )
        )

        resorted_wrong_sequence = reference_neighbor.brutal_match(
            wrong_sequence_neighbor, rtol=0.01
        )

        resorted_neighbor = neighbor_info_matcher.from_neighbor_sequences(
            neighbor_sequences=resorted_wrong_sequence
        )

        self.assertTrue(
            np.allclose(
                reference_neighbor.distance_matrix,
                resorted_neighbor.distance_matrix,
                rtol=0.01,
                atol=0.01,
            )
        )

    @pytest.mark.order(2)
    def test_generate_events(self):
        mobile_ion_identifier_type = "label"
        mobile_ion_specie_1_identifier = "Na1"
        mobile_ion_specie_2_identifier = "Na2"
        template_structure_fname = f"{file_path}/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif"
        local_env_cutoff_dict = {("Na+", "Na+"): 4, ("Na+", "Si4+"): 4}
        from kmcpy.event_generator import generate_events

        generate_events(
            template_structure_fname=template_structure_fname,
            local_env_cutoff_dict=local_env_cutoff_dict,
            mobile_ion_identifier_type=mobile_ion_identifier_type,
            mobile_ion_specie_1_identifier=mobile_ion_specie_1_identifier,
            mobile_ion_specie_2_identifier=mobile_ion_specie_2_identifier,
            species_to_be_removed=["O2-", "O", "Zr4+", "Zr"],
            distance_matrix_rtol=0.01,
            distance_matrix_atol=0.01,
            find_nearest_if_fail=False,
            convert_to_primitive_cell=False,
            export_local_env_structure=True,
            supercell_shape=[2, 1, 1],
            event_fname=f"{file_path}/events.json",
            event_kernal_fname=f"{file_path}/event_kernal.csv",
        )

        reference_local_env_dict = generate_events(
            template_structure_fname=template_structure_fname,
            local_env_cutoff_dict=local_env_cutoff_dict,
            mobile_ion_identifier_type=mobile_ion_identifier_type,
            mobile_ion_specie_1_identifier=mobile_ion_specie_1_identifier,
            mobile_ion_specie_2_identifier=mobile_ion_specie_2_identifier,
            species_to_be_removed=["O2-", "O", "Zr4+", "Zr"],
            distance_matrix_rtol=0.01,
            distance_matrix_atol=0.01,
            find_nearest_if_fail=False,
            convert_to_primitive_cell=True,
            export_local_env_structure=True,
            supercell_shape=[2, 1, 1],
            event_fname=f"{file_path}/events.json",
            event_kernal_fname=f"{file_path}/event_kernal.csv",
        )

        print("reference_local_env_dict:", reference_local_env_dict)

        self.assertEqual(
            len(reference_local_env_dict), 1
        )  # only one type of local environment should be found. If more than 1, raise error.

    @pytest.mark.order(3)
    def test_generate_local_cluster_exapnsion(self):
        
        from kmcpy.model import LocalClusterExpansion

        mobile_ion_identifier_type = "label"
        mobile_ion_specie_1_identifier = "Na1"
        a = LocalClusterExpansion()
        a.initialization(
            mobile_ion_identifier_type=mobile_ion_identifier_type,
            mobile_ion_specie_1_identifier=mobile_ion_specie_1_identifier,
            cutoff_cluster=[6, 6, 0],
            cutoff_region=4,
            template_structure_fname=f"{file_path}/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif",
            convert_to_primitive_cell=True,
        )
        a.to_json(f"{file_path}/input/lce.json")
        self.assertEqual(1, 1)

    def test_fitting(self):
        from kmcpy.fitting import Fitting
        import numpy as np

        local_cluster_expansion_fit = Fitting()

        y_pred, y_true = local_cluster_expansion_fit.fit(
            alpha=1.5,
            max_iter=1000000,
            ekra_fname=f"{file_path}/fitting/local_cluster_expansion/e_kra.txt",
            keci_fname=f"{file_path}/keci.txt",
            weight_fname=f"{file_path}/fitting/local_cluster_expansion/weight.txt",
            corr_fname=f"{file_path}/fitting/local_cluster_expansion/correlation_matrix.txt",
            fit_results_fname=f"{file_path}/fitting_results.json",
        )
        print("fitting", y_pred, y_true)
        self.assertTrue(np.allclose(y_pred, y_true, rtol=0.3, atol=10.0))

    @pytest.mark.order(4)
    def test_kmc_main_function(self):
        from kmcpy.io import InputSet
        from kmcpy.kmc import KMC
        import numpy as np

        inputset = InputSet.from_json(f"{file_path}/input/kmc_input.json")

        print(inputset._parameters.keys())
        print(inputset._parameters["initial_state"])

        kmc = KMC.from_inputset(inputset)

        kmc_tracker = kmc.run(inputset)

        print(kmc_tracker.return_current_info())
        self.assertTrue(
            np.allclose(
                np.array(kmc_tracker.return_current_info()),
                np.array(
                    (1.1193006038758543e-06, 307.37444494263616, 1.4630573145769372e-08, 4.5768825621743376e-09, 1.1823906621661553, 0.312830024946617, 0.21998150220477225)
                ),
                rtol=0.01,
                atol=0.01,
            )
        )

        # np.array((3.517242770690013e-06, 26.978226076495748, 3.187544456106211e-10, 1.2783794881088614e-10, 0.025760595723683707, 0.4010546380490277, 0.04309185078659044)) this is run from the given random number kernal and random number seed. This is a very strict criteria to see if the behavior of KMC is correct
        # with 0-7, 32-37 selected: np.array(1.1193006038758543e-06, 307.37444494263616, 1.4630573145769372e-08, 4.5768825621743376e-09, 1.1823906621661553, 0.312830024946617, 0.21998150220477225)

    @pytest.mark.order(5)
    def test_kmc_main_function_randomized(self):
        from kmcpy.io import InputSet
        from kmcpy.kmc import KMC
        import numpy as np

        inputset = InputSet.from_json(f"{file_path}/input/kmc_input.json")

        print(inputset._parameters.keys())
        print(inputset._parameters["initial_state"])
        
        inputset.set_parameter("random_seed", np.random.randint(0, 1000000))
        kmc = KMC.from_inputset(inputset)
        
        kmc_tracker = kmc.run(inputset)

        print(kmc_tracker.return_current_info())
        self.assertFalse(
            np.allclose(
                np.array(kmc_tracker.return_current_info()),
                np.array(
                    (1.1193006038758543e-06, 307.37444494263616, 1.4630573145769372e-08, 4.5768825621743376e-09, 1.1823906621661553, 0.312830024946617, 0.21998150220477225)
                ),
                rtol=0.01,
                atol=0.01,
            )
        )

        # np.array((3.517242770690013e-06, 26.978226076495748, 3.187544456106211e-10, 1.2783794881088614e-10, 0.025760595723683707, 0.4010546380490277, 0.04309185078659044)) this is run from the given random number kernal and random number seed. This is a very strict criteria to see if the behavior of KMC is correct
        # with 0-7, 32-37 selected: np.array(1.1193006038758543e-06, 307.37444494263616, 1.4630573145769372e-08, 4.5768825621743376e-09, 1.1823906621661553, 0.312830024946617, 0.21998150220477225)
    def test_gather_mc_data(self):
        
        from kmcpy.tools.gather_mc_data import generate_supercell, gather_data
        from kmcpy.external.structure import StructureKMCpy
        import numpy as np

        structure_from_json = generate_supercell(
            f"{file_path}/gather_mc_data/prim.json", (8, 8, 8)
        )
        df = gather_data(f"{file_path}/gather_mc_data/comp*", structure_from_json)
        df.to_json(f"{file_path}/gather_mc_data/mc_results_json.json", orient="index")
        occ1 = df["occ"]

        structure_from_cif = StructureKMCpy.from_cif(
            f"{file_path}/EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif", primitive=True
        )
        structure_from_cif.remove_species(["Zr", "O", "Zr4+", "O2-"])
        structure_from_cif.remove_oxidation_states()
        structure_from_cif = structure_from_cif.make_kmc_supercell([8, 8, 8])
        df2 = gather_data(f"{file_path}/gather_mc_data/comp*", structure_from_cif)
        df2.to_json(f"{file_path}/gather_mc_data/mc_results_cif.json", orient="index")
        occ2 = df2["occ"]
        for i in range(0, len(occ1[0])):
            if occ1[0][i] != occ2[0][i]:
                print(i, occ1[i], occ2[i])
        self.assertTrue(np.allclose(occ1[0], occ2[0], rtol=0.001, atol=0.001))


if __name__ == "__main__":
    unittest.main()
