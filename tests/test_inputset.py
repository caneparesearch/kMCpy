#!/usr/bin/env python3
"""
Unit tests for InputSet parameter handling and validation.
"""

import pytest
import json
import tempfile
from pathlib import Path

from kmcpy.io.io import InputSet, _load_occ


class TestInputSetParameterHandling:
    """Test cases for InputSet parameter handling."""
    
    def test_kmc_parameter_validation(self):
        """Test that all KMC parameters are properly validated."""
        # Complete set of valid KMC parameters
        valid_kmc_params = {
            'task': 'kmc',
            'attempt_frequency': 5e12,
            'equ_pass': 1,
            'kmc_pass': 10,
            'supercell_shape': [2, 1, 1],
            'fitting_results': 'test.json',
            'fitting_results_site': 'test.json',
            'lce_fname': 'test.json',
            'lce_site_fname': 'test.json',
            'template_structure_fname': 'test.cif',
            'event_fname': 'test.json',
            'event_kernel': 'test.csv',
            'initial_state': 'test.json',
            'temperature': 298.0,
            'dimension': 3,
            'q': 1.0,
            'elem_hop_distance': 3.5,
            'mobile_ion_specie': 'Na',
            'convert_to_primitive_cell': True,
            'immutable_sites': ['Zr', 'O'],
            # New parameters that should be accepted
            'random_seed': 12345,
            'name': 'test_simulation',
            'event_dependencies': 'test_deps.csv'
        }
        
        # This should not raise an exception
        inputset = InputSet(valid_kmc_params)
        inputset.parameter_checker()
        
        # Verify that all parameters are accessible
        assert inputset.task == 'kmc'
        assert inputset.attempt_frequency == 5e12
        assert inputset.random_seed == 12345
        assert inputset.name == 'test_simulation'
        assert inputset.event_dependencies == 'test_deps.csv'
        assert inputset.event_kernel == 'test.csv'
    
    def test_case_insensitive_parameters(self):
        """Test that parameter names are case-insensitive."""
        mixed_case_params = {
            'TASK': 'kmc',
            'ATTEMPT_FREQUENCY': 5e12,
            'Equ_Pass': 1,
            'KMC_Pass': 10,
            'Supercell_Shape': [2, 1, 1],
            'Fitting_Results': 'test.json',
            'Fitting_Results_Site': 'test.json',
            'LCE_Fname': 'test.json',
            'LCE_Site_Fname': 'test.json',
            'Template_Structure_Fname': 'test.cif',
            'Event_Fname': 'test.json',
            'Event_Dependencies': 'test.csv',  # Use modern parameter name
            'Initial_State': 'test.json',
            'Temperature': 298.0,
            'Dimension': 3,
            'Q': 1.0,
            'Elem_Hop_Distance': 3.5,
            'Mobile_Ion_Specie': 'Na',
            'Convert_To_Primitive_Cell': True,
            'Immutable_Sites': ['Zr', 'O'],
            'Random_Seed': 12345,
            'Name': 'test'
        }
        
        # Should work with mixed case
        inputset = InputSet(mixed_case_params)
        inputset.parameter_checker()
        
        # Parameters should be converted to lowercase
        assert inputset.task == 'kmc'
        assert inputset.attempt_frequency == 5e12
        assert inputset.random_seed == 12345
        assert inputset.event_dependencies == 'test.csv'
    
    def test_optional_parameters(self):
        """Test that optional parameters are properly handled."""
        # Minimal required parameters
        minimal_params = {
            'task': 'kmc',
            'attempt_frequency': 5e12,
            'equ_pass': 1,
            'kmc_pass': 10,
            'supercell_shape': [2, 1, 1],
            'fitting_results': 'test.json',
            'fitting_results_site': 'test.json',
            'lce_fname': 'test.json',
            'lce_site_fname': 'test.json',
            'template_structure_fname': 'test.cif',
            'event_fname': 'test.json',
            'event_dependencies': 'test.csv',
            'initial_state': 'test.json',
            'temperature': 298.0,
            'dimension': 3,
            'q': 1.0,
            'elem_hop_distance': 3.5,
            'mobile_ion_specie': 'Na',
            'convert_to_primitive_cell': True,
            'immutable_sites': ['Zr', 'O']
        }
        
        # Should work without optional parameters
        inputset = InputSet(minimal_params)
        inputset.parameter_checker()
        
        # Optional parameters should have default values or be None
        assert not hasattr(inputset, 'random_seed') or inputset.random_seed is None
        assert not hasattr(inputset, 'name') or inputset.name is None
    
    def test_missing_required_parameters(self):
        """Test that missing required parameters raise appropriate errors."""
        incomplete_params = {
            'task': 'kmc',
            'attempt_frequency': 5e12,
            'temperature': 298.0,
            # Missing many required parameters
        }
        
        inputset = InputSet(incomplete_params)
        with pytest.raises(ValueError, match="Missing required parameters"):
            inputset.parameter_checker()
    
    def test_invalid_task_parameter(self):
        """Test that invalid task parameter raises appropriate error."""
        invalid_params = {
            'task': 'invalid_task',
            'attempt_frequency': 5e12,
            'temperature': 298.0,
        }
        
        inputset = InputSet(invalid_params)
        with pytest.raises(ValueError, match="Unknown task"):
            inputset.parameter_checker()
    
    def test_backward_compatibility(self):
        """Test backward compatibility with event_kernel parameter."""
        old_style_params = {
            'task': 'kmc',
            'attempt_frequency': 5e12,
            'equ_pass': 1,
            'kmc_pass': 10,
            'supercell_shape': [2, 1, 1],
            'fitting_results': 'test.json',
            'fitting_results_site': 'test.json',
            'lce_fname': 'test.json',
            'lce_site_fname': 'test.json',
            'template_structure_fname': 'test.cif',
            'event_fname': 'test.json',
            'event_dependencies': 'test.csv',  # Use modern parameter name
            'initial_state': 'test.json',
            'temperature': 298.0,
            'dimension': 3,
            'q': 1.0,
            'elem_hop_distance': 3.5,
            'mobile_ion_specie': 'Na',
            'convert_to_primitive_cell': True,
            'immutable_sites': ['Zr', 'O']
        }
        
        # Should work with new parameter name
        inputset = InputSet(old_style_params)
        inputset.parameter_checker()
        
        assert inputset.event_dependencies == 'test.csv'
        assert inputset.event_kernel == 'test.csv'  # Backward compatibility property
    
    def test_new_parameter_names(self):
        """Test that new parameter names are accepted."""
        new_style_params = {
            'task': 'kmc',
            'attempt_frequency': 5e12,
            'equ_pass': 1,
            'kmc_pass': 10,
            'supercell_shape': [2, 1, 1],
            'fitting_results': 'test.json',
            'fitting_results_site': 'test.json',
            'lce_fname': 'test.json',
            'lce_site_fname': 'test.json',
            'template_structure_fname': 'test.cif',
            'event_fname': 'test.json',
            'event_dependencies': 'test.csv',  # New parameter name
            'initial_state': 'test.json',
            'temperature': 298.0,
            'dimension': 3,
            'q': 1.0,
            'elem_hop_distance': 3.5,
            'mobile_ion_specie': 'Na'
        }
        
        # Should work with new parameter name
        inputset = InputSet(new_style_params)
        inputset.parameter_checker()
        
        assert inputset.event_dependencies == 'test.csv'  # InputSet now uses event_dependencies internally
    
    def test_parameter_access(self):
        """Test parameter access via __getattr__."""
        params = {
            'task': 'kmc',
            'attempt_frequency': 5e12,
            'equ_pass': 1,
            'kmc_pass': 10,
            'supercell_shape': [2, 1, 1],
            'fitting_results': 'test.json',
            'fitting_results_site': 'test.json',
            'lce_fname': 'test.json',
            'lce_site_fname': 'test.json',
            'template_structure_fname': 'test.cif',
            'event_fname': 'test.json',
            'event_dependencies': 'test.csv',
            'initial_state': 'test.json',
            'temperature': 298.0,
            'dimension': 3,
            'q': 1.0,
            'elem_hop_distance': 3.5,
            'mobile_ion_specie': 'Na',
            'random_seed': 42,
            'name': 'test_sim'
        }
        
        inputset = InputSet(params)
        inputset.parameter_checker()
        
        # Test attribute access
        assert inputset.task == 'kmc'
        assert inputset.attempt_frequency == 5e12
        assert inputset.random_seed == 42
        assert inputset.name == 'test_sim'
        
        # Test non-existent parameter
        with pytest.raises(AttributeError):
            _ = inputset.non_existent_parameter
    
    def test_ignored_parameters_warning(self):
        """Test that ignored parameters generate warnings."""
        params_with_unknown = {
            'task': 'kmc',
            'attempt_frequency': 5e12,
            'equ_pass': 1,
            'kmc_pass': 10,
            'supercell_shape': [2, 1, 1],
            'fitting_results': 'test.json',
            'fitting_results_site': 'test.json',
            'lce_fname': 'test.json',
            'lce_site_fname': 'test.json',
            'template_structure_fname': 'test.cif',
            'event_fname': 'test.json',
            'event_dependencies': 'test.csv',
            'initial_state': 'test.json',
            'temperature': 298.0,
            'dimension': 3,
            'q': 1.0,
            'elem_hop_distance': 3.5,
            'mobile_ion_specie': 'Na',
            'convert_to_primitive_cell': True,
            'immutable_sites': ['Zr', 'O'],
            'unknown_parameter': 'should_be_ignored'  # This should generate a warning
        }
        
        inputset = InputSet(params_with_unknown)
        
        # Should work but generate warning
        with pytest.warns(UserWarning):
            inputset.parameter_checker()
        
        # Check that warning was generated (if logging is configured)
        # Note: This might not work in all test environments
        assert inputset.unknown_parameter == 'should_be_ignored'


class TestInputSetFileHandling:
    """Test cases for InputSet file handling."""
    
    def test_from_json(self):
        """Test InputSet creation from JSON file."""
        # Create a temporary JSON file
        test_params = {
            'task': 'kmc',
            'attempt_frequency': 5e12,
            'equ_pass': 1,
            'kmc_pass': 10,
            'supercell_shape': [2, 1, 1],
            'fitting_results': 'test.json',
            'fitting_results_site': 'test.json',
            'lce_fname': 'test.json',
            'lce_site_fname': 'test.json',
            'template_structure_fname': 'test.cif',
            'event_fname': 'test.json',
            'event_kernel': 'test.csv',
            'initial_state': 'test.json',
            'temperature': 298.0,
            'dimension': 3,
            'q': 1.0,
            'elem_hop_distance': 3.5,
            'mobile_ion_specie': 'Na',
            'random_seed': 42,
            'convert_to_primitive_cell': False
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_params, f)
            temp_path = f.name
        
        try:
            # This would fail due to missing structure file, but we can test the JSON loading
            with pytest.raises(FileNotFoundError):
                InputSet.from_json(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_from_dict(self):
        """Test InputSet creation from dictionary."""
        test_params = {
            'task': 'kmc',
            'attempt_frequency': 5e12,
            'equ_pass': 1,
            'kmc_pass': 10,
            'supercell_shape': [2, 1, 1],
            'fitting_results': 'test.json',
            'fitting_results_site': 'test.json',
            'lce_fname': 'test.json',
            'lce_site_fname': 'test.json',
            'template_structure_fname': 'test.cif',
            'event_fname': 'test.json',
            'event_dependencies': 'test.csv',
            'initial_state': 'test.json',
            'temperature': 298.0,
            'dimension': 3,
            'q': 1.0,
            'elem_hop_distance': 3.5,
            'mobile_ion_specie': 'Na',
            'convert_to_primitive_cell': True,
            'immutable_sites': ['Zr', 'O'],
            'random_seed': 42
        }
        
        # This would fail due to missing structure file, but we can test the dict loading
        with pytest.raises(FileNotFoundError):
            InputSet.from_dict(test_params)
    
    def test_parameter_setter(self):
        """Test parameter setting functionality."""
        params = {
            'task': 'kmc',
            'attempt_frequency': 5e12,
            'temperature': 298.0,
        }
        
        inputset = InputSet(params)
        
        # Test setting existing parameter
        inputset.set_parameter('attempt_frequency', 1e13)
        assert inputset.attempt_frequency == 1e13
        
        # Test setting non-existent parameter
        with pytest.raises(KeyError):
            inputset.set_parameter('non_existent', 'value')
    
    def test_enumerate_functionality(self):
        """Test parameter enumeration functionality."""
        params = {
            'task': 'kmc',
            'attempt_frequency': 5e12,
            'temperature': 298.0,
        }
        
        inputset = InputSet(params)
        
        # Test enumeration
        new_inputset = inputset.enumerate(temperature=400.0)
        
        # Original should be unchanged
        assert inputset.temperature == 298.0
        
        # New inputset should have updated parameter
        assert new_inputset.temperature == 400.0


class TestInputSetYAMLHandling:
    """Test cases for InputSet YAML handling."""
    
    def test_yaml_section_loading(self):
        """Test loading specific sections from YAML."""
        # Create a temporary YAML file with structure
        yaml_content = """
model:
  type: lce
  lce:
    center_frac_coord: [0.5, 0.5, 0.5]
    mobile_ion_identifier_type: "specie"
    mobile_ion_specie_identifier: "Na"
    cutoff_cluster: 5.0
    cutoff_region: 10.0
    template_structure_fname: "test.cif"
    is_write_basis: true
    species_to_be_removed: ["O"]
    convert_to_primitive_cell: false
    exclude_site_with_identifier: ["Zr"]

kmc:
  type: default
  default:
    attempt_frequency: 1.0
    equ_pass: 100
    kmc_pass: 1000
    supercell_shape: [2, 1, 1]
    fitting_results: "results.json"
    fitting_results_site: "results_site.json"
    lce_fname: "lce.json"
    lce_site_fname: "lce_site.json"
    template_structure_fname: "nonexistent.cif"
    event_fname: "event.json"
    event_dependencies: "dependencies.json"
    initial_state: "initial_state.json"
    temperature: 298
    dimension: 3
    q: 1
    elem_hop_distance: 3.47782
    convert_to_primitive_cell: true
    immutable_sites: ["Zr", "O"]
    mobile_ion_specie: "Na"
    random_seed: 42
    name: "Test Simulation"

generate_event:
  type: default
  default:
    template_structure_fname: "test.cif"
    convert_to_primitive_cell: false
    local_env_cutoff_dict: {"Na": 3.0, "O": 2.0}
    mobile_ion_identifier_type: "specie"
    mobile_ion_specie_identifier: "Na"
    mobile_ion_specie_2_identifier: "Li"
    species_to_be_removed: ["O"]
    distance_matrix_rtol: 1e-5
    distance_matrix_atol: 1e-8
    find_nearest_if_fail: true
    export_local_env_structure: false
    supercell_shape: [2, 1, 1]
    event_fname: "event.json"
    event_dependencies_fname: "dependencies.json"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            # Test loading KMC section - will fail due to missing file, but we test parameter loading
            with pytest.raises(FileNotFoundError):
                InputSet.from_yaml_section(temp_path, section="kmc")
            
            # Test loading Model section - should work since no occupation loading needed
            model_input = InputSet.from_yaml_section(temp_path, section="model", task_type="lce")
            assert model_input.task == "lce"
            assert model_input.cutoff_cluster == 5.0
            assert model_input.mobile_ion_specie_identifier == "Na"
            assert model_input.cutoff_region == 10.0
            assert model_input.is_write_basis == True
            
            # Test loading Generate Event section - should work since no occupation loading needed
            event_input = InputSet.from_yaml_section(temp_path, section="generate_event")
            assert event_input.task == "generate_event"
            assert event_input.find_nearest_if_fail == True
            assert event_input.local_env_cutoff_dict == {"Na": 3.0, "O": 2.0}
            
        finally:
            Path(temp_path).unlink()
    
    def test_yaml_convenience_methods(self):
        """Test convenience methods for loading specific sections."""
        yaml_content = """
model:
  type: composite_lce
  composite_lce:
    center_frac_coord: [0.5, 0.5, 0.5]
    mobile_ion_identifier_type: "specie"
    mobile_ion_specie_identifier: "Na"
    cutoff_cluster: 5.0
    cutoff_region: 10.0
    template_structure_fname: "test.cif"
    is_write_basis: true
    species_to_be_removed: ["O"]
    convert_to_primitive_cell: false
    exclude_site_with_identifier: ["Zr"]

kmc:
  type: default
  default:
    attempt_frequency: 1.0
    equ_pass: 100
    kmc_pass: 1000
    supercell_shape: [2, 1, 1]
    fitting_results: "results.json"
    fitting_results_site: "results_site.json"
    lce_fname: "lce.json"
    lce_site_fname: "lce_site.json"
    template_structure_fname: "nonexistent.cif"
    event_fname: "event.json"
    event_dependencies: "dependencies.json"
    initial_state: "initial_state.json"
    temperature: 298
    dimension: 3
    q: 1
    elem_hop_distance: 3.47782
    convert_to_primitive_cell: true
    immutable_sites: ["Zr", "O"]
    mobile_ion_specie: "Na"

generate_event:
  type: default
  default:
    template_structure_fname: "test.cif"
    convert_to_primitive_cell: false
    local_env_cutoff_dict: {"Na": 3.0, "O": 2.0}
    mobile_ion_identifier_type: "specie"
    mobile_ion_specie_identifier: "Na"
    mobile_ion_specie_2_identifier: "Li"
    species_to_be_removed: ["O"]
    supercell_shape: [2, 1, 1]
    event_fname: "event.json"
    event_dependencies_fname: "dependencies.json"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            # Test convenience methods - KMC will fail due to missing file, but others should work
            with pytest.raises(FileNotFoundError):
                InputSet.from_yaml_kmc(temp_path)
            
            model_input = InputSet.from_yaml_model(temp_path, model_type="composite_lce")
            assert model_input.task == "lce"
            assert model_input.cutoff_cluster == 5.0
            
            event_input = InputSet.from_yaml_generate_event(temp_path)
            assert event_input.task == "generate_event"
            assert event_input.local_env_cutoff_dict == {"Na": 3.0, "O": 2.0}
            
        finally:
            Path(temp_path).unlink()
    
    def test_model_registry(self):
        """Test model registry functionality with different model types."""
        yaml_content = """
model:
  type: composite_lce
  composite_lce:
    center_frac_coord: [0.5, 0.5, 0.5]
    mobile_ion_identifier_type: "specie"
    mobile_ion_specie_identifier: "Na"
    cutoff_cluster: 5.0
    cutoff_region: 10.0
    template_structure_fname: "test.cif"
    is_write_basis: true
    species_to_be_removed: ["O"]
    convert_to_primitive_cell: false
    exclude_site_with_identifier: ["Zr"]
  lce:
    center_frac_coord: [0.0, 0.0, 0.0]
    mobile_ion_identifier_type: "specie"
    mobile_ion_specie_identifier: "Li"
    cutoff_cluster: 3.0
    cutoff_region: 8.0
    template_structure_fname: "test.cif"
    is_write_basis: false
    species_to_be_removed: ["O"]
    convert_to_primitive_cell: true
    exclude_site_with_identifier: ["Zr"]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            # Test composite_lce model type
            composite_input = InputSet.from_yaml_model(temp_path, model_type="composite_lce")
            assert composite_input.task == "lce"  # Maps to lce task
            assert composite_input.cutoff_cluster == 5.0
            assert composite_input.mobile_ion_specie_identifier == "Na"
            assert composite_input.cutoff_region == 10.0
            
            # Test regular lce model type
            lce_input = InputSet.from_yaml_model(temp_path, model_type="lce")
            assert lce_input.task == "lce"
            assert lce_input.cutoff_cluster == 3.0
            assert lce_input.mobile_ion_specie_identifier == "Li"
            assert lce_input.cutoff_region == 8.0
            
        finally:
            Path(temp_path).unlink()
    
    def test_yaml_error_handling(self):
        """Test error handling for YAML files."""
        # Test missing section
        yaml_content = """
kmc:
  type: default
  default:
    attempt_frequency: 1.0
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            # Test missing section error
            with pytest.raises(ValueError, match="Section 'model' not found"):
                InputSet.from_yaml_section(temp_path, section="model")
            
            # Test missing task type error
            yaml_content_missing_type = """
model:
  type: nonexistent_type
  lce:
    cutoff_cluster: 5.0
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f2:
                f2.write(yaml_content_missing_type)
                temp_path2 = f2.name
            
            try:
                with pytest.raises(ValueError, match="Task type 'nonexistent_type' not found"):
                    InputSet.from_yaml_model(temp_path2, model_type="nonexistent_type")
            finally:
                Path(temp_path2).unlink()
                
        finally:
            Path(temp_path).unlink()
    
    def test_yaml_auto_detection(self):
        """Test YAML auto-detection functionality."""
        # Test single section auto-detection
        single_section_yaml = """
kmc:
  type: default
  default:
    attempt_frequency: 1.0
    equ_pass: 100
    kmc_pass: 1000
    supercell_shape: [2, 1, 1]
    fitting_results: "results.json"
    fitting_results_site: "results_site.json"
    lce_fname: "lce.json"
    lce_site_fname: "lce_site.json"
    template_structure_fname: "nonexistent.cif"
    event_fname: "event.json"
    event_dependencies: "dependencies.json"
    initial_state: "initial_state.json"
    temperature: 298
    dimension: 3
    q: 1
    elem_hop_distance: 3.47782
    convert_to_primitive_cell: true
    immutable_sites: ["Zr", "O"]
    mobile_ion_specie: "Na"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(single_section_yaml)
            temp_path = f.name
        
        try:
            # Should auto-detect single section but fail due to missing file
            with pytest.raises(FileNotFoundError):
                InputSet.from_yaml(temp_path)
        finally:
            Path(temp_path).unlink()
        
        # Test multiple sections error
        multi_section_yaml = """
model:
  type: lce
  lce:
    cutoff_cluster: 5.0
    cutoff_region: 10.0
    template_structure_fname: "test.cif"
    is_write_basis: true
    species_to_be_removed: ["O"]
    convert_to_primitive_cell: false
    exclude_site_with_identifier: ["Zr"]
    center_frac_coord: [0.5, 0.5, 0.5]
    mobile_ion_identifier_type: "specie"
    mobile_ion_specie_identifier: "Na"
kmc:
  type: default
  default:
    attempt_frequency: 1.0
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(multi_section_yaml)
            temp_path = f.name
        
        try:
            # Should raise error for multiple sections without specification
            with pytest.raises(ValueError, match="Multiple sections found"):
                InputSet.from_yaml(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_legacy_yaml_compatibility(self):
        """Test backward compatibility with legacy flat YAML files."""
        legacy_yaml = """
task: kmc
attempt_frequency: 5e12
equ_pass: 1
kmc_pass: 10
supercell_shape: [2, 1, 1]
fitting_results: "test.json"
fitting_results_site: "test.json"
lce_fname: "test.json"
lce_site_fname: "test.json"
template_structure_fname: "test.cif"
event_fname: "test.json"
event_dependencies: "test.csv"
initial_state: "test.json"
temperature: 298.0
dimension: 3
q: 1.0
elem_hop_distance: 3.5
mobile_ion_specie: "Na"
convert_to_primitive_cell: true
immutable_sites: ["Zr", "O"]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(legacy_yaml)
            temp_path = f.name
        
        try:
            # Should work with legacy format but fail due to missing file
            with pytest.raises(FileNotFoundError):
                InputSet.from_yaml(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_conditional_occupation_loading(self):
        """Test that occupation data is only loaded for KMC tasks."""
        # Create YAML with different task types
        yaml_content = """
model:
  type: lce
  lce:
    center_frac_coord: [0.5, 0.5, 0.5]
    mobile_ion_identifier_type: "specie"
    mobile_ion_specie_identifier: "Na"
    cutoff_cluster: 5.0
    cutoff_region: 10.0
    template_structure_fname: "nonexistent.cif"
    is_write_basis: true
    species_to_be_removed: ["O"]
    exclude_site_with_identifier: ["Zr"]

generate_event:
  type: default
  default:
    template_structure_fname: "nonexistent.cif"
    local_env_cutoff_dict: {"Na": 3.0}
    mobile_ion_identifier_type: "specie"
    mobile_ion_specie_identifier: "Na"
    mobile_ion_specie_2_identifier: "Li"
    species_to_be_removed: ["O"]
    supercell_shape: [2, 1, 1]
    event_fname: "event.json"
    event_dependencies_fname: "dependencies.json"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            # Model and generate_event tasks should not try to load occupation data
            # even if structure file doesn't exist
            model_input = InputSet.from_yaml_model(temp_path, model_type="lce")
            assert model_input.task == "lce"
            assert model_input.structure is None
            assert model_input.occupation == []
            assert model_input.n_sites == 0
            
            event_input = InputSet.from_yaml_generate_event(temp_path)
            assert event_input.task == "generate_event"
            assert event_input.structure is None
            assert event_input.occupation == []
            assert event_input.n_sites == 0
            
        finally:
            Path(temp_path).unlink()
    
    def test_parameter_validation_by_task(self):
        """Test that parameter validation works correctly for different tasks."""
        # Test LCE task parameters
        lce_yaml = """
model:
  type: lce
  lce:
    center_frac_coord: [0.5, 0.5, 0.5]
    mobile_ion_identifier_type: "specie"
    mobile_ion_specie_identifier: "Na"
    cutoff_cluster: 5.0
    cutoff_region: 10.0
    template_structure_fname: "test.cif"
    is_write_basis: true
    species_to_be_removed: ["O"]
    convert_to_primitive_cell: false
    exclude_site_with_identifier: ["Zr"]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(lce_yaml)
            temp_path = f.name
        
        try:
            # Should validate LCE parameters correctly
            model_input = InputSet.from_yaml_model(temp_path, model_type="lce")
            assert model_input.task == "lce"
            # No exception should be raised during parameter checking
        finally:
            Path(temp_path).unlink()
        
        # Test generate_event task parameters
        event_yaml = """
generate_event:
  type: default
  default:
    template_structure_fname: "test.cif"
    convert_to_primitive_cell: false
    local_env_cutoff_dict: {"Na": 3.0, "O": 2.0}
    mobile_ion_identifier_type: "specie"
    mobile_ion_specie_identifier: "Na"
    mobile_ion_specie_2_identifier: "Li"
    species_to_be_removed: ["O"]
    distance_matrix_rtol: 1e-5
    distance_matrix_atol: 1e-8
    find_nearest_if_fail: true
    export_local_env_structure: false
    supercell_shape: [2, 1, 1]
    event_fname: "event.json"
    event_dependencies_fname: "dependencies.json"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(event_yaml)
            temp_path = f.name
        
        try:
            # Should validate generate_event parameters correctly
            event_input = InputSet.from_yaml_generate_event(temp_path)
            assert event_input.task == "generate_event"
            # No exception should be raised during parameter checking
        finally:
            Path(temp_path).unlink()


class TestInputSetAdvancedFeatures:
    """Test cases for advanced InputSet features."""
    
    def test_string_representation(self):
        """Test string representation of InputSet."""
        params = {
            'task': 'kmc',
            'attempt_frequency': 5e12,
            'temperature': 298.0,
        }
        
        inputset = InputSet(params)
        str_repr = str(inputset)
        
        # Should contain the parameters
        assert 'task' in str_repr
        assert 'kmc' in str_repr
        assert 'attempt_frequency' in str_repr
    
    def test_change_key_name(self):
        """Test key name changing functionality."""
        params = {
            'task': 'kmc',
            'lce': 'old_value',
            'temperature': 298.0,
        }
        
        inputset = InputSet(params)
        inputset.change_key_name(oldname="lce", newname="lce_fname")
        
        # Should have new key
        assert inputset.lce_fname == 'old_value'
        # Old key should still exist (change_key_name adds, doesn't remove)
        assert inputset.lce == 'old_value'
    
    def test_yaml_with_complex_data_types(self):
        """Test YAML loading with complex data types like dictionaries and lists."""
        yaml_content = """
generate_event:
  type: default
  default:
    template_structure_fname: "test.cif"
    local_env_cutoff_dict: 
      Na: 3.0
      O: 2.0
      Zr: 2.5
    supercell_shape: [2, 1, 1]
    species_to_be_removed: ["O", "Zr4+", "O2-"]
    mobile_ion_identifier_type: "specie"
    mobile_ion_specie_identifier: "Na"
    mobile_ion_specie_2_identifier: "Li"
    distance_matrix_rtol: 1e-5
    distance_matrix_atol: 1e-8
    find_nearest_if_fail: true
    export_local_env_structure: false
    event_fname: "event.json"
    event_dependencies_fname: "dependencies.json"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            event_input = InputSet.from_yaml_generate_event(temp_path)
            
            # Test dictionary loading
            assert isinstance(event_input.local_env_cutoff_dict, dict)
            assert event_input.local_env_cutoff_dict["Na"] == 3.0
            assert event_input.local_env_cutoff_dict["O"] == 2.0
            assert event_input.local_env_cutoff_dict["Zr"] == 2.5
            
            # Test list loading
            assert isinstance(event_input.species_to_be_removed, list)
            assert "O" in event_input.species_to_be_removed
            assert "Zr4+" in event_input.species_to_be_removed
            assert "O2-" in event_input.species_to_be_removed
            
            assert isinstance(event_input.supercell_shape, list)
            assert event_input.supercell_shape == [2, 1, 1]
            
            # Test numeric types (may come as strings from YAML)
            assert isinstance(event_input.distance_matrix_rtol, (int, float, str))
            assert isinstance(event_input.distance_matrix_atol, (int, float, str))
            
            # Test boolean
            assert isinstance(event_input.find_nearest_if_fail, bool)
            assert event_input.find_nearest_if_fail == True
            
        finally:
            Path(temp_path).unlink()
