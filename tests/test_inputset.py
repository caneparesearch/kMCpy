#!/usr/bin/env python3
"""
Unit tests for InputSet parameter handling and validation.
"""

import pytest
import json
import tempfile
from pathlib import Path

from kmcpy.io import InputSet, _load_occ


class TestInputSetParameterHandling:
    """Test cases for InputSet parameter handling."""
    
    def test_kmc_parameter_validation(self):
        """Test that all KMC parameters are properly validated."""
        # Complete set of valid KMC parameters
        valid_kmc_params = {
            'task': 'kmc',
            'v': 5e12,
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
        assert inputset.v == 5e12
        assert inputset.random_seed == 12345
        assert inputset.name == 'test_simulation'
        assert inputset.event_dependencies == 'test_deps.csv'
        assert inputset.event_kernel == 'test.csv'
    
    def test_case_insensitive_parameters(self):
        """Test that parameter names are case-insensitive."""
        mixed_case_params = {
            'TASK': 'kmc',
            'V': 5e12,
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
        assert inputset.v == 5e12
        assert inputset.random_seed == 12345
        assert inputset.event_dependencies == 'test.csv'
    
    def test_optional_parameters(self):
        """Test that optional parameters are properly handled."""
        # Minimal required parameters
        minimal_params = {
            'task': 'kmc',
            'v': 5e12,
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
            'v': 5e12,
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
            'v': 5e12,
            'temperature': 298.0,
        }
        
        inputset = InputSet(invalid_params)
        with pytest.raises(ValueError, match="Unknown task"):
            inputset.parameter_checker()
    
    def test_backward_compatibility(self):
        """Test backward compatibility with event_kernel parameter."""
        old_style_params = {
            'task': 'kmc',
            'v': 5e12,
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
            'v': 5e12,
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
            'v': 5e12,
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
        assert inputset.v == 5e12
        assert inputset.random_seed == 42
        assert inputset.name == 'test_sim'
        
        # Test non-existent parameter
        with pytest.raises(AttributeError):
            _ = inputset.non_existent_parameter
    
    def test_ignored_parameters_warning(self):
        """Test that ignored parameters generate warnings."""
        params_with_unknown = {
            'task': 'kmc',
            'v': 5e12,
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
            'v': 5e12,
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
            'v': 5e12,
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
            'v': 5e12,
            'temperature': 298.0,
        }
        
        inputset = InputSet(params)
        
        # Test setting existing parameter
        inputset.set_parameter('v', 1e13)
        assert inputset.v == 1e13
        
        # Test setting non-existent parameter
        with pytest.raises(KeyError):
            inputset.set_parameter('non_existent', 'value')
    
    def test_enumerate_functionality(self):
        """Test parameter enumeration functionality."""
        params = {
            'task': 'kmc',
            'v': 5e12,
            'temperature': 298.0,
        }
        
        inputset = InputSet(params)
        
        # Test enumeration
        new_inputset = inputset.enumerate(temperature=400.0)
        
        # Original should be unchanged
        assert inputset.temperature == 298.0
        
        # New inputset should have updated parameter
        assert new_inputset.temperature == 400.0
