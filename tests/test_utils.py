"""
Test utilities for kMCpy unit tests.

This module contains utility functions that are specific to testing,
including configuration builders for specific material systems.
"""

from kmcpy.simulator.condition import SimulationConfig


def create_nasicon_config(
    name: str = "NASICON_Simulation",
    temperature: float = 573.0,
    supercell_shape: list = None,
    initial_occ: list = None,
    data_dir: str = "example",
    **kwargs
) -> SimulationConfig:
    """
    Create a standard NASICON simulation configuration for testing.
    
    This is a test utility function that creates a SimulationConfig
    with typical NASICON parameters. It should only be used in tests.
    
    Args:
        name: Simulation name
        temperature: Temperature in Kelvin
        supercell_shape: Supercell dimensions [nx, ny, nz]
        initial_occ: Initial occupation list
        data_dir: Directory containing input files
        **kwargs: Additional parameters to override defaults
    
    Returns:
        SimulationConfig: Configured simulation setup for NASICON testing
    """
    if supercell_shape is None:
        supercell_shape = [2, 2, 2]
    if initial_occ is None:
        initial_occ = [1, -1, 1, -1, 1, -1, 1, -1]
    
    default_config = {
        'name': name,
        'temperature': temperature,
        'attempt_frequency': 1e13,
        'equilibration_passes': 1000,
        'kmc_passes': 10000,
        'dimension': 3,
        'elementary_hop_distance': 2.5,
        'mobile_ion_charge': 1.0,
        'mobile_ion_specie': 'Na',
        'supercell_shape': supercell_shape,
        'initial_occ': initial_occ,
        'fitting_results': f"{data_dir}/fitting_results.json",
        'fitting_results_site': f"{data_dir}/fitting_results_site.json",
        'lce_fname': f"{data_dir}/lce.json",
        'lce_site_fname': f"{data_dir}/lce_site.json",
        'template_structure_fname': f"{data_dir}/0th_reference_local_env.cif",
        'event_fname': f"{data_dir}/events.json",
        'event_dependencies': f"{data_dir}/event_dependencies.csv"
    }
    
    # Override with user-provided kwargs
    default_config.update(kwargs)
    
    return SimulationConfig(**default_config)


def create_test_config(
    name: str = "Test_Simulation",
    temperature: float = 300.0,
    **kwargs
) -> SimulationConfig:
    """
    Create a minimal test configuration with dummy file paths.
    
    This is useful for testing the configuration system without
    requiring actual input files.
    
    Args:
        name: Simulation name
        temperature: Temperature in Kelvin
        **kwargs: Additional parameters to override defaults
    
    Returns:
        SimulationConfig: Minimal test configuration
    """
    default_config = {
        'name': name,
        'temperature': temperature,
        'attempt_frequency': 1e13,
        'equilibration_passes': 100,
        'kmc_passes': 1000,
        'dimension': 3,
        'elementary_hop_distance': 1.0,
        'mobile_ion_charge': 1.0,
        'mobile_ion_specie': 'Li',
        'supercell_shape': [1, 1, 1],
        'initial_occ': [1, -1],
        'fitting_results': 'test_fitting.json',
        'lce_fname': 'test_lce.json',
        'template_structure_fname': 'test_structure.cif',
        'event_fname': 'test_events.json',
        'event_dependencies': 'test_dependencies.csv'
    }
    
    # Override with user-provided kwargs
    default_config.update(kwargs)
    
    return SimulationConfig(**default_config)


def create_temperature_series(
    base_config: SimulationConfig,
    temperatures: list,
    name_template: str = "{base_name}_T_{temp}K"
) -> list[SimulationConfig]:
    """
    Create a series of configurations with different temperatures.
    
    This is a test utility function for parameter studies.
    
    Args:
        base_config: Base configuration to modify
        temperatures: List of temperatures to test
        name_template: Template for naming each configuration
    
    Returns:
        List of SimulationConfig objects
    """
    configs = []
    
    for temp in temperatures:
        new_name = name_template.format(
            base_name=base_config.name, 
            temp=temp
        )
        config = base_config.copy_with_changes(temperature=temp, name=new_name)
        configs.append(config)
    
    return configs
