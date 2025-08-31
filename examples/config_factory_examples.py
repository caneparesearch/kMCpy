"""
Examples showing the convenient SimulationConfig factory methods.

Before: You had to create SystemConfig, RuntimeConfig, and SimulationConfig separately.
After: Use simple factory methods for common use cases.
"""

from kmcpy.simulator.config import SimulationConfig

def example_old_way():
    """OLD WAY: Verbose and tedious"""
    from kmcpy.simulator.config import SystemConfig, RuntimeConfig
    
    # Create system configuration
    system = SystemConfig(
        structure_file="nasicon.cif",
        supercell_shape=(2, 2, 2),
        dimension=3,
        mobile_ion_specie="Na",
        mobile_ion_charge=1.0,
        cluster_expansion_file="lce.json",
        event_file="events.json"
    )
    
    # Create runtime configuration
    runtime = RuntimeConfig(
        name="NASICON_300K",
        temperature=300.0,
        attempt_frequency=1e13,
        equilibration_passes=1000,
        kmc_passes=50000,
        random_seed=42
    )
    
    # Finally create simulation configuration
    config = SimulationConfig(
        system=system,
        runtime=runtime,
        initial_occupations=(1, -1, 1, -1, 1, -1)
    )
    
    return config


def example_new_way_simple():
    """NEW WAY: Simple and clean"""
    
    config = SimulationConfig.create(
        # Required files
        structure_file="nasicon.cif",
        cluster_expansion_file="lce.json", 
        event_file="events.json",
        
        # Initial state
        initial_occupations=[1, -1, 1, -1, 1, -1],
        
        # Simulation parameters
        name="NASICON_300K",
        temperature=300.0,
        kmc_passes=50000,
        mobile_species="Na",
        supercell_shape=(2, 2, 2)
    )
    
    return config


def example_ultra_quick():
    """ULTRA QUICK: For rapid prototyping"""
    
    config = SimulationConfig.quick_setup(
        structure_file="structure.cif",
        model_files={
            'cluster_expansion': 'lce.json',
            'events': 'events.json'
        },
        initial_occupations=[1, -1, 1, -1],
        temperature=400.0,
        kmc_passes=20000
    )
    
    return config


def example_templates():
    """TEMPLATE-BASED: For common simulation types"""
    
    # Ion diffusion simulation
    config1 = SimulationConfig.from_template(
        template_name='ion_diffusion',
        structure_file='nasicon.cif',
        initial_occupations=[1, -1, 1, -1],
        cluster_expansion_file='lce.json',
        event_file='events.json',
        temperature=400.0  # Override template default
    )
    
    # Battery simulation  
    config2 = SimulationConfig.from_template(
        template_name='battery',
        structure_file='cathode.cif',
        initial_occupations=[1, -1, 0, 1],
        cluster_expansion_file='battery_lce.json',
        event_file='battery_events.json',
        mobile_species='Li'  # Override template default
    )
    
    return config1, config2


def example_parameter_variations():
    """Create variations easily"""
    
    # Base configuration
    base_config = SimulationConfig.create(
        structure_file="nasicon.cif",
        cluster_expansion_file="lce.json",
        event_file="events.json", 
        initial_occupations=[1, -1, 1, -1],
        name="Base_Simulation",
        temperature=300.0
    )
    
    # Temperature series
    temp_configs = []
    for T in [200, 300, 400, 500]:
        config = base_config.with_runtime_changes(
            temperature=T,
            name=f"NASICON_{T}K"
        )
        temp_configs.append(config)
    
    # Supercell size series  
    size_configs = []
    for n in [1, 2, 3, 4]:
        config = base_config.with_system_changes(
            supercell_shape=(n, n, n)
        )
        size_configs.append(config)
    
    return temp_configs, size_configs


def example_from_existing_config():
    """Convert from existing modern SimulationConfig file"""
    from kmcpy.io.config_io import SimulationConfigIO
    
    # Load existing modern config file
    config = SimulationConfigIO.read("modern_simulation.yaml")
    
    print(f"Loaded: {config.summary()}")
    
    return config


if __name__ == "__main__":
    print("=== SimulationConfig Factory Methods Examples ===\n")
    
    print("1. Simple creation:")
    config1 = example_new_way_simple()
    print(f"   {config1.summary()}\n")
    
    print("2. Ultra quick setup:")
    config2 = example_ultra_quick()
    print(f"   {config2.summary()}\n")
    
    print("3. Template-based:")
    ion_config, battery_config = example_templates()
    print(f"   Ion diffusion: {ion_config.summary()}")
    print(f"   Battery: {battery_config.summary()}\n")
    
    print("4. Parameter variations:")
    temp_configs, size_configs = example_parameter_variations()
    print(f"   Temperature series: {len(temp_configs)} configs")
    print(f"   Size series: {len(size_configs)} configs\n")
    
    print("✓ All factory methods working!")
