"""
SimulationConfig Factory Methods - User Guide

PROBLEM SOLVED:
Before, creating a SimulationConfig required manually creating SystemConfig and RuntimeConfig:
- Verbose and error-prone
- Required understanding internal structure 
- Hard to use for quick prototyping

NOW: Simple factory methods for common use cases!
"""

from kmcpy.simulator.config import SimulationConfig

# ===============================================
# METHOD 1: SimulationConfig.create() - Full Control
# ===============================================
def method_1_full_control():
    """
    Most flexible method with sensible defaults.
    Good for: Production simulations, full customization
    """
    config = SimulationConfig.create(
        # Required parameters
        structure_file="nasicon.cif",
        cluster_expansion_file="lce.json", 
        event_file="events.json",
        initial_occupations=[1, -1, 1, -1, 1, -1],
        
        # Common parameters  
        name="MySimulation",
        temperature=350.0,
        kmc_passes=50000,
        equilibration_passes=5000,
        
        # System parameters
        mobile_species="Na",  # Li, Na, K, H, etc.
        supercell_shape=(2, 2, 2),
        dimension=3,
        
        # Advanced parameters (optional)
        attempt_frequency=1e13,
        mobile_species_charge=1.0,
        random_seed=42,
        immutable_sites=[0, 1, 2]  # Sites that don't participate
    )
    return config


# ===============================================  
# METHOD 2: SimulationConfig.quick_setup() - Minimal
# ===============================================
def method_2_quick_setup():
    """
    Minimal parameters for rapid prototyping.
    Good for: Quick tests, parameter exploration
    """
    config = SimulationConfig.quick_setup(
        structure_file="structure.cif",
        model_files={
            'cluster_expansion': 'lce.json',
            'events': 'events.json'
        },
        initial_occupations=[1, -1, 1, -1],
        temperature=400.0,
        kmc_passes=20000,
        # Everything else uses sensible defaults
        mobile_species="Li"  # Optional override
    )
    return config


# ===============================================
# METHOD 3: SimulationConfig.from_template() - Presets  
# ===============================================
def method_3_templates():
    """
    Use predefined templates for common simulation types.
    Good for: Standard workflows, consistent setups
    """
    
    # Template 1: Ion diffusion
    ion_config = SimulationConfig.from_template(
        template_name='ion_diffusion',  # Optimized for diffusion studies
        structure_file='solid_electrolyte.cif',
        initial_occupations=[1, -1, 1, -1],
        cluster_expansion_file='lce.json',
        event_file='diffusion_events.json',
        temperature=400.0  # Override template default (300K)
    )
    
    # Template 2: Battery simulation
    battery_config = SimulationConfig.from_template(
        template_name='battery',  # Optimized for battery materials
        structure_file='cathode.cif', 
        initial_occupations=[1, 0, -1, 1],
        cluster_expansion_file='battery_lce.json',
        event_file='intercalation_events.json',
        kmc_passes=100000  # Override template default (50K)
    )
    
    # Template 3: Solid electrolyte
    electrolyte_config = SimulationConfig.from_template(
        template_name='solid_electrolyte',  # Optimized for conductivity  
        structure_file='nasicon.cif',
        initial_occupations=[1, -1, 1, -1],
        cluster_expansion_file='conductivity_lce.json',
        event_file='transport_events.json'
    )
    
    return ion_config, battery_config, electrolyte_config


# ===============================================
# METHOD 4: Parameter Variations - Easy Sweeps
# ===============================================
def method_4_parameter_sweeps():
    """
    Create parameter variations from base configuration.
    Good for: Systematic studies, optimization
    """
    
    # Create base configuration
    base_config = SimulationConfig.create(
        structure_file="base_structure.cif",
        cluster_expansion_file="base_lce.json",
        event_file="base_events.json",
        initial_occupations=[1, -1, 1, -1],
        name="BaseSimulation",
        temperature=300.0
    )
    
    # Temperature sweep
    temperature_series = []
    for temp in [200, 250, 300, 350, 400, 450, 500]:
        config = base_config.with_runtime_changes(
            temperature=temp,
            name=f"TempSweep_{temp}K"
        )
        temperature_series.append(config)
    
    # Supercell size sweep  
    supercell_series = []
    for size in [(1,1,1), (2,2,2), (3,3,3), (4,4,4)]:
        config = base_config.with_system_changes(
            supercell_shape=size
        )
        supercell_series.append(config)
    
    # Combined parameter sweep
    combined_series = []
    for temp in [300, 400, 500]:
        for size in [(2,2,2), (3,3,3)]:
            config = base_config.with_runtime_changes(temperature=temp) \
                               .with_system_changes(supercell_shape=size)
            config = config.with_runtime_changes(name=f"T{temp}_Size{size[0]}")
            combined_series.append(config)
    
    return temperature_series, supercell_series, combined_series


# ===============================================
# COMPARISON: Before vs After
# ===============================================
def comparison():
    """Show the difference between old way and new way."""
    
    print("=== BEFORE (Tedious) ===")
    print("""
    from kmcpy.simulator.config import SystemConfig, RuntimeConfig, SimulationConfig
    
    # Create system config
    system = SystemConfig(
        structure_file="nasicon.cif",
        supercell_shape=(2, 2, 2),
        dimension=3,
        mobile_ion_specie="Na",
        mobile_ion_charge=1.0,
        cluster_expansion_file="lce.json",
        event_file="events.json"
    )
    
    # Create runtime config  
    runtime = RuntimeConfig(
        name="NASICON_300K",
        temperature=300.0,
        attempt_frequency=1e13,
        equilibration_passes=1000,
        kmc_passes=50000
    )
    
    # Finally create simulation config
    config = SimulationConfig(
        system=system,
        runtime=runtime, 
        initial_occupations=(1, -1, 1, -1)
    )
    """)
    
    print("=== AFTER (Simple) ===")
    print("""
    from kmcpy.simulator.config import SimulationConfig
    
    config = SimulationConfig.create(
        structure_file="nasicon.cif",
        cluster_expansion_file="lce.json",
        event_file="events.json",
        initial_occupations=[1, -1, 1, -1],
        name="NASICON_300K",
        temperature=300.0,
        kmc_passes=50000,
        mobile_species="Na",
        supercell_shape=(2, 2, 2)
    )
    """)


# ===============================================
# AVAILABLE TEMPLATES  
# ===============================================
TEMPLATES = {
    'ion_diffusion': {
        'description': 'Optimized for ionic diffusion studies',
        'defaults': {
            'temperature': 300.0,
            'kmc_passes': 50000,
            'equilibration_passes': 5000,
            'mobile_species': 'Li'
        }
    },
    'battery': {
        'description': 'Optimized for battery material simulations',
        'defaults': {
            'temperature': 298.0,
            'kmc_passes': 100000,
            'equilibration_passes': 10000,
            'mobile_species': 'Li'
        }
    },
    'solid_electrolyte': {
        'description': 'Optimized for solid electrolyte conductivity',
        'defaults': {
            'temperature': 350.0,
            'kmc_passes': 200000,
            'equilibration_passes': 20000,
            'mobile_species': 'Na'
        }
    }
}


if __name__ == "__main__":
    print("SimulationConfig Factory Methods - Examples")
    print("=" * 50)
    
    comparison()
    
    print("\\nAvailable Templates:")
    for name, info in TEMPLATES.items():
        print(f"  {name}: {info['description']}")
        print(f"    Defaults: {info['defaults']}")
    
    print("\\n✓ Use these methods to create SimulationConfig easily!")
