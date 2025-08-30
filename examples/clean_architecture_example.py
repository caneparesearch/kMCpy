"""
Migration guide from old overlapping classes to clean architecture.

The problem with the old design:
- SimulationCondition: Only 4 fields, unclear purpose
- SimulationConfig: Inheritance hell, mixed concerns (config + initial state)  
- SimulationState: Confused responsibilities

New clean architecture:
- SystemConfig: WHAT you're simulating (physical system)
- RuntimeConfig: HOW you're simulating (algorithm parameters)
- SimulationConfig: Complete immutable configuration
- SimulationState: Pure mutable state during execution
"""

from pathlib import Path
from kmcpy.simulator.config import SystemConfig, RuntimeConfig, SimulationConfig  
from kmcpy.simulator.state_clean import SimulationState


# Example: How to create a proper simulation setup
def create_simulation_example():
    """Example of proper simulation setup with clean architecture."""
    
    # Step 1: Define the physical system (WHAT you're simulating)
    system = SystemConfig(
        structure_file="nasicon.cif",
        supercell_shape=(2, 2, 2),
        dimension=3,
        mobile_ion_specie="Na",
        mobile_ion_charge=1.0,
        cluster_expansion_file="lce.json",
        event_file="events.json"
    )
    
    # Step 2: Define runtime parameters (HOW you're simulating) 
    runtime = RuntimeConfig(
        name="NASICON_300K",
        temperature=300.0,
        attempt_frequency=1e13,
        equilibration_passes=1000,
        kmc_passes=50000,
        random_seed=42
    )
    
    # Step 3: Combine into complete configuration
    config = SimulationConfig(
        system=system,
        runtime=runtime,
        initial_state_file="initial_state.json"  # OR initial_occupations=(1, -1, 1, -1, ...)
    )
    
    return config


def run_simulation_example():
    """Example of running simulation with clean state management."""
    
    config = create_simulation_example()
    
    # Convert to InputSet for compatibility with existing code
    inputset = config.to_inputset()
    
    # Create initial state from configuration
    # (In real code, this would come from loading initial_state_file)
    initial_occupations = [1, -1, 1, -1, 1, -1]  # Example
    state = SimulationState(
        occupations=initial_occupations,
        time=0.0,
        step=0
    )
    
    # Initialize tracking if needed
    # state.initialize_tracking(structure_data)
    
    # Simulation loop (simplified)
    for kmc_step in range(100):  # Just an example
        # Select and execute event (this would be done by KMC engine)
        from_site, to_site = 0, 1  # Example transition
        dt = 1e-6  # Example time increment
        
        # Update state
        state.apply_event(from_site, to_site, dt)
        
        # Optionally save checkpoints
        if kmc_step % 10 == 0:
            state.save_checkpoint(f"checkpoint_{kmc_step}.json")
    
    # Get final statistics
    stats = state.get_statistics()
    print(f"Final stats: {stats}")
    
    return state


def migration_from_old_code():
    """How to migrate from old SimulationConfig to new architecture."""
    
    # OLD WAY (messy inheritance, mixed concerns):
    # from kmcpy.simulator.condition import SimulationConfig as OldConfig
    # old_config = OldConfig(
    #     name="test",
    #     temperature=300,
    #     template_structure_fname="structure.cif",
    #     event_fname="events.json", 
    #     initial_occ=[1, -1, 1, -1]
    # )
    
    # NEW WAY (clean separation):
    
    # 1. Split into system vs runtime concerns
    system = SystemConfig(
        structure_file="structure.cif",
        event_file="events.json",
        mobile_ion_specie="Li"
    )
    
    runtime = RuntimeConfig(
        name="test",
        temperature=300.0
    )
    
    # 2. Create clean configuration 
    new_config = SimulationConfig(
        system=system,
        runtime=runtime,
        initial_occupations=(1, -1, 1, -1)  # Use tuple for immutability
    )
    
    # 3. State is managed separately - no more mixed concerns
    state = SimulationState(
        occupations=[1, -1, 1, -1]  # Mutable copy for execution
    )
    
    return new_config, state


def parameter_updates_example():
    """How to create variations of simulations cleanly."""
    
    base_config = create_simulation_example()
    
    # Create temperature series (runtime parameter changes)
    temperatures = [200, 300, 400, 500]
    temp_configs = [
        base_config.with_runtime_changes(
            temperature=T,
            name=f"NASICON_{T}K"
        ) 
        for T in temperatures
    ]
    
    # Create supercell size series (system parameter changes)
    supercell_configs = [
        base_config.with_system_changes(
            supercell_shape=(n, n, n)
        )
        for n in [1, 2, 3, 4]
    ]
    
    return temp_configs, supercell_configs


def checkpoint_restart_example():
    """How to handle checkpointing and restart properly."""
    
    config = create_simulation_example()
    
    # Start fresh simulation
    initial_state = SimulationState(
        occupations=[1, -1, 1, -1, 1, -1]
    )
    
    # Run for a while and save checkpoint
    for step in range(100):
        # ... run KMC steps ...
        initial_state.apply_hop(0, 1, 1e-6)  # Example
    
    initial_state.save_checkpoint("checkpoint.json")
    
    # Later: restart from checkpoint
    # NOTE: Configuration stays the same, only state is restored
    restarted_state = SimulationState.load_checkpoint("checkpoint.json")
    
    # Continue simulation with same config, restored state
    print(f"Restarted from: {restarted_state}")
    
    return config, restarted_state


if __name__ == "__main__":
    # Test the examples
    print("=== Creating clean simulation setup ===")
    config = create_simulation_example()
    print(f"Config: {config.summary()}")
    
    print("\n=== Running simulation example ===") 
    final_state = run_simulation_example()
    print(f"Final state: {final_state}")
    
    print("\n=== Parameter variations ===")
    temp_configs, size_configs = parameter_updates_example()
    print(f"Temperature series: {[c.runtime.temperature for c in temp_configs]}")
    print(f"Size series: {[c.system.supercell_shape for c in size_configs]}")
