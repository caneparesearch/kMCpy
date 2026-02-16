# Agents.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**kMCpy** (Kinetic Monte Carlo Simulation with Python) is a scientific computing package for studying ion diffusion in crystalline materials using kinetic Monte Carlo techniques. It's designed for computational materials science research, particularly for battery electrolytes and transport properties.

## Common Development Commands

### Installation and Setup
```bash
# Standard installation
pip install .

# Development installation (editable mode with dev dependencies)
pip install -e ".[dev]"

# Using UV (recommended package manager)
uv sync                    # Install all dependencies
uv sync --extra dev       # Include dev dependencies
uv pip install -e .       # Editable installation

# GUI dependencies (deprecated GUI)
pip install -e ".[gui]"
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit            # Unit tests only
pytest -m integration     # Integration tests only
pytest -m slow            # Slow tests (use -m "not slow" to exclude)
pytest -m nasicon         # NASICON-specific tests

# Run specific test file
pytest tests/test_modern_event_generator.py

# Run with coverage
pytest --cov=kmcpy --cov-report=html
```

### Building Documentation
```bash
# Install documentation dependencies
uv sync --extra doc

# Build documentation locally
python scripts/build_doc.py
```

### Running Simulations
```bash
# Command line interface
run_kmc input.json
run_kmc input.yaml
run_kmc --help

# Deprecated GUI
start_kmcpy_gui
```

## Architecture Overview

kMCpy follows a modular architecture with these core systems:

### 1. Models System (`/kmcpy/models/`)
- **BaseModel**: Abstract base for all models
- **LocalClusterExpansion**: Core implementation of local cluster expansion
- **CompositeLCEModel**: Combines multiple LCE models
- **Fitting**: Model fitting infrastructure with LCEFitter

### 2. Event System (`/kmcpy/event/`)
- **Event**: Individual migration events with barriers and rates
- **EventLib**: Event libraries and collections
- **ModernEventGenerator**: Primary event generation implementation
- **NeighborInfoMatcher**: Crystallographic neighbor identification

### 3. Simulator Engine (`/kmcpy/simulator/`)
- **KMC**: Main kinetic Monte Carlo simulation class
- **SimulationConfig**: Configuration management (SystemConfig, RuntimeConfig)
- **SimulationState**: State tracking during simulations
- **Tracker**: Results collection and data export

### 4. Structure System (`/kmcpy/structure/`)
- **Basis**: Crystal structure basis and symmetry operations
- **LatticeStructure**: Lattice geometry and periodic boundaries
- **LocalEnvironmentComparator**: Environment matching for cluster identification

### 5. I/O System (`/kmcpy/io/`)
- **ConfigIO**: JSON/YAML configuration file handling
- **DataLoader**: Multi-format input data loading
- **IO**: General input/output utilities

## Key Development Patterns

### Testing Framework
- Uses **pytest** with comprehensive markers: `unit`, `integration`, `slow`, `nasicon`
- Test files follow `test_*.py` naming convention
- Integration tests often use fixture data in `/tests/files/`
- Import testing is crucial to prevent circular dependencies

### Performance Considerations
- **Numba** is extensively used for performance-critical code
- All computational kernels should be Numba-compatible
- Pay attention to type annotations for Numba optimization

### Configuration System
- JSON/YAML-based configuration files for simulations
- Modern system uses `SimulationConfigIO` and `SimulationConfig`
- CLI entry point is `run_kmc` script in `/kmcpy/cli/`

### Data Flow
1. **Input**: Crystal structure + migration barrier data
2. **Model Fitting**: Train LocalClusterExpansion model
3. **Event Generation**: Generate possible migration events
4. **KMC Simulation**: Run rejection-free kMC algorithm
5. **Output**: Transport properties (diffusivity, conductivity)

## Module Dependencies

The project has a carefully managed dependency structure:
- `external/`: Custom structure handling and CIF parsing
- `structure/`: Crystal structure fundamentals
- `models/`: Model implementations (depends on structure)
- `event/`: Event generation (depends on models and structure)
- `simulator/`: Core simulation engine (depends on all above)
- `io/`: Input/output handling
- `cli/`: Command-line interfaces

## Important Notes

- **Python 3.11+** is required
- **Clean design and maintainability** are priorities over compatibility
- The GUI is **deprecated** - focus development on CLI and API
- Performance is critical - prioritize Numba optimization, but maintain code clarity
- Comprehensive test coverage is required for new features
- Use `uv` for dependency management when possible
- Avoid legacy compatibility patterns that compromise code quality

## Entry Points

- **CLI**: `run_kmc input.json` - Command line simulation runner
- **API**: Import from `kmcpy` modules for programmatic use
- **Examples**: Check `/example/` directory for usage patterns