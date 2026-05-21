# CLAUDE.md

## Overview
This repository implements kmcpy, a Python framework for kinetic Monte Carlo (kMC) simulations of ion transport in crystalline solids.  
It provides modular components for lattice construction, event generation, and simulation control, with interfaces for machine-learning-based models and external packages.

The package supports CLI execution, YAML-based configuration, and reproducible simulations for systems like NASICON-type materials.

## Repository Structure
kmcpy/
│
├── cli/ # Command-line interface tools
├── event/ # Event generation and dependency management
├── external/ # Interfaces to external codes or data
├── io/ # Input/output and serialization utilities
├── models/ # Energy, hopping, and machine learning models
├── simulator/ # Core kinetic Monte Carlo loop and simulation logic
├── structure/ # Lattice and structure representation
├── tools/ # Utility functions and post-processing
├── fitting.py # Model fitting and regression utilities
└── _version.py # Version information

Other important directories:
- `example/`: Example input files and Jupyter notebooks (`NASICON.ipynb`, YAML configs)
- `tests/`: Comprehensive `pytest` test suite
- `docs/`: Sphinx documentation
- `scripts/`: Utilities for documentation and citation updates

## Design Philosophy
- Modular — each subpackage handles a single concern.
- Composable — simulations are built from modular models, events, and structures.
- Reproducible — all configurations are defined in YAML and serialized outputs.
- Extendable — new materials, event types, or energy models can be added easily.
- 
## Core Logic
| Component | Description | Key Files |
|------------|--------------|-----------|
| Simulator | Main event loop and time evolution | `kmcpy/simulator/` |
| Event Management | Event generation, rate calculation, and dependency resolution | `kmcpy/event/` |
| Structure | Lattice and site occupation representation | `kmcpy/structure/` |
| Models | Physical and ML-based rate/energy models | `kmcpy/models/` |
| I/O | YAML config parsing, data export | `kmcpy/io/` |
| CLI | Entry points for running simulations from terminal | `kmcpy/cli/` |

## Development Guidelines
- Follow PEP8 and use type hints for public functions.
- Use Google-style docstrings.
- Run `pytest` before committing (`pytest -v --maxfail=1`).
- Avoid modifying `kmcpy/simulator` core unless explicitly requested.
- Use `logging` instead of `print()` for all runtime messages.
- Maintain deterministic outputs when random seeds are used.

## Environment Setup
### Installation
```bash
uv pip install -e .
```
### Docs build
```bash
python scripts/build_docs.py
```
### Running Tests
```bash
pytest
```
