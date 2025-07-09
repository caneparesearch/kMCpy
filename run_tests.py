#!/usr/bin/env python3
"""
Test runner script for kMCpy test suite.
Provides easy commands to run different categories of tests.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="kMCpy Test Runner")
    parser.add_argument(
        "category", 
        choices=[
            "all", "unit", "integration", "development", "nasicon", 
            "fast", "slow", "coverage", "imports", "simulation", 
            "inputset", "kmc", "occupation"
        ],
        help="Test category to run"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no-capture", action="store_true", help="Don't capture output (useful for debugging)")
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        base_cmd.append("-v")
    
    if args.no_capture:
        base_cmd.append("-s")
    
    # Test category commands
    commands = {
        "all": {
            "cmd": base_cmd + ["tests/"],
            "desc": "All tests"
        },
        "unit": {
            "cmd": base_cmd + [
                "tests/test_simulation_condition.py",
                "tests/test_inputset.py", 
                "tests/test_kmc_integration.py",
                "tests/test_occupation_management.py",
                "tests/test_imports.py"
            ],
            "desc": "Unit tests"
        },
        "integration": {
            "cmd": base_cmd + [
                "tests/test_integration.py",
                "tests/test_nasicon_bulk.py",
                "tests/test_na3sbs4.py"
            ],
            "desc": "Integration tests"
        },
        "development": {
            "cmd": base_cmd + ["tests/test_phase_development.py"],
            "desc": "Development phase tests"
        },
        "nasicon": {
            "cmd": base_cmd + ["tests/", "-m", "nasicon"],
            "desc": "NASICON material tests"
        },
        "fast": {
            "cmd": base_cmd + ["tests/", "-m", "not slow"],
            "desc": "Fast tests (excluding slow integration tests)"
        },
        "slow": {
            "cmd": base_cmd + ["tests/", "-m", "slow"],
            "desc": "Slow integration tests"
        },
        "coverage": {
            "cmd": base_cmd + ["tests/", "--cov=kmcpy", "--cov-report=html", "--cov-report=term-missing"],
            "desc": "All tests with coverage report"
        },
        "imports": {
            "cmd": base_cmd + ["tests/test_imports.py"],
            "desc": "Import tests"
        },
        "simulation": {
            "cmd": base_cmd + ["tests/test_simulation_condition.py"],
            "desc": "SimulationCondition/Config/State tests"
        },
        "inputset": {
            "cmd": base_cmd + ["tests/test_inputset.py"],
            "desc": "InputSet tests"
        },
        "kmc": {
            "cmd": base_cmd + ["tests/test_kmc_integration.py"],
            "desc": "KMC integration tests"
        },
        "occupation": {
            "cmd": base_cmd + ["tests/test_occupation_management.py"],
            "desc": "Occupation management tests"
        }
    }
    
    if args.category not in commands:
        print(f"Unknown test category: {args.category}")
        return 1
    
    # Run the selected test category
    cmd_info = commands[args.category]
    success = run_command(cmd_info["cmd"], cmd_info["desc"])
    
    if not success:
        return 1
    
    # Print summary
    print(f"\n🎉 Test category '{args.category}' completed successfully!")
    
    if args.category == "coverage":
        print("\n📊 Coverage report generated in htmlcov/index.html")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
