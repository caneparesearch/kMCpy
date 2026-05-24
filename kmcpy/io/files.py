"""Internal helpers for JSON and YAML configuration files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from kmcpy.io.serialization import to_json_compatible

logger = logging.getLogger(__name__)


def load_json(filepath: str) -> dict[str, Any]:
    """Load a raw dictionary from a JSON file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON configuration from {filepath}")
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}")


def load_yaml(filepath: str) -> dict[str, Any]:
    """Load a raw dictionary from a YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for YAML support. Install with: pip install PyYAML")

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    try:
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        logger.debug(f"Loaded YAML configuration from {filepath}")
        return data
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {filepath}: {e}")


def load_yaml_section(
    filepath: str,
    section: str,
    task_type: str | None = None,
) -> dict[str, Any]:
    """Load a flat or registry-style section from a YAML file."""
    yaml_data = load_yaml(filepath)

    if section not in yaml_data:
        available = list(yaml_data.keys())
        raise ValueError(f"Section '{section}' not found in {filepath}. Available: {available}")

    section_data = yaml_data[section]

    if isinstance(section_data, dict) and "type" in section_data:
        if task_type is None:
            task_type = section_data["type"]

        if task_type not in section_data:
            available_types = [k for k in section_data.keys() if k != "type"]
            raise ValueError(
                f"Task type '{task_type}' not found in section '{section}'. "
                f"Available: {available_types}"
            )

        parameters = section_data[task_type].copy()
        logger.debug(f"Loaded section '{section}' with task type '{task_type}' from {filepath}")
    else:
        parameters = section_data.copy()
        logger.debug(f"Loaded flat section '{section}' from {filepath}")

    return parameters


def save_json(data: dict[str, Any], filepath: str, indent: int = 2) -> None:
    """Save a dictionary to a JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=indent, default=json_serializer)
        logger.debug(f"Saved JSON configuration to {filepath}")
    except TypeError as e:
        raise ValueError(f"Cannot serialize configuration to JSON: {e}")


def save_yaml(data: dict[str, Any], filepath: str) -> None:
    """Save a dictionary to a YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for YAML support. Install with: pip install PyYAML")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(filepath, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, indent=2)
        logger.debug(f"Saved YAML configuration to {filepath}")
    except yaml.YAMLError as e:
        raise ValueError(f"Cannot serialize configuration to YAML: {e}")


def json_serializer(obj):
    """JSON serializer for numpy types and other supported objects."""
    return to_json_compatible(obj)


def detect_file_format(filepath: str) -> str:
    """Detect a file format from its extension."""
    suffix = Path(filepath).suffix.lower()

    if suffix == ".json":
        return "json"
    if suffix in [".yaml", ".yml"]:
        return "yaml"
    return "unknown"
