"""
Clean simulation configuration classes with clear separation of concerns.

Architecture:

- SystemConfig: Physical system definition (immutable)
- RuntimeConfig: Simulation runtime fields (immutable)
- Configuration: Complete simulation setup (immutable)
- State: Mutable state during execution

This module provides field routing and configuration management for kinetic
Monte Carlo simulations. It handles both system fields (what you're simulating)
and runtime fields (how you run the simulation).
"""

from typing import Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from monty.serialization import dumpfn, loadfn
from kmcpy.simulator.property import BUILTIN_PROPERTY_FIELDS, validate_schedule
from kmcpy.units import UNIT_CONVENTIONS

LOADER_FIELD_NAMES = {
    "structure_file",
    "model_file",
    "event_file",
    "initial_state_file",
}

SYSTEM_FIELD_NAMES = {
    "structure_file",
    "supercell_shape",
    "dimension",
    "mobile_ion_specie",
    "mobile_ion_charge",
    "elementary_hop_distance",
    "model_type",
    "model_file",
    "event_file",
    "site_mapping",
    "convert_to_primitive_cell",
    "initial_state_file",
    "initial_occupations",
}

RUNTIME_FIELD_NAMES = {
    "temperature",
    "attempt_frequency",
    "equilibration_passes",
    "kmc_passes",
    "random_seed",
    "name",
    "property_sampling_interval",
    "property_sampling_time_interval",
    "builtin_property_enabled",
    "property_callbacks",
}

CONFIG_FIELD_NAMES = SYSTEM_FIELD_NAMES | RUNTIME_FIELD_NAMES

SYSTEM_FIELD_UNITS = {
    "dimension": UNIT_CONVENTIONS["dimension"],
    "mobile_ion_charge": UNIT_CONVENTIONS["mobile_ion_charge"],
    "elementary_hop_distance": UNIT_CONVENTIONS["elementary_hop_distance"],
}

RUNTIME_FIELD_UNITS = {
    "temperature": UNIT_CONVENTIONS["temperature"],
    "attempt_frequency": UNIT_CONVENTIONS["attempt_frequency"],
    "property_sampling_time_interval": UNIT_CONVENTIONS[
        "property_sampling_time_interval"
    ],
}

CONFIG_FIELD_UNITS = {
    **SYSTEM_FIELD_UNITS,
    **RUNTIME_FIELD_UNITS,
}


def _detect_config_file_format(filepath: str) -> str:
    suffix = Path(filepath).suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    return "unknown"


def _extract_section_data(
    raw_data: dict[str, Any],
    filepath: str | Path,
    section: str,
    task_type: str | None = None,
) -> dict[str, Any]:
    """Extract a flat or registry-style section from loaded file data."""
    if section not in raw_data:
        available = list(raw_data.keys())
        raise ValueError(
            f"Section '{section}' not found in {filepath}. Available: {available}"
        )

    section_data = raw_data[section]

    if isinstance(section_data, dict) and "type" in section_data:
        if task_type is None:
            task_type = section_data["type"]

        if task_type not in section_data:
            available_types = [key for key in section_data.keys() if key != "type"]
            raise ValueError(
                f"Task type '{task_type}' not found in section '{section}'. "
                f"Available: {available_types}"
            )

        section_data = section_data[task_type]

    if not isinstance(section_data, dict):
        raise ValueError(
            f"Selected section '{section}' in {filepath} must be a dictionary"
        )

    return section_data.copy()


def _validate_builtin_property_enabled(values: dict[str, bool]) -> None:
    """Validate built-in property enable/disable map."""
    if not isinstance(values, dict):
        raise TypeError("builtin_property_enabled must be a dictionary")
    for key, enabled in values.items():
        if key not in BUILTIN_PROPERTY_FIELDS:
            raise ValueError(
                f"Unknown built-in property '{key}'. "
                f"Supported: {list(BUILTIN_PROPERTY_FIELDS)}"
            )
        if not isinstance(enabled, bool):
            raise TypeError(
                f"builtin_property_enabled['{key}'] must be a boolean"
            )


@dataclass(frozen=True)
class SystemConfig:
    """
    Physical system configuration - completely immutable.
    This defines WHAT you're simulating.
    """
    # Structure definition
    structure_file: str = ""
    supercell_shape: tuple[int, int, int] = (1, 1, 1)
    dimension: int = 3  # dimensionless
    
    # Mobile ion properties
    mobile_ion_specie: str = "Li"
    mobile_ion_charge: float = 1.0  # |e|
    elementary_hop_distance: float = 1.0  # Angstrom
    
    # Model configuration
    model_type: str = "composite_lce"
    model_file: str = ""
    event_file: str = ""
    # Site-space definition
    site_mapping: Optional[dict] = None
    convert_to_primitive_cell: bool = False
    
    # Initial state specification
    initial_state_file: Optional[str] = None
    initial_occupations: Optional[list] = None
    
    def __post_init__(self):
        """Validate system configuration."""
        if self.dimension not in [1, 2, 3]:
            raise ValueError(f"Invalid dimension: {self.dimension}")
        
        if len(self.supercell_shape) != 3:
            raise ValueError("Supercell shape must have 3 components")
        
        # Temporarily disabled for testing
        # if not Path(self.structure_file).exists():
        #     raise FileNotFoundError(f"Structure file not found: {self.structure_file}")
        # 
        # if not Path(self.event_file).exists():
        #     raise FileNotFoundError(f"Event file not found: {self.event_file}")
    
    def as_dict(self, include_loader_paths: bool = False) -> dict[str, Any]:
        """Convert to a JSON/YAML-serializable dictionary.

        File path fields are loader inputs, not intrinsic system metadata. They
        are omitted by default from recorded payloads and can be included when
        writing a reloadable input file.
        """
        from dataclasses import asdict

        data = asdict(self)
        # Convert tuple back to list for compatibility.
        data["supercell_shape"] = list(self.supercell_shape)
        if not include_loader_paths:
            for key in LOADER_FIELD_NAMES:
                data.pop(key, None)
        return data


@dataclass(frozen=True)
class RuntimeConfig:
    """
    Simulation runtime configuration - completely immutable.
    This defines HOW you're simulating.
    """
    # Thermodynamic conditions
    temperature: float = 300.0  # K
    attempt_frequency: float = 1e13  # Hz
    
    # KMC algorithm fields
    equilibration_passes: int = 1000
    kmc_passes: int = 10000
    random_seed: Optional[int] = None
    
    # Simulation identification
    name: str = "DefaultSimulation"

    # Optional property sampling controls
    property_sampling_interval: Optional[int] = None
    property_sampling_time_interval: Optional[float] = None  # s
    builtin_property_enabled: dict[str, bool] = field(default_factory=dict)
    property_callbacks: list[dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate runtime fields."""
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        if self.attempt_frequency <= 0:
            raise ValueError("Attempt frequency must be positive")
        
        if self.equilibration_passes < 0:
            raise ValueError("Equilibration passes must be non-negative")
        
        if self.kmc_passes <= 0:
            raise ValueError("KMC passes must be positive")

        validate_schedule(
            interval=self.property_sampling_interval,
            time_interval=self.property_sampling_time_interval,
        )

        _validate_builtin_property_enabled(self.builtin_property_enabled)

        if not isinstance(self.property_callbacks, list):
            raise TypeError("property_callbacks must be a list")
        for callback_spec in self.property_callbacks:
            if not isinstance(callback_spec, dict):
                raise TypeError("Each property callback spec must be a dictionary")
            allowed_keys = {
                "callable",
                "name",
                "interval",
                "time_interval",
                "store",
                "max_records",
                "enabled",
            }
            unknown_keys = set(callback_spec.keys()) - allowed_keys
            if unknown_keys:
                raise ValueError(
                    f"Unknown keys in property callback spec: {sorted(unknown_keys)}"
                )
            callable_ref = callback_spec.get("callable")
            if not isinstance(callable_ref, str) or not callable_ref.strip():
                raise ValueError(
                    "Each property callback spec must include a non-empty 'callable' field"
                )
            validate_schedule(
                interval=callback_spec.get("interval"),
                time_interval=callback_spec.get("time_interval"),
            )
            max_records = callback_spec.get("max_records")
            if max_records is not None:
                if not isinstance(max_records, int):
                    raise TypeError("property callback max_records must be an integer")
                if max_records <= 0:
                    raise ValueError("property callback max_records must be positive")
            for bool_key in ("store", "enabled"):
                if bool_key in callback_spec and not isinstance(callback_spec[bool_key], bool):
                    raise TypeError(
                        f"property callback '{bool_key}' must be a boolean when provided"
                    )
    
    def as_dict(self) -> dict[str, Any]:
        """Convert to a JSON/YAML-serializable dictionary."""
        return {
            "temperature": self.temperature,
            "attempt_frequency": self.attempt_frequency,
            "equilibration_passes": self.equilibration_passes,
            "kmc_passes": self.kmc_passes,
            "random_seed": self.random_seed,
            "name": self.name,
            "property_sampling_interval": self.property_sampling_interval,
            "property_sampling_time_interval": self.property_sampling_time_interval,
            "builtin_property_enabled": dict(self.builtin_property_enabled),
            "property_callbacks": [dict(callback) for callback in self.property_callbacks],
        }


@dataclass(frozen=True)
class Configuration:
    """Complete simulation configuration combining system and runtime fields."""
    
    system_config: SystemConfig
    runtime_config: RuntimeConfig
    
    def __init__(self, system_config=None, runtime_config=None, **kwargs):
        """
        Create Configuration with automatic field routing.
        
        You can either:
        1. Pass pre-built configs: Configuration(system_config=sys, runtime_config=run)
        2. Pass fields directly: Configuration(temperature=300, structure_file="x.cif", ...)
        3. Mix both: Configuration(system_config=sys, temperature=400)
        
        Fields are automatically routed to SystemConfig or RuntimeConfig by name.
        """
        if system_config is None and runtime_config is None and not kwargs:
            raise ValueError("Must provide either configs or fields")

        # Split kwargs into system and runtime fields.
        system_fields = {}
        runtime_fields = {}
        unknown_fields = {}
        
        # Route fields.
        for key, value in kwargs.items():
            if key in SYSTEM_FIELD_NAMES:
                system_fields[key] = value
            elif key in RUNTIME_FIELD_NAMES:
                runtime_fields[key] = value
            else:
                unknown_fields[key] = value
        
        # Reject unknown fields; legacy aliases are intentionally unsupported.
        if unknown_fields:
            raise ValueError(
                f"Unknown configuration fields: {list(unknown_fields.keys())}. "
                "Use Configuration.help_fields() to see valid fields."
            )
        
        # Create or update configs
        if system_config is None:
            system_config = SystemConfig(**system_fields)
        elif system_fields:
            # Update existing system config with new fields.
            from dataclasses import replace
            system_config = replace(system_config, **system_fields)
        
        if runtime_config is None:
            runtime_config = RuntimeConfig(**runtime_fields)
        elif runtime_fields:
            # Update existing runtime config with new fields.
            from dataclasses import replace
            runtime_config = replace(runtime_config, **runtime_fields)
        
        # Set the attributes using object.__setattr__ since the class is frozen
        object.__setattr__(self, 'system_config', system_config)
        object.__setattr__(self, 'runtime_config', runtime_config)
    
    @classmethod
    def create(cls, **kwargs):
        """
        Alternative factory method for cleaner API.
        
        Examples::
        
            config = Configuration.create(
                structure_file="test.cif",
                temperature=400.0,
                kmc_passes=50000
            )
        """
        return cls(**kwargs)
    
    
    def as_dict(self, include_loader_paths: bool = False) -> dict[str, Any]:
        """Convert to a recorded dictionary.

        By default, loader-only path fields such as structure_file, model_file,
        event_file, and initial_state_file are omitted because they are only
        needed to construct the loaded simulation inputs. Use
        ``include_loader_paths=True`` when writing a reloadable simulation input
        file.
        """
        result = {}
        result.update(
            self.system_config.as_dict(include_loader_paths=include_loader_paths)
        )
        result.update(self.runtime_config.as_dict())
        return result

    def as_record_dict(self) -> dict[str, Any]:
        """Return metadata suitable for recording after simulation inputs are loaded."""
        return self.as_dict(include_loader_paths=False)

    def as_input_dict(self) -> dict[str, Any]:
        """Return a reloadable input dictionary including loader path fields."""
        return self.as_dict(include_loader_paths=True)

    @classmethod
    def field_units(cls) -> dict[str, str]:
        """Return configured units for numeric configuration fields."""
        return dict(CONFIG_FIELD_UNITS)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Configuration":
        """Create from dictionary."""
        if not isinstance(config_dict, dict):
            raise ValueError("Configuration.from_dict expects a dictionary")
        config_dict = dict(config_dict)

        return cls(**config_dict)

    @classmethod
    def _extract_file_config_data(
        cls,
        raw_data: dict[str, Any],
        filepath: str | Path,
        section: str | None = None,
        task_type: str | None = None,
    ) -> dict[str, Any]:
        """Normalize flat or generated-template file payloads to config fields."""
        if not isinstance(raw_data, dict):
            raise ValueError(f"Configuration file {filepath} must contain a dictionary")

        if section is not None:
            return _extract_section_data(raw_data, filepath, section, task_type)

        if CONFIG_FIELD_NAMES.intersection(raw_data.keys()):
            return raw_data

        if "kmc" not in raw_data:
            return raw_data

        return _extract_section_data(raw_data, filepath, "kmc", task_type)
    
    # ===== FILE I/O METHODS =====
    
    @classmethod
    def from_file(
        cls,
        filename: str | Path,
        section: str | None = None,
        task_type: str | None = None,
    ) -> "Configuration":
        """Load Configuration from a JSON/YAML file.

        Args:
            filename: Path to configuration file (.json, .yaml, .yml).
            section: Optional top-level section to load from a YAML workflow file.
            task_type: Optional task type for registry-style sections.
        """
        file_format = _detect_config_file_format(str(filename))

        if file_format == "json":
            raw_data = loadfn(filename, cls=None)
        elif file_format == "yaml":
            raw_data = loadfn(filename)
        else:
            raise ValueError(
                f"Unsupported file format for {filename}. Supported: .json, .yaml, .yml"
            )

        config_data = cls._extract_file_config_data(
            raw_data, filename, section=section, task_type=task_type
        )
        return cls.from_dict(config_data)
    
    def to(
        self,
        filename: str | Path,
        include_loader_paths: bool = False,
        section: str | None = None,
        task_type: str = "default",
        **kwargs,
    ) -> None:
        """Write Configuration to a JSON/YAML file.

        Args:
            filename: Output file path (.json, .yaml, .yml).
            include_loader_paths: Include loader-only path fields for reloadable
                input files. The default recorded form omits them.
            section: Optional top-level YAML section for workflow files.
            task_type: Registry-style task type when ``section`` is provided.
            **kwargs: Additional formatting arguments, such as ``indent`` for JSON.
        """
        file_format = _detect_config_file_format(str(filename))
        if file_format not in {"json", "yaml"}:
            raise ValueError(
                f"Unsupported file format for {filename}. Supported: .json, .yaml, .yml"
            )

        config_data = self.as_dict(include_loader_paths=include_loader_paths)

        if section is not None:
            if file_format != "yaml":
                raise ValueError("section output is only supported for YAML files")
            path = Path(filename)
            if path.exists():
                try:
                    yaml_data = loadfn(path)
                except Exception:
                    yaml_data = {}
            else:
                yaml_data = {}
            if not isinstance(yaml_data, dict):
                raise ValueError(f"Existing YAML file {filename} must contain a dictionary")
            yaml_data[section] = {"type": task_type, task_type: config_data}
            dumpfn(yaml_data, filename)
            return

        if file_format == "json":
            indent = kwargs.get("indent", 2)
            dumpfn(config_data, filename, indent=indent)
        else:
            dumpfn(config_data, filename)
    
    def with_runtime_changes(self, **changes) -> "Configuration":
        """Create new config with runtime field changes."""
        from dataclasses import replace
        new_runtime = replace(self.runtime_config, **changes)
        return replace(self, runtime_config=new_runtime)
    
    def with_system_changes(self, **changes) -> "Configuration":
        """Create new config with system field changes."""
        from dataclasses import replace
        new_system = replace(self.system_config, **changes)
        return replace(self, system_config=new_system)
    
    def summary(self) -> str:
        """Human-readable summary."""
        system_name = (
            Path(self.system_config.structure_file).name
            if self.system_config.structure_file
            else "loaded"
        )
        return (
            f"{self.runtime_config.name}: "
            f"T={self.runtime_config.temperature}K, "
            f"passes={self.runtime_config.kmc_passes}, "
            f"system={system_name}"
        )
    
    # ===== CONVENIENT PROPERTY ACCESS =====
    # Users don't need to remember which config contains what field.
    
    # Runtime properties
    @property
    def temperature(self) -> float:
        """Access temperature directly."""
        return self.runtime_config.temperature
    
    @property
    def name(self) -> str:
        """Access simulation name directly."""
        return self.runtime_config.name
    
    @property
    def kmc_passes(self) -> int:
        """Access KMC passes directly."""
        return self.runtime_config.kmc_passes
    
    @property
    def equilibration_passes(self) -> int:
        """Access equilibration passes directly."""
        return self.runtime_config.equilibration_passes
    
    @property
    def attempt_frequency(self) -> float:
        """Access attempt frequency directly."""
        return self.runtime_config.attempt_frequency
    
    @property
    def random_seed(self) -> Optional[int]:
        """Access random seed directly."""
        return self.runtime_config.random_seed

    @property
    def property_sampling_interval(self) -> Optional[int]:
        """Access global property sampling step interval directly."""
        return self.runtime_config.property_sampling_interval

    @property
    def property_sampling_time_interval(self) -> Optional[float]:
        """Access global property sampling time interval directly."""
        return self.runtime_config.property_sampling_time_interval

    @property
    def builtin_property_enabled(self) -> dict[str, bool]:
        """Access built-in property enable/disable map directly."""
        return self.runtime_config.builtin_property_enabled

    @property
    def property_callbacks(self) -> list[dict[str, Any]]:
        """Access callback attachment specs directly."""
        return self.runtime_config.property_callbacks
    
    # System properties
    @property
    def structure_file(self) -> str:
        """Access structure file directly."""
        return self.system_config.structure_file
    
    @property
    def mobile_ion_specie(self) -> str:
        """Access mobile ion species directly."""
        return self.system_config.mobile_ion_specie
    
    @property
    def supercell_shape(self) -> tuple[int, int, int]:
        """Access supercell shape directly."""
        return self.system_config.supercell_shape
    
    @property
    def dimension(self) -> int:
        """Access dimension directly."""
        return self.system_config.dimension
    
    @property
    def model_type(self) -> str:
        """Access model type directly."""
        return self.system_config.model_type
    
    @property
    def model_file(self) -> str:
        """Access model file directly."""
        return self.system_config.model_file
    
    @property
    def event_file(self) -> str:
        """Access event file directly."""
        return self.system_config.event_file
    
    @property
    def site_mapping(self) -> Optional[dict]:
        """Access site mapping directly."""
        return self.system_config.site_mapping
    
    @property
    def elementary_hop_distance(self) -> float:
        """Access elementary hop distance directly."""
        return self.system_config.elementary_hop_distance
    
    @property  
    def mobile_ion_charge(self) -> float:
        """Access mobile ion charge directly."""
        return self.system_config.mobile_ion_charge
    
    @property
    def convert_to_primitive_cell(self) -> bool:
        """Access convert to primitive cell directly."""
        return self.system_config.convert_to_primitive_cell
    
    @property
    def initial_state_file(self) -> Optional[str]:
        """Access initial state file directly."""
        return self.system_config.initial_state_file
    
    @property
    def initial_occupations(self) -> Optional[list]:
        """Access initial occupations directly."""
        return self.system_config.initial_occupations
    
    # ===== HELPER METHODS =====
    
    @classmethod
    def help_fields(cls):
        """Print available configuration fields and where they are routed."""
        print("Configuration Fields:\n")

        print("SYSTEM FIELDS (physical setup):")
        system_fields = sorted(SYSTEM_FIELD_NAMES)
        for field_name in system_fields:
            unit = CONFIG_FIELD_UNITS.get(field_name)
            suffix = f" [{unit}]" if unit else ""
            print(f"  - {field_name}{suffix}")

        print("\nRUNTIME FIELDS (simulation settings):")
        runtime_fields = sorted(RUNTIME_FIELD_NAMES)
        for field_name in runtime_fields:
            unit = CONFIG_FIELD_UNITS.get(field_name)
            suffix = f" [{unit}]" if unit else ""
            print(f"  - {field_name}{suffix}")

        print("\nUsage examples:")
        print("  config = Configuration(structure_file='x.cif', temperature=400)")
        print("  config = Configuration.create(temperature=300, kmc_passes=10000)")
        print("  print(config.temperature)  # Direct access to any field")

    @classmethod
    def help_parameters(cls):
        """Compatibility alias for :meth:`help_fields`."""
        cls.help_fields()

    def which_config(self, field_name: str) -> str:
        """Show which sub-config contains a field."""
        if field_name in SYSTEM_FIELD_NAMES:
            return f"'{field_name}' is in system_config (physical setup)"
        elif field_name in RUNTIME_FIELD_NAMES:
            return f"'{field_name}' is in runtime_config (simulation settings)"
        else:
            return f"'{field_name}' is not a recognized configuration field"

    def validate(self) -> bool:
        """Validate the configuration."""
        try:
            # Basic validation - configs validate themselves in __post_init__
            # This could be expanded with more complex cross-field validation.
            return True
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def copy_with_changes(self, **changes) -> "Configuration":
        """Create a copy of this config with some fields changed.
        
        Args:
            **changes: Field changes to apply.
            
        Returns:
            Configuration: New config with changes applied
        """
        # Get current config as a reloadable dict so loader paths survive copies.
        current_dict = self.as_dict(include_loader_paths=True)
        
        # Apply changes
        current_dict.update(changes)
        
        # Create new config
        return Configuration.from_dict(current_dict)
