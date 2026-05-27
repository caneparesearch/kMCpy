"""Fitted parameter records for kMCpy models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from monty.json import MSONable
from monty.serialization import dumpfn, loadfn


@dataclass
class LCEModelParameters(MSONable):
    """Fitted parameters for a :class:`LocalClusterExpansion` model."""

    keci: list[float]
    empty_cluster: float
    cluster_site_indices: list[int] | list[list[int]]
    weight: list[float]
    alpha: float
    time_stamp: float
    time: str
    rmse: float
    loocv: float
    normalize: bool = True
    orbit_fingerprints: list[str] | None = None
    local_environment_hash: str | None = None
    local_site_order: dict | None = None
    name: str = "LCEModelParameters"

    def as_dict(self) -> dict[str, Any]:
        """Return a Monty/pymatgen-style dictionary payload."""
        data = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "keci": self.keci,
            "empty_cluster": self.empty_cluster,
            "cluster_site_indices": self.cluster_site_indices,
            "weight": self.weight,
            "alpha": self.alpha,
            "time_stamp": self.time_stamp,
            "time": self.time,
            "rmse": self.rmse,
            "loocv": self.loocv,
            "normalize": self.normalize,
        }
        if self.orbit_fingerprints is not None:
            data["orbit_fingerprints"] = self.orbit_fingerprints
        if self.local_environment_hash is not None:
            data["local_environment_hash"] = self.local_environment_hash
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LCEModelParameters":
        """Create fitted parameters from a dictionary payload."""
        if not isinstance(data, dict):
            raise ValueError("LCEModelParameters.from_dict expects a dictionary")
        payload = {
            key: value
            for key, value in data.items()
            if not key.startswith("@") and key != "name"
        }
        return cls(
            keci=payload.get("keci", []),
            empty_cluster=payload.get("empty_cluster", 0.0),
            cluster_site_indices=payload.get("cluster_site_indices", []),
            weight=payload.get("weight", []),
            alpha=payload.get("alpha", 0.0),
            time_stamp=payload.get("time_stamp", ""),
            time=payload.get("time", ""),
            rmse=payload.get("rmse", 0.0),
            loocv=payload.get("loocv", 0.0),
            normalize=payload.get("normalize", True),
            orbit_fingerprints=payload.get("orbit_fingerprints"),
            local_environment_hash=payload.get("local_environment_hash"),
            local_site_order=payload.get("local_site_order"),
        )

    @classmethod
    def from_file(cls, filename: str | Path) -> "LCEModelParameters":
        """Load fitted parameters from JSON/YAML or HDF5."""
        filename = Path(filename)
        if filename.suffix == ".h5":
            return cls.from_dict(_read_hdf5_parameter_group(filename))
        return cls.from_dict(loadfn(filename, cls=None))

    def to(self, filename: str | Path, indent: int = 4) -> None:
        """Write fitted parameters to JSON/YAML or HDF5."""
        filename = Path(filename)
        if filename.suffix == ".h5":
            _write_hdf5_parameter_group(filename, self.as_dict())
            return
        dumpfn(self.as_dict(), filename, indent=indent)

    def __str__(self) -> str:
        values = ", ".join(
            f"{key}={value}" for key, value in self.as_dict().items()
            if not key.startswith("@")
        )
        return f"{self.name}: {values}"


@dataclass
class LCEModelParamHistory(MSONable):
    """Ordered fitted-parameter records from repeated LCE fits."""

    history: list[LCEModelParameters] = field(default_factory=list)

    def append(self, parameters: LCEModelParameters) -> None:
        """Add one fitted parameter record."""
        self.history.append(parameters)

    def as_dict(self) -> dict[str, Any]:
        """Return a Monty/pymatgen-style dictionary payload."""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "history": [parameters.as_dict() for parameters in self.history],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | list[dict[str, Any]],
    ) -> "LCEModelParamHistory":
        """Create a history from either the current or legacy list payload."""
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = data.get("history", [])
        else:
            raise ValueError(
                "LCEModelParamHistory.from_dict expects a dictionary or list"
            )
        return cls(history=[LCEModelParameters.from_dict(record) for record in records])

    @classmethod
    def from_file(cls, filename: str | Path) -> "LCEModelParamHistory":
        """Load fitted-parameter history from JSON/YAML or HDF5."""
        filename = Path(filename)
        if filename.suffix == ".h5":
            records = []
            try:
                import h5py
            except ImportError as exc:
                raise ImportError("h5py required for HDF5 parameter history") from exc
            with h5py.File(filename, "r") as h5file:
                for key in sorted(h5file.keys()):
                    records.append(_read_hdf5_group(h5file[key]))
            return cls.from_dict(records)
        return cls.from_dict(loadfn(filename, cls=None))

    def to(self, filename: str | Path, indent: int = 4) -> None:
        """Write fitted-parameter history to JSON/YAML or HDF5."""
        filename = Path(filename)
        if filename.suffix == ".h5":
            try:
                import h5py
            except ImportError as exc:
                raise ImportError("h5py required for HDF5 parameter history") from exc
            with h5py.File(filename, "w") as h5file:
                for index, parameters in enumerate(self.history):
                    group = h5file.create_group(f"parameter_set_{index}")
                    _write_hdf5_group(group, parameters.as_dict())
            return
        dumpfn(self.as_dict(), filename, indent=indent)


def _read_hdf5_parameter_group(filename: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py required for HDF5 parameters") from exc
    with h5py.File(filename, "r") as h5file:
        return _read_hdf5_group(h5file)


def _read_hdf5_group(group) -> dict[str, Any]:
    return {key: _decode_hdf5_value(value[()]) for key, value in group.items()}


def _write_hdf5_parameter_group(filename: Path, data: dict[str, Any]) -> None:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py required for HDF5 parameters") from exc
    with h5py.File(filename, "w") as h5file:
        _write_hdf5_group(h5file, data)


def _write_hdf5_group(group, data: dict[str, Any]) -> None:
    for key, value in data.items():
        if key.startswith("@") or value is None:
            continue
        group.create_dataset(key, data=value)


def _decode_hdf5_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
