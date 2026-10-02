"""Assemble a kMC simulation from interchangeable components.

A simulation runs on one :class:`~kmcpy.structure.LatticeStructure` (every
site of the simulated supercell and the species it may hold) and has one
component per slot; any component that fits the slot can be used:

========  =========================================================  ==========================
Slot      Accepts                                                    Built-in options
========  =========================================================  ==========================
events    ``EventLib``, an event-file path, or an object with        :class:`HopEvents`
          ``generate(lattice_structure) -> EventLib``
model     a rate model object (``compute_probability``) or a         ``LocalBarrierModel``,
          model-file path (any registered model type)                ``CompositeLCEModel``, ...
state     an ordered pymatgen ``Structure`` (one configuration),     :class:`RandomOccupation`
          ``State``, an initial-state file path, a list of active-
          site occupations, or an object with
          ``build(lattice_structure) -> State``
========  =========================================================  ==========================

Example::

    lattice = kmcpy.LatticeStructure.from_cif(
        "nasicon.cif", site_mapping={"Na": ["Na", "X"], "Si": ["Si", "P"]}
    )
    lattice.make_supercell((2, 1, 1))
    sim = kmcpy.Simulation(
        lattice,
        events=kmcpy.HopEvents(cutoff=4.0),
        model=kmcpy.LocalBarrierModel.constant_barrier(300.0),
        state=kmcpy.RandomOccupation({"Na": 0.75}, seed=1),
        temperature=298.0,
        kmc_passes=1000,
        random_seed=1,
    )
    tracker = sim.run(output_dir="results")

The same simulation as a YAML input file (``Simulation.from_file``,
``kmcpy run --input``); relative paths are resolved from the file's folder::

    lattice_structure:
      structure: nasicon.cif
      site_mapping: {Na: [Na, X], Si: [Si, P]}   # not needed if the CIF has partial occupancies
      supercell_shape: [2, 1, 1]
    events: {type: hop, cutoff: 4.0}          # or: events.json
    model: {type: local_barrier, default_barrier: 300.0}   # or: model.json
    state: {type: random, fractions: {Na: 0.75}, seed: 1}  # or: initial_state.json
    run: {temperature: 298.0, kmc_passes: 1000, random_seed: 1, output_dir: results}

Custom event sources and state builders become available by ``type`` name with
:func:`register_event_source` and :func:`register_state_builder` (models:
:func:`kmcpy.register_model`).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from pymatgen.core import Structure

from kmcpy.event import EventGenerator, EventLib
from kmcpy.io.files import load_raw_data
from kmcpy.models.base import BaseModel
from kmcpy.models.registry import model_class_for_type
from kmcpy.simulator.config import CONFIG_FIELD_NAMES, Configuration
from kmcpy.simulator.kmc import KMC
from kmcpy.simulator.state import State
from kmcpy.simulator.tracker import Tracker
from kmcpy.structure.lattice_structure import LatticeStructure
from kmcpy.structure.species import normalize_species, species_label

# Configuration fields that Simulation takes from the lattice structure or the slots.
_FIELDS_FROM_PARTS = {
    "structure_file": "the LatticeStructure",
    "site_mapping": "the LatticeStructure",
    "supercell_shape": "the LatticeStructure",
    "convert_to_primitive_cell": "the LatticeStructure",
    "event_file": "events=",
    "model_file": "model=",
    "model_type": "model=",
    "initial_state_file": "state=",
    "initial_occupations": "state=",
}

# ``type`` names usable in input files for the events and state slots.
EVENT_SOURCES: dict[str, type] = {}
STATE_BUILDERS: dict[str, type] = {}


def register_event_source(name: str, *, replace: bool = False):
    """Class decorator making an event source available as ``events: {type: name}``.

    The class needs ``generate(lattice_structure) -> EventLib`` and a constructor (or
    ``from_dict``) that accepts the remaining input-file keys. Registering a
    different class under an existing name raises ``ValueError`` unless
    ``replace=True``.
    """
    return _registrar(EVENT_SOURCES, "Event source", name, replace)


def register_state_builder(name: str, *, replace: bool = False):
    """Class decorator making a state builder available as ``state: {type: name}``.

    The class needs ``build(lattice_structure) -> State`` and a constructor (or
    ``from_dict``) that accepts the remaining input-file keys. Registering a
    different class under an existing name raises ``ValueError`` unless
    ``replace=True``.
    """
    return _registrar(STATE_BUILDERS, "State builder", name, replace)


def _registrar(registry: dict[str, type], kind: str, name: str, replace: bool):
    def decorator(component: type) -> type:
        existing = registry.get(name)
        if existing is not None and existing is not component and not replace:
            raise ValueError(
                f"{kind} type '{name}' is already registered to {existing!r}; "
                "pass replace=True to override it."
            )
        registry[name] = component
        return component

    return decorator


@register_event_source("hop")
class HopEvents:
    """Event slot: hops of the mobile species to neighboring sites.

    Parameters:
        cutoff: Local-environment radius in Angstrom, used for every pair of
            species around the mobile ion.
        cutoffs: Per-species-pair radii instead of ``cutoff``, e.g.
            ``{("Na+", "Na+"): 4.0, ("Na+", "Si4+"): 4.0}`` (oxidation states
            are guessed from the structure).
        labels: ``(initial, target)`` CIF labels to restrict the hop
            endpoints, e.g. ``("Na1", "Na2")``. By default every site of the
            mobile species can hop.
        mobile_species: Mobile species; inferred from the site mapping by default.
        rtol, atol: Tolerances for matching local environments.
    """

    def __init__(
        self,
        cutoff: float = 4.0,
        *,
        cutoffs: Mapping[tuple[str, str], float] | None = None,
        labels: tuple[str, str] | None = None,
        mobile_species: Sequence[str] | None = None,
        rtol: float = 0.01,
        atol: float = 0.01,
    ) -> None:
        self.cutoff = cutoff
        self.cutoffs = dict(cutoffs) if cutoffs is not None else None
        self.labels = tuple(labels) if labels is not None else None
        self.mobile_species = list(mobile_species) if mobile_species is not None else None
        self.rtol = rtol
        self.atol = atol

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HopEvents":
        """Build from input-file keys; ``cutoffs`` is a list of ``[a, b, radius]``."""
        data = dict(data)
        cutoffs = data.pop("cutoffs", None)
        if cutoffs is not None and not isinstance(cutoffs, Mapping):
            cutoffs = {(str(a), str(b)): float(radius) for a, b, radius in cutoffs}
        return cls(cutoffs=cutoffs, **data)

    def generate(self, lattice_structure: LatticeStructure) -> EventLib:
        generator = EventGenerator()
        generator.generate_events(
            structure=lattice_structure.template_structure,
            site_mapping=lattice_structure.site_mapping,
            supercell_shape=list(lattice_structure.supercell_shape),
            local_env_cutoff=None if self.cutoffs else self.cutoff,
            local_env_cutoff_dict=self.cutoffs,
            mobile_ion_identifiers=self.labels,
            mobile_species=self.mobile_species,
            distance_matrix_rtol=self.rtol,
            distance_matrix_atol=self.atol,
            find_nearest_if_fail=False,
            export_local_env_structure=False,
            event_file=None,
        )
        return generator.event_lib


@register_state_builder("random")
class RandomOccupation:
    """State slot: random occupation with given species fractions.

    Each entry ``species: fraction`` places ``species`` on that fraction
    (rounded) of the active sites that allow it, chosen at random; the other
    sites of that kind get their first allowed species other than ``species``.
    Sites not covered by any entry keep their first allowed species. Use
    ``"X"`` (or ``"Va"``) for vacancies.

    Example: ``RandomOccupation({"Na": 0.75, "P": 0.5}, seed=1)`` fills 75% of
    the Na/vacancy sites with Na and half of the Si/P sites with P.
    """

    def __init__(self, fractions: Mapping[str, float], seed: int | None = None) -> None:
        for species, fraction in fractions.items():
            if not 0.0 <= float(fraction) <= 1.0:
                raise ValueError(f"Fraction for {species!r} must be between 0 and 1")
        self.fractions = dict(fractions)
        self.seed = seed

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RandomOccupation":
        return cls(**data)

    def build(self, lattice_structure: LatticeStructure) -> State:
        allowed = lattice_structure.active_site_order.allowed_species_by_active_site
        occupations = [0] * len(allowed)
        rng = np.random.default_rng(self.seed)
        for species, fraction in self.fractions.items():
            token = species_label(normalize_species(species))
            sites = [index for index, states in enumerate(allowed) if token in states]
            if not sites:
                raise ValueError(f"No active site allows species {species!r}")
            chosen = set(
                rng.choice(sites, size=round(float(fraction) * len(sites)), replace=False).tolist()
            )
            for index in sites:
                states = allowed[index]
                if index in chosen:
                    occupations[index] = states.index(token)
                else:
                    occupations[index] = next(
                        state for state, label in enumerate(states) if label != token
                    )
        return State(occupations=occupations)


class Simulation:
    """A kMC run on a lattice structure, assembled from one component per slot.

    Parameters:
        lattice_structure: The simulated supercell (``LatticeStructure``,
            see ``LatticeStructure.make_supercell``).
        events: Event slot (see the module docstring).
        model: Rate-model slot.
        state: Initial-state slot.
        output_dir: Default result folder for :meth:`run`.
        **settings: Run settings. Usually only ``temperature`` (K) and
            ``kmc_passes`` are needed, plus ``random_seed`` for reproducible
            runs. Optional: ``attempt_frequency`` (Hz, default 1e13),
            ``equilibration_passes`` (default 1000), ``name`` (label in result
            file names), ``dimension`` (of diffusion, default 3), and the
            property-sampling settings (see ``Configuration.help_fields()``).

    Derived from the parts unless given explicitly:

    - ``mobile_ion_specie``: the species with a vacancy state in ``site_mapping``;
    - ``mobile_ion_charge``: its oxidation state in the structure (guessed if
      the CIF has none);
    - ``elementary_hop_distance``: the hop length of the events (Angstrom).
    """

    def __init__(
        self,
        lattice_structure: LatticeStructure,
        *,
        events: Any,
        model: Any,
        state: Any,
        output_dir: str | Path | None = None,
        **settings: Any,
    ) -> None:
        owned = sorted(set(settings) & set(_FIELDS_FROM_PARTS))
        if owned:
            raise ValueError(
                "These settings come from the simulation parts: "
                + ", ".join(f"{name} (set via {_FIELDS_FROM_PARTS[name]})" for name in owned)
            )
        if "mobile_ion_specie" not in settings:
            mobile_species = lattice_structure.mobile_species
            if len(mobile_species) != 1:
                raise ValueError(
                    f"Cannot infer one mobile species from the site mapping "
                    f"(found {mobile_species}); pass mobile_ion_specie=..."
                )
            settings["mobile_ion_specie"] = mobile_species[0]
        if "mobile_ion_charge" not in settings:
            settings["mobile_ion_charge"] = _mobile_ion_charge(
                lattice_structure, settings["mobile_ion_specie"]
            )
        self._derive_hop_distance = "elementary_hop_distance" not in settings

        self.lattice_structure = lattice_structure
        self.events = events
        self.model = model
        self.state = state
        self.output_dir = output_dir
        self.config = Configuration(
            site_mapping=lattice_structure.site_mapping,
            supercell_shape=lattice_structure.supercell_shape,
            **settings,
        )
        self._event_lib: EventLib | None = None
        self._attachments: list[tuple[Callable, dict[str, Any]]] = []

    def attach(self, func: Callable, **kwargs: Any) -> "Simulation":
        """Attach a property callback ``func(state, step, sim_time)``; see ``KMC.attach``."""
        self._attachments.append((func, kwargs))
        return self

    def event_lib(self) -> EventLib:
        """Return the event library, generating or loading it once."""
        if self._event_lib is None:
            self._event_lib = _resolve_events(self.events, self.lattice_structure)
        return self._event_lib

    def build(self) -> KMC:
        """Assemble a ready-to-run ``KMC`` with a fresh copy of the initial state."""
        if self._derive_hop_distance:
            self.config = self.config.with_system_changes(
                elementary_hop_distance=_hop_distance(self.lattice_structure, self.event_lib())
            )
            self._derive_hop_distance = False
        kmc = KMC.from_parts(
            self.lattice_structure,
            _resolve_model(self.model),
            self.event_lib(),
            _resolve_state(self.state, self.lattice_structure),
            self.config,
        )
        for func, kwargs in self._attachments:
            kmc.attach(func, **kwargs)
        return kmc

    def run(self, *, output_dir: str | Path | None = None, label: str | None = None) -> Tracker:
        """Run the simulation; result files go to ``output_dir``.

        Defaults to the ``output_dir`` given at construction, then the working
        directory.
        """
        return self.build().run(
            label=label,
            output_dir=output_dir if output_dir is not None else self.output_dir,
        )

    @classmethod
    def from_file(cls, filename: str | Path) -> "Simulation":
        """Load a simulation input file (YAML/JSON) with ``lattice_structure``/``events``/... sections."""
        path = Path(filename)
        data = load_raw_data(path)
        return cls.from_dict(data, base_dir=path.parent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], base_dir: str | Path = ".") -> "Simulation":
        """Build from input-file sections; relative paths are resolved from ``base_dir``."""
        validate_simulation_input(data)
        base = Path(base_dir)
        lattice_data = data["lattice_structure"]
        lattice_structure = LatticeStructure.from_cif(
            _resolve_path(lattice_data["structure"], base),
            site_mapping=lattice_data.get("site_mapping"),
            primitive=bool(lattice_data.get("primitive", False)),
            supercell_shape=lattice_data.get("supercell_shape", (1, 1, 1)),
        )
        run = dict(data.get("run") or {})
        output_dir = run.pop("output_dir", None)
        return cls(
            lattice_structure,
            events=_component_from_spec(data["events"], EVENT_SOURCES, base),
            model=_model_from_spec(data["model"], base),
            state=_state_from_spec(data["state"], base),
            output_dir=_resolve_path(output_dir, base) if output_dir is not None else None,
            **run,
        )


SIMULATION_SECTIONS = ("lattice_structure", "events", "model", "state", "run")
_LATTICE_STRUCTURE_KEYS = {"structure", "site_mapping", "supercell_shape", "primitive"}


def is_simulation_input(data: Any) -> bool:
    """Return whether loaded input-file data uses the sectioned simulation format."""
    return isinstance(data, Mapping) and "lattice_structure" in data


def validate_simulation_input(data: Mapping[str, Any]) -> None:
    """Check sections, ``type`` names, and run-setting names without reading any files."""
    if not isinstance(data, Mapping):
        raise ValueError("Simulation input must be a mapping of sections")
    unknown = sorted(set(data) - set(SIMULATION_SECTIONS))
    if unknown:
        raise ValueError(f"Unknown simulation input sections: {unknown}. Expected {list(SIMULATION_SECTIONS)}")
    missing = [section for section in ("lattice_structure", "events", "model", "state") if section not in data]
    if missing:
        raise ValueError(f"Simulation input is missing sections: {missing}")

    lattice = data["lattice_structure"]
    if not isinstance(lattice, Mapping) or "structure" not in lattice:
        raise ValueError("lattice_structure needs 'structure' (a CIF path)")
    unknown = sorted(set(lattice) - _LATTICE_STRUCTURE_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown lattice_structure keys: {unknown}. Expected {sorted(_LATTICE_STRUCTURE_KEYS)}"
        )

    for section, registry in (("events", EVENT_SOURCES), ("state", STATE_BUILDERS)):
        spec = data[section]
        if isinstance(spec, Mapping) and "type" in spec and spec["type"] not in registry:
            raise ValueError(f"Unknown {section} type {spec['type']!r}. Available: {sorted(registry)}")
    model = data["model"]
    if isinstance(model, Mapping) and "type" in model:
        model_class_for_type(model["type"])

    run = data.get("run") or {}
    allowed = (CONFIG_FIELD_NAMES - set(_FIELDS_FROM_PARTS)) | {"output_dir"}
    unknown = sorted(set(run) - allowed)
    if unknown:
        raise ValueError(f"Unknown run settings: {unknown}. See Configuration.help_fields().")


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _component_from_spec(spec: Any, registry: Mapping[str, type], base: Path) -> Any:
    """A path (string or ``{file: ...}``) or a ``{type: ..., **options}`` plugin spec."""
    if isinstance(spec, (str, Path)):
        return _resolve_path(spec, base)
    if isinstance(spec, Mapping) and set(spec) == {"file"}:
        return _resolve_path(spec["file"], base)
    if isinstance(spec, Mapping) and "type" in spec:
        options = {key: value for key, value in spec.items() if key != "type"}
        component = registry[spec["type"]]
        from_dict = getattr(component, "from_dict", None)
        return from_dict(options) if callable(from_dict) else component(**options)
    raise ValueError(f"Expected a file path, {{file: ...}}, or {{type: ...}}; got {spec!r}")


def _model_from_spec(spec: Any, base: Path) -> Any:
    if isinstance(spec, Mapping) and "type" in spec and "file" not in spec:
        options = {key: value for key, value in spec.items() if key != "type"}
        return model_class_for_type(spec["type"]).from_dict(options)
    if isinstance(spec, Mapping) and "file" in spec:
        return BaseModel.load(_resolve_path(spec["file"], base), model_type=spec.get("type"))
    return _component_from_spec(spec, {}, base)


def _state_from_spec(spec: Any, base: Path) -> Any:
    if isinstance(spec, Mapping) and set(spec) == {"occupations"}:
        return list(spec["occupations"])
    return _component_from_spec(spec, STATE_BUILDERS, base)


def _mobile_ion_charge(lattice_structure: LatticeStructure, species: str) -> float:
    """Oxidation state of ``species`` in the structure, guessed if the CIF has none."""
    structure = lattice_structure.template_structure.copy()
    charges = {getattr(site.specie, "oxi_state", None) for site in structure if site.specie.symbol == species}
    if charges in ({None}, {0}, set()):
        structure.remove_oxidation_states()
        try:
            structure.add_oxidation_state_by_guess()
        except ValueError:
            pass
        charges = {getattr(site.specie, "oxi_state", None) for site in structure if site.specie.symbol == species}
    charges.discard(None)
    if len(charges) != 1 or not next(iter(charges)):
        warnings.warn(
            f"Could not determine the charge of {species!r}; using 1. "
            "Pass mobile_ion_charge=... for the conductivity.",
            UserWarning,
            stacklevel=3,
        )
        return 1.0
    return float(abs(next(iter(charges))))


def _hop_distance(lattice_structure: LatticeStructure, event_lib: EventLib) -> float:
    """Hop length of the events in Angstrom (root mean square if they differ)."""
    structure = lattice_structure.active_structure()
    lengths = np.array(
        [structure.get_distance(*map(int, event.mobile_ion_indices)) for event in event_lib.events]
    )
    if np.ptp(lengths) > 1e-3:
        rms = float(np.sqrt(np.mean(lengths**2)))
        warnings.warn(
            f"Events have hop lengths from {lengths.min():.3f} to {lengths.max():.3f} "
            f"Angstrom; using their root-mean-square {rms:.3f} Angstrom as "
            "elementary_hop_distance for the correlation factor. Pass "
            "elementary_hop_distance=... to choose another value.",
            UserWarning,
            stacklevel=3,
        )
        return rms
    return float(np.mean(lengths))


def _resolve_events(events: Any, lattice_structure: LatticeStructure) -> EventLib:
    if isinstance(events, EventLib):
        return events
    if isinstance(events, (str, Path)):
        return EventLib.from_file(str(events))
    if callable(getattr(events, "generate", None)):
        return events.generate(lattice_structure)
    raise TypeError(
        "events must be an EventLib, an event-file path, or an object with generate(lattice_structure)"
    )


def _resolve_model(model: Any):
    if isinstance(model, (str, Path)):
        return BaseModel.load(str(model))
    if callable(getattr(model, "compute_probability", None)):
        return model
    raise TypeError(
        "model must be a model-file path or an object with compute_probability(...)"
    )


def _resolve_state(state: Any, lattice_structure: LatticeStructure) -> State:
    order = lattice_structure.active_site_order
    if isinstance(state, Structure):
        return State(occupations=lattice_structure.occupations_from_structure(state))
    if isinstance(state, State):
        return state.copy()
    if isinstance(state, (str, Path)):
        return State.from_file(str(state), supercell_shape=lattice_structure.supercell_shape, active_site_order=order)
    if callable(getattr(state, "build", None)):
        return state.build(lattice_structure)
    if isinstance(state, Sequence):
        return State.from_occupations(list(state), active_site_order=order)
    raise TypeError(
        "state must be an ordered Structure, a State, an initial-state file path, "
        "a list of occupations, or an object with build(lattice_structure)"
    )
