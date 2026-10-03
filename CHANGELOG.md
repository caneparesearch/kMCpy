# Changelog

## Unreleased

### Performance

- KMC steps are about 8x faster on the NASICON test system (605 to 72 us per
  step), with bit-identical seeded trajectories:
  - `LocalClusterExpansion.compute` gathers only local-environment
    occupations instead of copying the full occupation vector, and validates
    `keci` once per parameter assignment instead of on every call.
  - Correlation kernels use cached flat index arrays instead of nested
    `numba.typed.List` arguments.
  - `CompositeLCEModel` evaluates all dependent-event rates in one compiled
    call when both submodels are plain `LocalClusterExpansion` objects.
  - `EventLib.get_dependent_events` caches dependency rows as Python tuples.
  - Event sampling no longer passes the random generator into numba, the
    tracker caches fractional coordinates, and the per-pass summary table is
    only formatted when INFO logging is enabled.
  - Callable signature checks in `KMC` and `SiteEnergyModel` are cached.

### Added

- `kmcpy.Simulation`: a simulation is assembled from one shared
  `LatticeStructure` (exported as `kmcpy.LatticeStructure`) and one component
  per slot, so models, event sources, and initial states can be swapped
  independently and used in memory without intermediate files:
  - events: an event file, an `EventLib`, or a generator such as
    `kmcpy.HopEvents(cutoff=...)`;
  - model: a model file of any registered type or a model object;
  - state: an initial-state file, a `State`, occupations, or a builder such as
    `kmcpy.RandomOccupation({"Na": 0.75}, seed=...)`.
  `Configuration`/`KMC.from_config` and input files keep working; both paths
  build the simulation through the new `KMC.from_parts`.
- `Simulation` derives settings that previously had to be entered by hand,
  unless given explicitly: the mobile ion (from `site_mapping`), its charge
  (from the structure's oxidation states, guessed if the CIF has none;
  previously defaulted to 1 for every ion), and `elementary_hop_distance`
  (from the event hop lengths, root mean square with a warning if they
  differ; previously defaulted to 1 Angstrom). A run needs only
  `temperature` and `kmc_passes` (plus `random_seed` for reproducibility);
  the `kmcpy init` template's `run:` section lists only these.
- `LatticeStructure` is the disordered structure a simulation runs on: every
  site that can be occupied and the species it may hold.
  - Partial occupancies in the CIF define the allowed species (`Li: 0.5` is Li
    or vacancy `"X"`), so `site_mapping` is optional. When given, it lists only
    the varying species; unlisted species are fixed.
  - `LatticeStructure.from_cif(...)` loads a CIF; `make_supercell((2, 1, 1))`
    makes the supercell, in place or as a new object (`in_place=False`) as in
    pymatgen. The template stays the unit cell and `supercell_shape` is
    stored and serialized. It replaces the old `make_supercell(sc_matrix)`,
    which expanded the template in place without updating the active-site
    order; `get_structure_from_occ` is replaced by
    `structure_from_occupations`. The CIF path and `primitive` flag are loading options and
    are not stored.
  - `occupations_from_structure(structure)` and
    `structure_from_occupations(occupations)` convert between active-site
    occupations and ordered pymatgen structures, so `Simulation(state=...)`
    also accepts an ordered `Structure`.
  - Plain pymatgen structures get default `label`/`local_index`/
    `wyckoff_sequence` site properties, so they can generate events like
    CIF-loaded structures.
  - `as_dict`/`from_dict` round-trip it, and `to`/`from_file` write and read
    it.
- Sectioned input files (`lattice_structure`, `events`, `model`, `state`, `run`) for
  `Simulation.from_file` and `kmcpy run --input`. Components are file paths
  or `{type: ..., ...}` plugin specs (`hop` events, `random` state, any
  registered model type); relative paths resolve from the input file's
  folder. `validate_simulation_input` checks an input without reading files.
  `kmcpy init` and `kmcpy sample` write this format by default
  (`--format configuration` for the flat `Configuration` format, which
  `kmcpy run` still accepts). `kmcpy run --output_dir` sets the result folder.
- `register_event_source` / `register_state_builder` make custom event sources
  and initial-state builders available by `type` name in input files.
- In-memory LCE fitting: `LocalClusterExpansion.fit_data(correlation_matrix,
  targets, alpha=...)` and `NEBDataLoader.fit(alpha=...)` fit and attach the
  parameters without writing fitting files (`LCEFitter.fit_arrays` is the
  shared core of the file-based `fit`).
- `LocalLatticeStructure.from_lattice_structure(...)` accepts an event as
  `center`, and `LocalEnvironmentEnumerator(lattice)` uses the supercell's
  active-site indices for a supercell lattice, so LCE and NEB steps reuse the
  simulation's structure and `site_mapping`.
- `kmcpy.register_model("name")` registers custom model classes so model files
  and `BaseModel.load(path)` can refer to them by type.
- `output_dir=` for `KMC.run`, `Simulation.run`, and `Tracker.write_results`
  (result files previously always went to the working directory).
- `EventGenerator.generate_events` accepts a loaded `structure=` and
  `event_file=None`; the generated library is available as `generator.event_lib`.
- Active-site metadata (in event files and site-energy models) records the
  supercell lattice and active-site positions. Loading a file whose site
  indices refer to a different cell or lattice basis now raises an error
  instead of running with mismatched sites; older files without positions are
  accepted with a warning.
- `BaseModel.compute_probabilities(...)` batched rate hook. The default calls
  `compute_probability(...)` per event; KMC uses it for dependent-event
  updates.

### Changed

- `LocalBarrierModel.rules` holds `BarrierRule` objects instead of plain
  dictionaries. `BarrierRule.from_dict`/`as_dict` validate and serialize rules;
  model files and the `rules=[...]`/`add_*_rule` inputs are unchanged, and
  `add_rule` also accepts a `BarrierRule`.
- `Configuration` forwards system/runtime fields (`config.temperature`, ...)
  through one `__getattr__` instead of 23 hand-written properties, and the
  routing sets `SYSTEM_FIELD_NAMES`/`RUNTIME_FIELD_NAMES` are derived from the
  `SystemConfig`/`RuntimeConfig` dataclass fields (now `frozenset`s). A new
  config field only needs to be added to its dataclass.
- `SiteEnergyModel`'s mapping onto an external code's sites and occupation
  values is the new `kmcpy.models.ExternalSiteMapping`, held as
  `model.external_mapping`. Constructor arguments and model files are
  unchanged; the mapping fields (`site_mapping`, `state_mapping`,
  `state_mapping_by_site`, `initial_occupation`, `external_size`,
  `external_fill_value`, `external_dtype`) are now read from
  `model.external_mapping` instead of the model itself.
- `LocalClusterExpansion` declares all of its fields in `__init__` (unset
  fields are `None`; `has_parameters()` reports whether `keci` and
  `empty_cluster` are set). `from_dict` decodes the known payload keys
  explicitly instead of setting every key as an attribute: the old
  `MigrationUnit_structure` key is renamed, the unused `clusters` and
  `template_structure` keys are dropped, and other unknown keys are ignored
  with a warning. `str()`/`repr()` no longer fail on a model without orbits.
- `kmcpy.io.cif` uses pymatgen's public `CifParser` instead of a modified copy
  of pymatgen's private parser code (removed with `cif_LICENSE.rst`). Site
  properties `label`, `wyckoff_sequence`, and `local_index` are unchanged.
- Model file I/O lives in `BaseModel`: one `to(fname, indent=2)`, one
  `from_file`, and one envelope unwrapper driven by each class's `MODEL_TYPE`
  and `PAYLOAD_KEY`. `LocalClusterExpansion.to` now writes with indent 2
  (was 4), and `SiteEnergyModel.from_file` now rejects files with an unknown
  `filetype` instead of reading them as a bare payload.
- The model-type registry moved from `kmcpy.io.registry` to
  `kmcpy.models.registry`, with `model_class_for_type` and
  `model_class_for_payload`. `BaseModel.from_config` and `CompositeLCEModel`
  site-model loading both use it, so composite files also fall back to the
  registered class when a site model's module path has moved.
- Local-environment enumeration is implemented by
  `kmcpy.structure.LocalEnvironmentEnumerator`, which builds the active lattice
  once per lattice structure; `enumerate_local_environments`,
  `generate_neb_endpoint_pair`, and `enumerate_neb_endpoint_pairs` are
  unchanged wrappers around it. `LocalLatticeStructure` and the enumerator
  share one local-site selection (`resolve_center_site` and
  `LocalSiteOrder.order_local_env_sites`).
- Numba kernels for LCE correlations and batched composite rates live in
  `kmcpy.models.lce_kernels`; `LocalClusterExpansion.kernel_inputs()` exposes
  a model's arrays to them.
- Callable helpers (`module:function` references and keyword-support checks)
  live in `kmcpy.callables` instead of being duplicated in `KMC`,
  `CompositeLCEModel`, and `SiteEnergyModel`.
- `site_mapping` parsing lives in one place, the new
  `kmcpy.structure.SiteMapping`, used by `LatticeStructure`, `ActiveSiteOrder`,
  and `EventGenerator`.

### Removed

- The `exclude_species` parameters of `LocalLatticeStructure`,
  `LocalClusterExpansion.get_occ_corr_from_structure`/`get_corr_from_structure`,
  `NEBEntry`/`NEBDataLoader`, and the local-environment enumeration functions.
  They only raised "no longer supported" since 0.3.0; encode fixed sites in
  `site_mapping` with a single allowed species.
- `kmcpy.tools` (`gather_kmc_data`, `gather_mc_data`, `get_data`). These
  NASICON/CASM analysis scripts moved to `scripts/`, and `glob2` and `joblib`
  are no longer direct runtime dependencies.
- `kmcpy.structure.SupercellComparator`. It was unused and matched every pair
  of species; use pymatgen's `FrameworkComparator` for species-agnostic
  structure matching.
- `Orbit.get_cluster_function` and `Cluster.get_cluster_function`. They
  assumed the old binary occupation encoding and did not match the correlation
  functions used by `LocalClusterExpansion`.
- Stale example artifacts: `example/files/input/kmc_input.json` (pre-0.3
  input format) and old generated files under `example/output/`, which is now
  ignored.
- `example/input_example.yaml` and `example/lce_only.yaml`. They used a
  pre-0.3 configuration schema and missing input paths and could not be
  loaded; use `kmcpy init` or `kmcpy sample` to generate current inputs.
- Internal pass-through helpers and unreachable `exclude_species` filtering in
  the structure code.

### Fixed

- Event generation silently produced wrong events when the supercell was
  shorter than the local-environment cutoff along some axis: periodic images
  of one site folded onto the same supercell site, giving self-hops such as
  `(0, 0)`, duplicate events, and local environments that counted a site more
  than once. It now raises an error asking for a larger `supercell_shape` (or
  a smaller cutoff).
- `kmcpy run --input` reported every loading error as "Legacy InputSet format
  is no longer supported"; the original error is now shown.
- `example/NASICON.ipynb` used the pre-0.3 API (`cluster_expansion_file`,
  `immutable_sites`, ...) and could not run. It is rewritten around
  `LatticeStructure`/`Simulation` (events, LCE building and in-memory fitting, model
  assembly, swapping models, input files) and its code is run by the test
  suite.
- `example/minimal_example.py` (the Quickstart's first command) and
  `example/tutorial_nasicon.py` failed with "Unknown configuration fields:
  ['immutable_sites']". Both are rewritten with `LatticeStructure`/`Simulation` and
  are now run by the test suite.
- Configuration YAML files written by `Configuration.to`/`kmcpy init`/`kmcpy
  sample` could not be read back with monty 2026.x, which decodes their
  `@module`/`@class` entries into objects; they are now always loaded as plain
  data.
- kMCpy works with pymatgen 2026.x. pymatgen 2026 picks different (equivalent)
  primitive lattice vectors, which changed supercell site indices and made
  existing event, initial-state, and model files silently describe other sites
  (e.g. 2x the NASICON conductivity). Primitive cells loaded from CIF are now
  expressed in a kMCpy-defined lattice basis
  (`kmcpy.structure.lattice_basis.standardize_lattice_basis`), which reproduces
  the files generated with earlier versions under both pymatgen 2025 and 2026.
- `SiteEnergyModel` with a string `initial_occupation` (fixed-width NumPy
  string dtype) silently truncated longer mapped values, e.g. `"Va"` became
  `"V"`. The external occupation dtype is now widened to fit every mapped
  state value.
- The GUI's (`start_kmcpy_gui`) "LocalClusterExpansion" command always failed
  with `UnboundLocalError`: a later function-local import made
  `LocalClusterExpansion` a local name for the whole function.
- Vacancy labels are recognized consistently everywhere. `EventGenerator`
  previously accepted only `X`/`Vacancy` when inferring the mobile species, so
  a `site_mapping` using `Va` failed with "Could not infer mobile species".
  Vacancy labels (`X`, `Va`, `Vacancy`) are now case-insensitive.
- `str(orbit)` returned `None` (and raised `TypeError`) instead of the cluster
  summary.

## 0.3.0 - 2026-05-27

This release is a breaking cleanup release focused on making kMCpy easier to
understand, document, and extend for research workflows.

### Breaking API changes

- `KMC.run()` now uses the configuration already attached to the `KMC` object.
  Use `kmc.run()` instead of `kmc.run(config)`.
- Mutable occupations are owned by `State`; `KMC` no longer keeps a separate
  mutable `occ_global` copy.
- Active-site and local-environment order APIs were renamed for clearer domain
  terminology: use `ActiveSiteOrder` and `LocalSiteOrder`.
- Event generation now uses `site_mapping` as the canonical active-site
  convention and no longer exposes `mobile_ion_identifier_type`.
- Hop-direction helpers live in `kmcpy.event.hop`.
- Site-energy models use `compute(...)` consistently for site-energy
  differences.

### Added

- `LocalBarrierModel` for constant barriers, condition-based barriers, wildcard
  local-environment rules, and exact local-environment matching.
- Multicomponent Chebyshev basis support for sites with more than two species.
- Array-backed active-site mapping for external site-energy adapters.
- Explicit unit conventions in `kmcpy.units`, configuration metadata, and
  tracker result metadata.
- Documentation for local barrier models, site-order mapping, external
  site-energy models, and property attachment.
- CI install checks for built wheels, `uv pip`, and pip installs inside Conda
  environments.

### Changed

- Configuration serialization omits loader-only paths by default, while input
  templates still include them for simulation setup.
- Tracker output writing is separated from the core simulation loop.
- Built-in transport metrics are explicit tracker behavior rather than a hidden
  attached callback.
- Model serialization follows Monty-style `as_dict`/`from_dict` patterns more
  consistently.

### Release checks

- Full test suite: `256 passed`.
- Documentation build succeeds.
- Wheel and source distribution pass `twine check`.
- Built wheel installs successfully with `pip` and `uv pip` in fresh Python
  3.13 environments.
