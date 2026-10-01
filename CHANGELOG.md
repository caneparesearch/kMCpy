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
- `example/input_example.yaml` and `example/lce_only.yaml`. They used a
  pre-0.3 configuration schema and missing input paths and could not be
  loaded; use `kmcpy init` or `kmcpy sample` to generate current inputs.
- Internal pass-through helpers and unreachable `exclude_species` filtering in
  the structure code.

### Fixed

- Configuration YAML files written by `Configuration.to`/`kmcpy init`/`kmcpy
  sample` could not be read back with monty 2026.x, which decodes their
  `@module`/`@class` entries into objects; they are now always loaded as plain
  data.
- `pymatgen` is limited to `<2026`. pymatgen 2026.x picks different (equivalent)
  primitive lattice vectors, which reorders primitive-cell sites and silently
  invalidates event, initial-state, and model files generated with earlier
  versions.
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
