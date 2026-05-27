# Changelog

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

### Changed

- Configuration serialization omits loader-only paths by default, while input
  templates still include them for simulation setup.
- Tracker output writing is separated from the core simulation loop.
- Built-in transport metrics are explicit tracker behavior rather than a hidden
  attached callback.
- Model serialization follows Monty-style `as_dict`/`from_dict` patterns more
  consistently.

### Release checks

- Full test suite: `255 passed`.
- Documentation build succeeds.
- Wheel and source distribution pass `twine check`.
