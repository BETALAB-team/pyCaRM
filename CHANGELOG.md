# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2026-10-13

### Fixed

- FLS `sqrt` mode now uses the receiver's equivalent radius, as documented.
  Results change only for fields with Voronoi cells of different sizes.
  Old FLS cache files are recomputed.

### Added

- Python 3.14 support.
- Documentation: references page, FLS method comparison and limitations.

## [0.2.1] - 2026-10-13

### Changed

- System matrices and their factorizations are cached per run and rebuilt only
  for a ground radius `rn`, mass flow and adiabatic flag not seen before (or
  after a material property change). Boreholes with the same `rn` share one
  factorization. Results are unchanged.
- The cache has no size limit: it holds one factorization per distinct
  `(rn, mass flow, adiabatic)` value. Flows that switch between a few fixed
  values keep it small; continuously varying flows make it grow with the
  number of distinct values.

## [0.2.0] - 2026-10-09

### Added

- Opt-in FLS response cache: `Simulation(..., fls_cache_dir=...)`. The FLS
  response increments are stored in that folder as `.npy` files keyed by a
  hash of the field, ground and time-step inputs they depend on, with a `.json`
  sidecar. Later simulations of the same field reuse them memory-mapped; a
  cached file from a longer run serves any shorter run by slicing.
- Opt-in disk-backed history: `Simulation.run(..., history_path=...)` writes
  `T_history` to a float32 `.npy` memmap as the run goes, instead of keeping
  it in RAM. With `save_results=True` the `.npz` stores `history_path` in place
  of `T_history`.
- Memory check: `Simulation.run()` raises `MemoryError` before the first step
  when an in-RAM `T_history` would not fit in the available memory, naming the
  sizes and suggesting `history_path=` (and `fls_cache_dir=` when the FLS
  increments are held in RAM). This also affects existing calls without the
  new arguments. Adds the `psutil` dependency.
- FLS image source: `Simulation(..., fls_image=True)` and
  `FiniteLineSolution(..., fls_image=True)` add the image part of the Finite
  Line Source. Tested against
  pygfunction `finite_line_source(..., imgSource=True)`. The FLS cache key
  includes `fls_image` and `D` (cache format version 2, so earlier cache files
  are not reused).

### Changed

- The FLS image source is on by default (`fls_image=True`).
- FLS load superposition is computed from response increments built once, as
  one matrix-vector product per step. `FiniteLineSolution` keeps only the increments;
  `response_matrix` is rebuilt from them on access.

## [0.1.0] - 2026-09-21

Initial public release.

### Added

- Single and multi-borehole field configurations
- Supported BHE types: single U-tube, double U-tube, coaxial, helical
- Ground stratification support
- Surface boundary conditions (solar radiation, sky radiation, convection)
- Voronoi-based field decomposition for multi-borehole layouts
- Finite Line Source (FLS) thermal interference model
- Parallel and series borehole connection modes
- Heat flux mode: drive the simulation from a heat pump-side thermal load and
  supply temperature, with heat pump COP/EER performance accounted for
- Time-variable grout thermophysical properties driven by soil moisture
  content (irrigation/precipitation input)

[0.2.1]: https://github.com/BETALAB-team/pyCaRM/releases/tag/v0.2.1
[0.2.0]: https://github.com/BETALAB-team/pyCaRM/releases/tag/v0.2.0
[0.1.0]: https://github.com/BETALAB-team/pyCaRM/releases/tag/v0.1.0
