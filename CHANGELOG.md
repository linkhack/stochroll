## Unreleased

- Add an opt-in local ASV benchmark suite for reproducible hot-path and
  implementation-candidate comparisons.
- Add public `stack` and `concatenate` functions for homogeneous structural
  assembly of `Roll`, `Event`, and `Pool` values.
- Add `Roll.route_sum` and `Event.route_any` for duplicate-safe destination
  routing with explicit shape, axis, dtype, and validation semantics.
- Add fixed structural `select` and per-repetition `lookup` for `Roll`,
  `Event`, and structural Pool axes, plus singleton-axis insertion for `Roll`
  and `Event`.
- Add `Event.indicator()` for shape-preserving numeric 0/1 conversion.
- Type Roll and Event statistical summaries as `NDArray[np.float64]` while
  preserving NumPy's scalar runtime results for scalar-valued simulations.
