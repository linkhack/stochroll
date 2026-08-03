# WP-012 Event-Masked Index Sampling Evidence

## Result

The isolated prototype implements uniform, with-replacement sampling of valid
indices along one structural `Event` axis. `ActiveBatch.sample_indices`
validates the complete request before accessing its shared parent generator,
draws only the non-deterministic varying-range ordinals required by the chosen
strategy, maps uniform prefix masks directly, and maps other masks through
compressed ascending true positions. It returns a signed `np.intp` `Roll`
whose selected axis is replaced by `size`.

The clear reference mapper receives the same validated request and prepared
ordinal array. Exhaustive deterministic cases over uneven masks show that
ordinal `0..n-1` maps exactly once to each of the `n` eligible positions;
there is no fixed-range remainder operation and therefore no modulo bias.

## Semantics and composition

- Sampling is independent for each repetition and fixed coordinate of every
  preserved structural axis. It is with replacement, so duplicate indices are
  retained.
- Axis values use absolute NumPy numbering. Axis 0, including its equivalent
  negative value, is rejected. The selected extent is replaced in place and
  result rank is unchanged.
- Every actual slice must contain a true entry. Empty slices fail before any
  draw. A zero-sized preserved axis has no actual slices, so it returns an
  empty result of the requested shape without consuming randomness.
- `size` and `axis` accept Python and NumPy integers through `operator.index`,
  but reject Booleans. `size` must be positive.
- The returned indices compose directly with `Roll.lookup`, `Event.lookup`,
  `Roll.route_sum`, and `Event.route_any`. `Pool.lookup` composes after the
  caller adds the explicit singleton dice dimension with `add_axis()`.
- Dragon targeting samples three attacks independently from players alive at
  the beginning of the phase. Masks with one through four living players are
  supported; a zero-player row fails if it bypasses the lifecycle guard.

Weighted sampling, sampling without replacement, joint multi-axis sampling,
optional/sentinel results for empty slices, structural-cell lifecycle packing,
and a production API remain outside WP-012.

## RNG evidence

For eligibility `(B, teams, players)` sampled along `players`, varying counts
use broadcast array-valued exclusive upper bounds. Uniform counts use a faster
scalar bound. Uniform singleton masks are deterministic and consume no random
values; singleton-heavy varying masks draw only for non-singleton slices when
that removes at least three quarters of the nominal draws. Interior-axis
requests use moved slice order and restore the selected axis in place.

Instrumented checks show that validation failures, zero-output calls, and
deterministic singleton calls make no generator calls. Uniform and varying
non-singleton checks record their strategy-specific request shapes and shared
parent stream. Eligibility and ActiveBatch positions remain unchanged.

Each fixed implementation remains seeded and self-reproducible for the same
environment, state, eligibility, strategy, and operation order. Historical
bit-exact equivalence across array/scalar bounds, singleton filtering, dense
target draws, rejection sampling, or a future production implementation is
not promised; the required compatibility is the same probability law.

## Optimization-target benchmarks

Environment: Python 3.12.13, NumPy 2.5.1, Linux x86-64 under WSL2. Eligibility
masks and ordinal arrays were prepared outside mapping timings. The
non-persisted ASV command used three samples and one invocation per sample:

```text
uv run asv run --config asv.wp009.conf.json -E existing --dry-run \
  --bench wp012_event_sampling -a repeat=3 -a number=1 -a warmup_time=0
```

The suite feature-detects the optimized ordinal-drawing and unchecked-mapping
helpers. When ASV installs the immediately preceding sampler revision, which
does not expose those helpers, the same benchmark names use that revision's
array-bound ordinal draw and validated vectorized mapper. This keeps the
expanded suite runnable across the optimization boundary; the internal-mapping
case means the mapping path available to ActiveBatch in each measured revision.

The complete module was exercised in quick, non-persisting mode for both
`01ca014` and `8fdfad7`; all 16 benchmark methods and their parameter matrices
completed for both revisions. The current working implementation completed the
same suite. The benchmark separates distinct optimization targets:

- `time_prepare` measures axis movement, eligible counting, empty-slice
  validation, uniform-prefix detection or compressed positions and offsets,
  and output-shape preparation.
- `time_ordinal_generation` measures the selected uniform, varying, or
  singleton-eliding draw strategy.
- `time_vectorized`, `time_internal_mapping`, and `time_reference` separate
  validated compressed mapping, the unchecked ActiveBatch hot path, and the
  direct per-slice oracle on the exact same prepared ordinal arrays.
- `time_end_to_end` measures validation, RNG, mapping, and `Roll` construction
  through `ActiveBatch.sample_indices`.
- `CandidateExtent` holds 8,192 slices constant while varying the candidate
  axis through 4, 16, 64, and 256 positions. It reports prefix preparation and
  mapping under the established names, plus explicit scattered-mask cases for
  the compressed fallback.
- `OrdinalDrawing` compares uniform-four, mixed, singleton-heavy, and fully
  deterministic singleton masks over 8,192 slices.
- `DragonHuntScenario` compares complete fixed-slot and event-masked targeting
  simulations at 2,000, 10,000, and 100,000 repetitions.

The final-axis stage workload used input `(2048, 32)`, axis `-1`, and output
sizes one and three. Across 100%, 75%, 25%, and one-candidate eligibility,
preparation measured about 0.068–0.079 ms and unchecked direct-prefix mapping
about 0.0027–0.0048 ms. Ordinal generation, validated mapping, and reference
mapping retain their separately measured cost centers.

The interior-axis workload used input `(256, 16, 32)`, axis `1`, and 8,192 or
24,576 output positions. Preparation measured about 0.111–0.122 ms and
unchecked direct-prefix mapping about 0.0028–0.0048 ms.

### Candidate-extent observation

The owner-provided comparison exposed 1.2–2.6x preparation regressions because
the first compressed implementation materialized positions even for uniform
prefix masks. Proving that representation directly from the uniform count and
prefix now skips both positions and offsets. The follow-up measurement used:

```text
.venv/bin/asv run --config asv.wp009.conf.json -E existing --dry-run \
  --bench 'wp012_event_sampling.(CandidateExtent.time_prepare|EventSampling.time_prepare)' \
  -a repeat=7 -a number=100 -a warmup_time=0
```

Established `CandidateExtent.time_prepare` cases measured about 0.134–0.805 ms
instead of 0.159–1.94 ms. Width-64 and width-256 partial cases returned to or
improved on the preceding implementation's range; width-4 cases matched it,
and the width-16/25% case retained only a roughly 25% proof-scan cost rather
than its previous 51–62% regression.

Unchecked direct-prefix mapping measured about 0.0025–0.0049 ms regardless of
candidate extent. The added scattered-mask cases continue to measure the
compressed fallback: its preparation intentionally pays to build positions,
then avoids the previous extent-256 cumulative mapper's 31–36 ms and large
candidates-by-output allocation during every mapping call.

The compressed representation uses memory proportional to eligible positions
plus output positions instead of candidates multiplied by output size. A
manual traced width-256, 25%-eligible, size-three comparison reduced peak
allocation from approximately 32.3 MiB to 4.7 MiB. Uniform prefix masks,
including fully eligible masks, store no compressed positions and map ordinals
directly.

For 8,192 three-sample slices, ordinal drawing measured about 0.19 ms for a
uniform count of four, 0.42 ms for mixed counts, 0.41 ms for the interleaved
singleton-heavy strategy, and 0.01 ms for deterministic singleton masks.

A bounded exploratory rejection mapper was slower than compressed ordinal
mapping for widths 4 and 32 across the tested densities. It became competitive
only for a width-256 mask near 75% eligibility, before including the validation
and fallback needed to make it robust. That narrow crossover did not justify
an unbounded, draw-variable production path in this prototype.

### Full Dragon Hunt observation

The complete WP-012 scenario retains fixed four-player state, ActiveBatch
whole-repetition packing, player attacks, three simultaneous dragon attacks,
route-back collision handling, lifecycle termination, shared RNG ownership,
and a 15-round bound. All dragon attacks sample with replacement from players
alive at the beginning of the dragon phase.

At 2,000, 10,000, and 100,000 repetitions, event-masked targeting measured
approximately 8.6 ms, 19.6 ms, and 200 ms. The completed WP-009-01 fixed-slot
baseline measured approximately 15.9 ms, 21.5 ms, and 181 ms in the same
short local run. These scenarios have intentionally different target and RNG
semantics, so this is an
integration-cost observation rather than a result-equivalence comparison.

For the deterministic 2,000-repetition input, fixed-slot targeting recorded
10,985 compact transitions and 241,670 draws; event-masked targeting recorded
10,411 transitions and 228,841 draws. Deterministic singleton choices account
for 201 fewer sampler draws than the earlier implementation. Changed combat
outcomes still alter the active trajectory, so the remaining draw-count
difference is not sampler overhead alone.

These local observations are descriptive rather than timing thresholds. The
suite now covers all required eligibility densities and axes, both output
sizes, four candidate extents, six sampler cost centers, and three complete
scenario sizes. Raw timing results were not persisted.

## Integration implications

The method is a narrow extension of the shared isolated ActiveBatch and does
not change production `stochroll`. It supplies the structural targeting
operation specified by WP-009-02. That dependent callback-runner package is
still blocked until WP-012 receives explicit owner approval, so runner/manual
equivalence belongs to WP-009-02 rather than being implemented prematurely in
this milestone. Its specification already requires both paths to consume this
same shared method.

ActiveBatch construction now injects the shared parent generator directly
instead of creating and immediately discarding a fresh default generator. A
local constructor microbenchmark decreased from roughly 12 µs to 0.5 µs per
nonempty batch without changing compact Pool ownership or draw behavior.

The prototype supports a plausible production primitive, but strict empty
slices and shared-stream draw-shape consequences are important ergonomics and
compatibility constraints. WP-009 retains the decision about public naming,
placement, extensions, performance strategy, and seeded guarantees.
