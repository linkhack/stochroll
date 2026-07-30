# Local benchmarks

This directory contains the opt-in ASV suite for performance exploration and
regression checks. It is outside pytest's configured `tests/` path, is never
run by CI, and is not imported by `stochroll`.

The suite has two complementary layers:

- `public_api.py` asks, “Did StochRoll get slower for users?” It measures
  realistic end-to-end behavior for every current public method and function.
- `reductions.py` asks, “Is our custom optimization still better than NumPy?”
  It compares only code paths for which StochRoll has a custom optimized
  implementation.

Correctness, edge cases, and exhaustive dtype combinations belong in unit
tests. Do not turn the reference suite into a dtype × workload × operation
matrix. Exhaustive performance investigations should use a separately
filtered extended module.

## Install and validate

Install the dedicated dependency group, initialize ASV once per machine, then
validate discovery and deterministic setup:

```bash
uv sync --group bench
uv run --group bench asv machine --yes
uv run --group bench asv check
uv run --group bench python -m benchmarks.check_suite
```

ASV stores the machine description in its standard per-user
`~/.asv-machine.json` file and copies it into local result metadata. The setup
check verifies workload matrices, shapes, configuration metadata, and
reproducibility of seeded prepared inputs. It does not run candidates or
compare their outputs.

## Public API layer

Every public scenario uses two standard workloads:

- `small`: 1,000 repetitions, representing a common simulation.
- `large`: 250,000 repetitions, exposing throughput and memory behavior.

Roll and Event inputs have shape `(R, 6)`. Pool inputs have shape
`(R, 6, 12)`. Pool lookup indices are full rank with shape `(R, 3, 12)`.

Run the public layer or select a method and workload:

```bash
uv run --group bench asv run --quick --bench 'public_api\.'
uv run --group bench asv run --bench 'public_api\.PoolMethods\.time_sum'
uv run --group bench asv run --bench 'public_api\.RollMethods.*large'
```

Keep/drop benchmarks parameterize direct, identity, and delegated branches.
Reroll benchmarks parameterize no-match, sparse-match, and dense-match inputs.

## Focused implementation layer

Focused reduction comparisons use 100,000 repetitions and normal Pool
`uint8` inputs. Each NumPy/StochRoll pair receives the same input array, final
axis, and output or accumulator dtype.

The reference sizes cover singleton handling, an interior custom path,
cutoff/cutoff-plus-one behavior, the normal twelve-die Pool, and a wide
fallback:

```bash
uv run --group bench asv run --quick --bench 'reductions\.'
uv run --group bench asv run --bench 'reductions\.SumLastAxis'
uv run --group bench asv run --bench 'reductions\.DropLowestSum'
```

The layer compares:

- `_reduce_sum_last_axis` with `numpy.sum`;
- `_reduce_min_last_axis` with `numpy.min`;
- `_reduce_max_last_axis` with `numpy.max`;
- `Pool.drop_lowest_sum` with its straightforward NumPy composition.

Routing candidates are measured separately in `routing.py`. The indexed
candidate uses duplicate-safe indexed accumulation, while the mask candidate
is a straightforward label-mask reference. Both receive arrays prepared by
the same public validation/canonicalization path. Correctness is tested in
`tests/test_routing.py`; benchmark execution only measures candidates.

Run the focused routing cases with:

```bash
uv run --group bench asv run --quick --bench 'routing\.'
```

It also measures `Pool.sum` immediately below and above a `uint8`-to-`uint16`
output-dtype boundary. Add other dtypes only when they select different
StochRoll code, not merely for coverage.

## Save and compare revisions

ASV builds selected project revisions in isolated environments and stores
machine-readable results under `.asv/results/`:

```bash
uv run --group bench asv run OLD_REVISION^!
uv run --group bench asv run NEW_REVISION^!
uv run --group bench asv compare OLD_REVISION NEW_REVISION
```

Ranges such as `OLD_REVISION..NEW_REVISION` use Git revision-range semantics.
Use `asv compare --help` to choose an explicit comparison factor when useful;
there is no project-wide timing threshold. Missing cases remain missing data
and are not silently accepted.

Only compare compatible machine, Python, NumPy, benchmark-version, and
configuration results. ASV records Git revision and timing samples in result
JSON; environment names and requirements identify Python/NumPy configuration,
machine metadata records platform details, and benchmark parameters identify
workloads and code paths. Include the command, revisions, machine,
environment, and relevant parameters when reporting results.

Results, environments, and generated reports under `.asv/` are local and
Git-ignored. Do not commit them or add benchmark execution to pytest or GitHub
Actions. Baselines are never updated automatically.

## Add a scenario

For public API coverage:

1. Use both `STANDARD_WORKLOADS`.
2. Construct deterministic inputs through `Roller` and the public domain API.
3. Use the normal Roll/Event `(R, 6)` and Pool `(R, 6, 12)` shapes unless the
   public operation itself requires a focused branch case.
4. Put only one public operation in each `time_*` method.

For a focused implementation comparison:

1. Add it only when StochRoll has custom optimized code.
2. Give NumPy and StochRoll the same prepared input array, final axis, and
   dtype.
3. Parameterize only sizes or dtypes that select meaningful code paths.
4. Put correctness and candidate-equivalence checks in ordinary tests, never
   in ASV setup or timing.

Shared public workload metadata and fixtures live in `_support.py`. Future
exploratory work can add a separate filtered module without changing the ASV
runner or expanding the normal reference matrix.
