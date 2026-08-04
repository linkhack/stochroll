# WP-009-02 Callback Runner Evidence

## Result

The isolated prototype implements a direct generic callback runner on the
completed `ActiveBatch`. It centralizes bounded iteration, exact scalar
activity validation, stable active packing, adapter-controlled merge-back,
monotonicity, termination metadata, and recoverable limit failure. It does not
traverse state, infer fields, draw randomness, or add a production export.

The first implementation deliberately has no density dispatch, callback
fusion, cached activity, reflection, or other microoptimization. It validates
only the runner's public invariants. Existing `ActiveBatch` validation remains
responsible for wrapper counts, trailing shapes, wrapper kinds, Pool ownership,
sides, and dtypes; caller-defined adapters remain responsible for declaring
every state field.

## Runner and caller responsibilities

The generic runner owns this flow:

1. validate a positive `max_steps` accepted by `operator.index`;
2. evaluate an exact `(R,)` dense activity `Event`;
3. create one stable `ActiveBatch` for the active repetitions;
4. ask the adapter for compact state;
5. call the domain transition with that compact state and the same batch;
6. ask the adapter to merge a complete compact update into dense state;
7. validate next activity and reject reactivation;
8. record newly terminal repetitions; and
9. return dense state or raise with a dense partial result at the bound.

The caller keeps each scenario's immutable state dataclass, explicit adapter,
activity predicate, and transition together. The runner never receives domain
rules. The transition never receives dense state or permission to merge it.
The equivalent manual references reuse the exact same adapters, activity
predicates, and transitions, so their duplicated code is orchestration rather
than game behavior.

Lantern Run uses scalar `haul` and `busted` fields. Its activity predicate
stops busted repetitions and expresses the successful six-room horizon as an
all-false event. Canonical Dragon Hunt uses scalar dragon HP plus four-player
HP, matching WP-012's event-masked scenario profile. Its activity excludes a
dead dragon and battles with no living player. The compact transition uses
WP-012 to select three phase-start living targets with replacement, then routes
simultaneous damage back to the fixed player axis.

A separate reporting variant adds player damage dealt, hits, and damage taken.
Its five-field adapter makes the main ergonomics trade-off visible: packing and
merging are repetitive, but every correlation-bearing field and its wrapper
behavior is explicit. Keeping it separate prevents reporting state from
changing the canonical cross-package workload.

## Termination and failure evidence

`SimulationResult` returns dense state, the number of globally executed
transitions, and an `int64` termination-step `Roll`. Initially inactive rows
receive zero, rows ending after a transition receive its one-based number, and
unfinished rows retain `-1` in a limit result.

Immediate termination performs no batch creation, adapter call, transition,
or random draw. A repetition becoming terminal on the final permitted
transition succeeds. Remaining activity after the bound raises
`SimulationLimitExceeded` with the dense partial result and exact remaining
activity. Callback exceptions propagate unchanged. Reactivation fails before
another transition.

Deterministic tests cover mixed initial activity, mixed termination turns,
successful horizons, limit exhaustion, invalid activity kind/rank/count,
reactivation, callback propagation, adapter call order, exact batch reuse,
inactive preservation, non-mutation, invalid compact Pool metadata, compact
Pool drawing/reduction/reroll/merge, and zero-player lifecycle exclusion.

## RNG and scenario equivalence

The runner performs no draws. Both scenario comparisons start separate parent
generators from the same seed and record every integer call. Callback-runner
and hand-written executions are bitwise equal for all dense state fields,
termination metadata, step count, draw values implied by the results, call
count, scalar or array upper bounds, request shapes, and dtypes.

For 2,000 repetitions, Lantern Run executed six global steps and 5,631 active
row transitions. Termination counts for steps one through six were 633, 422,
332, 202, 116, and 295. It drew one room value per active transition.

Dragon Hunt executed nine global steps and 10,411 active row transitions.
Termination counts at steps two through nine were 1, 122, 501, 607, 458, 239,
65, and 7. Its shared stream recorded 228,841 draws. These counts match the
completed WP-012 event-masked scenario because runner bookkeeping adds no RNG
work.

## Typing observation

Strict mypy preserves the scenario state type through
`StateAdapter[StateT]`, both callbacks, `run_simulation`, and
`SimulationResult[StateT]`. Successful Lantern results retain
`SimulationResult[LanternState]`; the termination field remains a `Roll`.
Mismatched adapters, activity returns, and transition returns are rejected.

Python exception matching cannot express a caught generic specialization.
`except SimulationLimitExceeded as error` is therefore observed by mypy as
`SimulationLimitExceeded[Any]`; its `result.state` is `Any`, while `active`
retains `Event`. Callers that require typed partial state would need a
non-exception result or an explicit narrowing boundary. WP-009 retains that
production decision.

## Benchmarks

The benchmark suite is split by concern:

- `wp009_02_implementations.py` compares the direct generic runner with the
  hand-written ActiveBatch loop at active fractions 100%, 75%, 25%, 1%, and
  0%.
- `wp009_02_scenarios.py` compares complete Lantern Run and Dragon Hunt paths
  at 2,000, 10,000, and 100,000 repetitions, and keeps the wider reporting
  variant as a separate pair of methods.
- `wp009_02_api.py` isolates adapter take, adapter merge, and immediate runner
  termination.
- `wp009_scenarios.py` is the WP-009 umbrella suite. It measures stable Lantern
  and HP-only event-masked Dragon entry points, with historical instrumented
  compatibility loaders and a raw series beginning at WP-009-02. The older
  fixed-slot Dragon baseline remains a distinctly named series. Its `_get...`
  dispatchers import and return package-specific callables; ASV `setup()` binds
  them to `self.run...`, leaving timed methods to invoke only the prepared
  function. Unsupported historical raw cases raise `SkipNotImplemented`.

The implementation and API benchmarks import candidate implementations behind
setup or individual benchmark methods and call functions through
module-qualified names. Future functions added to an existing implementation
module therefore fail only their own benchmark methods on revisions that lack
them. The scenario benchmark imports its WP-009-02-local scenario modules at
module scope, guarded only so ASV can discover the umbrella suite against
earlier installed revisions; its methods remain WP-009-02-local and are not
measured for earlier milestones.

Environment: Python 3.12.13, NumPy 2.5.1, Linux x86-64 under WSL2. A
current-tree, non-persisting three-sample run used one invocation and no
warmup:

```text
uv run asv run --config asv.wp009.conf.json -E existing --dry-run \
  --bench wp009_02 -a repeat=3 -a number=1 -a warmup_time=0
```

The initial nine benchmark methods and their parameter matrices completed
before the comparable-scenario amendment. API cases
measured approximately 25 microseconds for adapter take, 48 microseconds for
adapter merge, and 52 microseconds for immediate termination at 10,000
repetitions. The Lantern implementation comparison ranged from roughly 58
microseconds at zero activity to 2.10 milliseconds at full activity for the
runner and 64 microseconds to 1.36 milliseconds for the hand-written loop.

Complete callback/manual observations were approximately 1.29/0.91 ms at
2,000 Lantern repetitions, 2.09/1.38 ms at 10,000, and 10.3/11.8 ms at
100,000. Dragon Hunt measured approximately 9.55/8.78 ms, 26.2/28.5 ms, and
269/237 ms at the same sizes. Three samples with one invocation are noisy;
these values demonstrate cost visibility and successful execution, not stable
performance rankings or thresholds.

A refreshed current-tree focused quick run on 2026-08-04 completed all 11
WP-009-02 benchmark methods:

```text
uv run --group bench asv run --config asv.wp009.conf.json -E existing \
  --quick --bench 'wp009_02'
```

The run used the same Python/NumPy environment and reported these approximate
times (callback runner / hand-written loop):

| Workload | 2,000 repetitions | 10,000 repetitions | 100,000 repetitions |
| --- | ---: | ---: | ---: |
| Lantern | 8.34 / 9.19 ms | 8.21 / 9.52 ms | 22.0 / 17.4 ms |
| Dragon Hunt | 21.2 / 18.4 ms | 31.6 / 38.1 ms | 203 / 201 ms |
| Dragon Hunt reporting | 21.0 / 17.6 ms | 36.5 / 34.8 ms | 247 / 263 ms |

The active-fraction comparison at 10,000 repetitions measured callback/manual
times of 4.14/1.95 ms at 100% activity, 2.62/2.01 ms at 75%, 1.39/1.16 ms
at 25%, 1.01/0.823 ms at 1%, and 108/109 microseconds at 0%. The isolated
API timings were 23.8 microseconds for adapter take, 37.3 microseconds for
adapter merge, and 104 microseconds for immediate termination. These quick
measurements are directional rather than release thresholds, but they make
the lifecycle cost visible: the generic runner is close to the manual path in
scenario workloads, while explicit compact-state packing and dense merge-back
are measurable overhead at high activity.

After the amendment, a current-tree quick run completed all 16 WP-009-02 and
umbrella methods. A final scenario-only run completed all 11 canonical,
reporting, raw, and instrumented methods. Historical selected-revision runs
completed the WP-009-01 Lantern/fixed-slot compatibility methods and all three
WP-012 instrumented methods. Exact cross-version output equivalence was
explicitly excluded; package-local callback/manual equality remains tested.

## Production implications and limits

The runner removes repeated lifecycle validation and bookkeeping while leaving
domain state and transitions typed and local. It also makes terminal-state
preservation and partial failure uniform. The cost is an explicit adapter for
every state shape, an additional callback boundary, and generic-exception type
erasure when caught.

Adapter completeness cannot be proven by the generic runner. Mutable objects,
implicit dataclass traversal, history, structural-cell activity, persistence,
parallelism, alternate RNG guarantees, and optimized runner strategies remain
outside this prototype.

## Decision

The benchmark evidence supports simulation-owned buffers as a useful next
step. The callback/manual results show that the generic lifecycle is viable and
often comparable, but the active-fraction and API measurements expose the cost
of repeatedly allocating compact updates and merging them back into dense
state. The production design should therefore keep typed, caller-owned domain
state at the API boundary while letting the simulation own reusable compact
active buffers and dense merge scratch space internally. Adapters should fill
and consume those buffers in place where practical.

This recommendation targets the measured overhead without changing the
validated callback semantics, explicit state ownership, or RNG behavior. It is
an optimization and API-design follow-up, not an implementation in this
prototype; mutable state, persistence, parallelism, and the final public
result/failure API still require separate decisions.
