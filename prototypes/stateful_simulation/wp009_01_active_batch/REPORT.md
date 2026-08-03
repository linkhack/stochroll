# WP-009-01 ActiveBatch evidence

## Outcome

The isolated prototype demonstrates computational early termination by packing
whole active repetitions, drawing only for that compact batch, and merging
updates back into caller-owned dense state. This differs from semantic masking:
the existing dense loop protects terminal values but continues work and random
draws for repetitions that have already ended.

No production API is proposed by these files. The prototype subclass adds
`active_batch` only inside this evidence directory, and no symbol is exported
from `stochroll`.

## Lifecycle and state contract

Both reference scenarios use an explicit positive `max_steps` horizon, capped at
their domain horizon. A missing active batch terminates normally without a
transition or random draw. Repetitions still active at the configured horizon
are assigned that horizon as their termination step; bounded truncation is a
normal result rather than an exception.

Lantern Run treats a trap as terminal and also marks room six as a successful
domain horizon. Dragon Hunt treats dragon defeat, party defeat, and round 15 as
terminal. Each scenario returns dense final state plus a dense integer
termination step maintained by the caller.

Activity has exact shape `(R,)` and applies only to whole repetitions. Shaped
state, such as the four-player axis in Dragon Hunt, remains dense within every
active repetition. Structural-cell activity and history are deferred.

## Wrapper and Pool evidence

`take` and `merge` retain Roll, Event, and Pool wrapper types and structural
shapes. Merge is non-broadcasting, does not mutate its inputs, and changes only
stable active positions in its returned dense value.
Roll merge uses NumPy result-type promotion; Event stays Boolean. Pool merge
requires matching sides, dice extent, structural shape, dtype, and ownership.
Every compact Pool entry is a real die result, so sum, min/max, keep/drop, and
value-dependent `reroll_once` need no inactive filler.

The tests cover scalar and multidimensional state, invalid counts and wrapper
pairs, Pool ownership and metadata, stable read-only positions, inactive-state
preservation, all-active equivalence, zero-active non-consumption, Pool reroll
counts, and overload-specific typing.

## RNG and compatibility

The compact Roller view shares the parent's generator. Draws follow stable
ascending active positions. The same seed, environment, operation order,
activity trajectory, and shapes are reproducible. Once activity becomes
partial, packed and dense masking intentionally diverge because their draw
shapes and stream consumption differ. The prototype supplies neither private
per-repetition substreams nor an adaptive density threshold.

Production `Roller.d`, `Roller.pool`, fixed-round examples, exports, and seeded
behavior are unchanged.

## Benchmark method

The isolated `asv.wp009.conf.json` configuration keeps prototype cases out of
the normal ASV matrix. `ActiveFraction` compares one shaped d20 draw at active
fractions 100%, 75%, 25%, 1%, and 0%. It also compares ActiveBatch and direct
NumPy implementations for `take`, `merge`, and the combined batch, roll, take,
and merge path. `ReferenceScenarios` compares dense and packed Lantern Run plus
dense, packed, and NumPy-compacted Dragon Hunt at 2,000, 10,000, and 100,000
repetitions. Ordinary tests assert draw contracts but contain no timing
threshold.

The final child-branch measurement ran non-persistently against revision
`bf39803` on Python 3.12 and NumPy 2.5.1. The active-fraction draw counts for
10,000 repetitions and structural shape 4 are deterministic:

| Active | Dense draws | Packed draws | Dense time | Packed time |
| ---: | ---: | ---: | ---: | ---: |
| 100% | 40,000 | 40,000 | 110 ± 3 µs | 108 ± 3 µs |
| 75% | 40,000 | 30,000 | 112 ± 2 µs | 83.9 ± 1 µs |
| 25% | 40,000 | 10,000 | 109 ± 1 µs | 35.5 ± 4 µs |
| 1% | 40,000 | 400 | 110 ± 5 µs | 9.13 ± 0.5 µs |
| 0% | 40,000 | 0 | 109 ± 4 µs | 85.8 ± 3 ns |

The combined batch, draw, take, update, and non-mutating merge path exposes the
cost of the complete immutable boundary:

| Active | Dense combined | Packed combined |
| ---: | ---: | ---: |
| 100% | 192 ± 2 µs | 351 ± 6 µs |
| 75% | 186 ± 3 µs | 277 ± 3 µs |
| 25% | 185 ± 3 µs | 126 ± 4 µs |
| 1% | 186 ± 5 µs | 53.8 ± 2 µs |
| 0% | 183 ± 1 µs | 1.84 ± 0.05 µs |

The retained single-compaction scenario results across every configured size
were:

| Scenario | 2,000 | 10,000 | 100,000 |
| --- | ---: | ---: | ---: |
| Lantern Run, dense | 593 ± 1 µs | 1.73 ± 0.01 ms | 15.2 ± 0.3 ms |
| Lantern Run, packed | 712 ± 5 µs | 1.44 ± 0.05 ms | 10.1 ± 0.1 ms |
| Dragon Hunt, dense | 11.4 ± 0.4 ms | 35.4 ± 0.5 ms | 315 ± 4 ms |
| Dragon Hunt, packed | 8.59 ± 0.06 ms | 19.4 ± 1 ms | 140 ± 3 ms |
| Dragon Hunt, NumPy compacted | 7.92 ± 0.1 ms | 17.1 ± 0.3 ms | 132 ± 2 ms |

The deterministic draw and transition observations for the previously cited
scenario sizes remain:

| Scenario | Dense draws | Packed draws | Dense transitions | Packed transitions | Dense time | Packed time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lantern Run, 10,000 repetitions | 60,000 | 27,596 | 60,000 | 27,596 | 1.73 ± 0.01 ms | 1.44 ± 0.05 ms |
| Dragon Hunt, 2,000 repetitions | 660,000 | 241,670 | 30,000 | 10,985 | 11.4 ± 0.4 ms | 8.59 ± 0.06 ms |

### Dragon Hunt compaction granularity

A review experiment added a second active batch between the player and dragon
attack phases so dragons defeated by players consumed no counterattack draws.
At 2,000 repetitions it reduced packed consumption from 241,670 draws across
10,985 round-level transitions to 220,102 draws across 10,871 transitions.
The additional position discovery, compact-batch setup, packing, merge-back,
and allocation nevertheless decreased end-to-end performance in the owner's
measurement. No exact timing value or recoverable implementation was retained,
so that performance comparison is qualitative and is not presented as a
portable threshold.

The milestone therefore retains one compaction per round. Phase-local
recompaction remains semantically valid, but should be justified by the cost
of the work it skips rather than by draw count alone.

An isolated packed draw is already competitive when all repetitions are
active, but the complete immutable batch/take/merge path is slower at 100% and
75% activity. It crosses over by the measured 25% point and is materially
faster at lower activity. Lantern Run has fixed overhead at 2,000 repetitions
but benefits at 10,000 and 100,000; Dragon Hunt benefits at all measured sizes.
These timing values are descriptive evidence, not portable thresholds or
correctness requirements.

## Recommendation

The evidence supports a bounded production package for explicit
whole-repetition packing and merge-back, subject to the remaining WP-009
comparison with event-masked sampling and callback orchestration. Production
names, module placement, Pool persistence, performance strategy, history, and
stronger RNG guarantees remain owner decisions. A hidden density switch should
not be adopted because it silently changes RNG consumption. Production design
should likewise avoid unconditional phase-local recompaction: fewer random
draws did not imply lower runtime in the Dragon Hunt experiment.
