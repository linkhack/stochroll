# StochRoll

StochRoll is a typed, vectorized Python library for simulating dice mechanics and estimating their outcomes with Monte Carlo sampling.

It is designed for tabletop and RPG mechanics that are awkward to express with a single dice expression: pools, keep/drop rules, rerolls, conditional damage, and probabilities of compound events.

## Installation

```bash
python -m pip install stochroll
```

StochRoll requires Python 3.12 or newer and NumPy.

## A 30-second example

Roll 4d6, drop the lowest die, add 3, and estimate the expected result and the chance of reaching 15:

```python
from stochroll import Roller

roller = Roller(repetitions=100_000, seed=42)
damage = roller.pool(4, d=6).drop_lowest().sum() + 3

print(f"Expected result: {damage.expected():.2f}")
print(f"P(result >= 15): {damage.probability_at_least(15):.1%}")
print(f"Typical result: {damage.quantile(0.5):.0f}")
print(f"Spread: {damage.standard_deviation():.2f}")
```

`seed` is optional. The default is one million repetitions, so estimates are useful without having to choose a sample size first.

## Events and conditional outcomes

Comparisons produce `Event` objects rather than ordinary booleans. This makes it possible to combine conditions and use them to select simulated values with `where`:

```python
from stochroll import Roller, where

roller = Roller(repetitions=200_000, seed=2026)

# Six independent attacks in every simulation sample.
d20 = roller.d(20, shape=6)
natural_1 = d20 == 1
critical = d20 == 20
hits = critical | ((d20 + 5 >= 15) & ~natural_1)

# A critical hit deals 2d6 + 3; a normal hit deals 1d8 + 3.
normal_damage = roller.d(8, shape=6) + 3
critical_damage = roller.pool(2, d=6, shape=6).sum() + 3
damage_per_attack = where(
    critical,
    critical_damage,
    where(hits, normal_damage, 0),
)

total_damage = damage_per_attack.sum()
print(f"Expected total damage: {total_damage.expected():.2f}")
print(f"P(4+ hits): {hits.indicator().sum().probability_at_least(4):.1%}")
```

The arrays behind these objects have a leading repetitions axis. `shape=6` adds six structural values to each simulation sample, so the second example evaluates six attacks in parallel. `broadcast_to(...)` can expand a shared value or event to match such a structural shape.

## Structural operations

`select` chooses fixed structural entries, while `lookup` can choose different
entries in every repetition. Axes use NumPy's absolute numbering: axis 0 is
always repetitions and cannot be selected. Explicit indices are zero-based,
must have an integer dtype, and cannot be negative.

The following single example shows fixed selection, one-axis shorthand,
explicit full-rank lookup across teams, singleton-axis construction, and Pool
structural selection and assembly:

```python
from stochroll import Roller, concatenate, stack, where

roller = Roller(repetitions=100_000, seed=9)

# One structural axis: (repetitions, players).
initiative = roller.d(20, shape=4)
first_player = initiative.select(0)
random_player = initiative.lookup(roller.d(4) - 1)  # (R,) shorthand -> (R, 1)

# Assemble independent values without exposing the repetitions axis.
second_party = roller.d(20, shape=4)
two_parties = stack([initiative, second_party])  # (R, parties=2, players=4)
all_players = concatenate([initiative, second_party])  # (R, players=8)

# Two structural axes: (repetitions, teams, players).
defense = roller.d(12, shape=(2, 4)) + 8
targets = roller.d(4, shape=2) - 1  # (R, teams)
target_defense = defense.lookup(targets.add_axis(), axis=-1)
# targets.add_axis() is (R, teams, 1), making the full-rank lookup explicit.

attack_rolls = roller.d(20, shape=(2, 1))
attack_hits = attack_rolls >= target_defense
dealt = where(attack_hits, roller.d(6, shape=(2, 1)), 0)
damage_by_player = dealt.route_sum(targets.add_axis(), size=4)

# Pool lookup/select operates only on structural axes and retains the dice axis.
team_pools = roller.pool(3, d=6, shape=2)  # (R, teams, dice)
first_team_pool = team_pools.select(0)  # (R, dice), still a Pool
```

For `Roll` and `Event`, scalar selection removes the selected axis; slices and
integer arrays replace it with their index shape. `lookup` normally requires
indices with the same rank as the source. Non-selected dimensions may be
singleton to broadcast or must match the source. The narrow `(R,)` and
`(R, K)` shorthand is available only when the source has exactly one
structural axis; multi-axis sources must make all dimensions explicit.
`Roll.add_axis()` and `Event.add_axis()` insert a singleton structural axis
without drawing new random values.

`Pool.select` and `Pool.lookup` default to axis `-2`, the last structural axis.
They reject axis `-1`, which is the dedicated dice axis. Use `first()`,
`last()`, or `single()` to resolve existing Pool dice positions. Lookup indices
remain caller-managed values; results do not retain hidden index provenance.

`stack` inserts a new structural axis using output-array coordinates, matching
NumPy; `concatenate` joins an existing structural axis using input-array
coordinates. Both default to `axis=1`, immediately after repetitions. Inputs
must use one wrapper type, have matching repetition counts, and have compatible
shapes without implicit broadcasting. Repetitions cannot be assembled because
they represent independent simulation samples rather than structural values.

Pool assembly additionally requires matching `sides`, matching dice extents,
and the same `Roller` object. `stack` may insert only before the final dice
axis, while `concatenate` may target only an existing structural axis. Neither
operation combines or resizes the Pool dice axis.

`route_sum` replaces the selected structural axis with a positive destination
extent and sums duplicate destinations; `route_any` performs the same routing
with Boolean OR collisions. Destinations are zero-based integer values, and
shaped destinations use explicit full-rank singleton broadcasting. The public
methods validate all sizes, axes, dtypes, bounds, ranks, repetitions, and
shapes before dispatching to the active routing implementation. Pool routing
and overwrite routing are not part of this API.

## Core concepts

| Concept | Purpose |
| --- | --- |
| `Roller` | Owns the random generator and creates simulations. `d(sides)` creates resolved die rolls; `pool(dice, d=sides)` creates unresolved dice pools. |
| `Pool` | Represents dice that still need a pool operation. It supports structural `select` and `lookup`; use `sum`, `min`, `max`, `first`, `last`, `single`, keep/drop methods, `reroll_once`, or `count_at_least` for dice operations. |
| `Roll` | Represents a resolved numeric outcome for every repetition. Rolls support arithmetic, comparisons, structural selection and lookup, singleton-axis insertion, broadcasting, reductions, expected values, and threshold probabilities. |
| `Event` | Represents a boolean condition for every repetition. Events support structural selection and lookup, singleton-axis insertion, `&`, `\|`, `~`, `indicator()`, `count()`, and `probability()`. Pass an event to `where` for conditional selection. |
| `stack` / `concatenate` | Assemble homogeneous Rolls, Events, or Pools along structural axes without combining repetitions. |

For example, `roller.d(20) >= 15` is an `Event`, while `roller.d(20) + 5` is a `Roll`. Events cannot be used in arithmetic; use `event.indicator()` for a numeric 0/1 score per repetition, `count()` to reduce structural axes, or `where(event, yes, no)` to choose between outcomes.

## Reproducibility and accuracy

Pass an integer `seed` to `Roller` to reproduce the same random stream:

```python
first = Roller(repetitions=10_000, seed=7).d(20)
second = Roller(repetitions=10_000, seed=7).d(20)
# first and second contain the same simulated values.
```

Reproducibility applies when the seed, repetition count, operation order, and draw shapes are the same. The random generator advances as values are drawn, so inserting or reordering a draw changes subsequent results. Results can also vary across changes to the Python/NumPy environment; StochRoll does not replace a statistical test with a fixed exact calculation.

Monte Carlo estimates converge with more repetitions but have sampling error. For an estimated probability `p` from `N` independent repetitions, the approximate standard error is `sqrt(p * (1 - p) / N)`. Expected values have standard error approximately `sigma / sqrt(N)`, where `sigma` is the standard deviation of the simulated quantity. Increase `repetitions` when you need tighter estimates, especially for rare events.

`variance()` and `standard_deviation()` summarize spread across repetitions;
they accept `ddof=0` by default, or a valid sample-statistics degree of freedom
below the repetition count. `quantile(q)` reports percentiles, with scalar `q`
preserving structural shape and array-like `q` adding leading quantile axes.
The default discrete `inverted_cdf` method is useful for dice outcomes; NumPy's
other supported quantile methods are available through `method=`. Quantiles may
sort or copy the simulation values, so they can use more memory than simple
summaries. `probability_at_most(target)` is the inclusive lower-tail counterpart
to `probability_at_least(target)`.

## Development

The repository uses [uv](https://docs.astral.sh/uv/). After cloning the project:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy
```

To build and validate a distribution:

```bash
uv build
uv run twine check dist/*
```

The test suite is in `tests/`; the worked example is in `examples/advanced_guidance.py`.

## License

StochRoll is distributed under the [MIT License](LICENSE).
