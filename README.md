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

## Core concepts

| Concept | Purpose |
| --- | --- |
| `Roller` | Owns the random generator and creates simulations. `d(sides)` creates resolved die rolls; `pool(dice, d=sides)` creates unresolved dice pools. |
| `Pool` | Represents dice that still need a pool operation. Use `sum`, `min`, `max`, `keep_highest`, `keep_lowest`, `drop_lowest`, `drop_highest`, `drop_lowest_sum`, `reroll_once`, or `count_at_least`. |
| `Roll` | Represents a resolved numeric outcome for every repetition. Rolls support arithmetic, comparisons, broadcasting, reductions, expected values, and threshold probabilities. |
| `Event` | Represents a boolean condition for every repetition. Events support `&`, `\|`, `~`, `indicator()`, `count()`, and `probability()`. Pass an event to `where` for conditional selection. |

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
