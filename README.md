# stochroll

## Shape semantics

Every stochastic value has a nonempty leading **repetitions axis**. Additional
axes describe the structure of each simulation sample:

- `Roller.d(sides, shape=shape)` returns
  `(repetitions, *shape)`.
- `Roller.pool(dice, d=sides, shape=shape)` returns
  `(repetitions, *shape, dice)`. The final axis is the unresolved dice axis.

`shape` may be an integer, a tuple of integers, or `None`. Its entries must be
non-negative. Zero-sized structural axes are valid and represent empty
collections. For example, `r.d(6, shape=0)` represents zero d6 rolls in every
simulation sample and has shape `(repetitions, 0)`.

Reductions over empty structural axes follow NumPy semantics:

- `sum()` returns the additive identity, zero.
- `mean()` returns `NaN` and emits `RuntimeWarning`.
- `min()` and `max()` raise `ValueError`.
- Reducing a nonempty axis remains valid when another axis is empty.

Shape reductions may target only structural axes. Negative axes are normalized
as in NumPy, but axis `0`, its negative equivalent, tuples containing it, and
`axis=None` are rejected because they would remove the repetitions axis and
produce a value that can no longer represent stochastic samples.

The repetitions axis must contain at least one sample because probabilities and
expected values divide by the number of repetitions. A `Pool` must contain at
least one die because operations such as `first()`, `min()`, and `max()` require
a die.
