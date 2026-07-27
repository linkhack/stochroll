from operator import add, mul, sub

import numpy as np
import pytest

from stochroll import Event, Pool, Roll, Roller
from stochroll.core import _normalize_axis


def test_repetitions_must_be_positive() -> None:
    with pytest.raises(ValueError, match="repetitions must be >= 1"):
        Roller(repetitions=0)
    with pytest.raises(ValueError, match="repetitions must be >= 1"):
        Roller(repetitions=-1)


@pytest.mark.parametrize("sides", [0, -1])
def test_die_sides_must_be_positive(sides: int) -> None:
    with pytest.raises(ValueError, match=rf"sides must be >= 1, got {sides}"):
        Roller(repetitions=2).d(sides)


def test_die_shape_entries_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="shape entries must be non-negative"):
        Roller(repetitions=2).d(6, shape=(2, -1))


@pytest.mark.parametrize("dice", [0, -1])
def test_pool_dice_count_must_be_positive(dice: int) -> None:
    with pytest.raises(ValueError, match=rf"dice must be >= 1, got {dice}"):
        Roller(repetitions=2).pool(dice, d=6)


def test_pool_die_size_and_shape_are_validated() -> None:
    roller = Roller(repetitions=2)
    with pytest.raises(ValueError, match="d must be >= 1, got 0"):
        roller.pool(2, d=0)
    with pytest.raises(ValueError, match="shape entries must be non-negative"):
        roller.pool(2, d=6, shape=(-1,))


@pytest.mark.parametrize(
    ("method", "k"),
    [
        ("keep_highest", -1),
        ("keep_highest", 5),
        ("keep_lowest", -1),
        ("keep_lowest", 5),
        ("drop_lowest", -1),
        ("drop_lowest", 5),
        ("drop_highest", -1),
        ("drop_highest", 5),
    ],
)
def test_pool_selector_k_must_be_in_range(method: str, k: int) -> None:
    pool = Roller(repetitions=2).pool(4, d=6)
    with pytest.raises(ValueError, match=rf"k must be between 0 and 4, got {k}"):
        getattr(pool, method)(k)


@pytest.mark.parametrize("method", ["keep_highest", "keep_lowest"])
def test_pool_keep_selector_zero_is_consistently_rejected(method: str) -> None:
    pool = Roller(repetitions=2).pool(4, d=6)
    with pytest.raises(ValueError, match="k=0 is not supported"):
        getattr(pool, method)(0)


@pytest.mark.parametrize("method", ["drop_lowest", "drop_highest"])
def test_pool_drop_selector_n_is_consistently_rejected(method: str) -> None:
    n = 4
    pool = Roller(repetitions=2).pool(n, d=6)
    with pytest.raises(ValueError, match="k=n is not supported"):
        getattr(pool, method)(n)


def test_reroll_values_must_be_on_the_die() -> None:
    pool = Roller(repetitions=2).pool(4, d=6)
    with pytest.raises(ValueError, match="reroll values must be between 1 and 6"):
        pool.reroll_once([0, 6, 7])


@pytest.mark.parametrize("values", [1.0, [1.0, 2.0]])
def test_reroll_accepts_integral_float_values(values: object) -> None:
    pool = Roller(repetitions=2).pool(4, d=6)

    result = pool.reroll_once(values)

    assert result.values.shape == pool.values.shape


@pytest.mark.parametrize("values", [1.5, [1.0, 2.5]])
def test_reroll_rejects_fractional_values(values: object) -> None:
    pool = Roller(repetitions=2).pool(4, d=6)

    with pytest.raises(ValueError, match="reroll values must be integers"):
        pool.reroll_once(values)


def test_count_at_least_accepts_targets_outside_die_range() -> None:
    pool = Roller(repetitions=2).pool(4, d=6)

    np.testing.assert_array_equal(pool.count_at_least(7).values, np.zeros(2))
    np.testing.assert_array_equal(pool.count_at_least(0).values, np.full(2, 4))


@pytest.mark.parametrize("operation", [add, sub, mul])
def test_roll_arithmetic_rejects_events(operation: object) -> None:
    roller = Roller(repetitions=2)
    roll = roller.d(6)
    event = roll >= 4

    with pytest.raises(TypeError, match="Event cannot be combined arithmetically"):
        operation(roll, event)  # type: ignore[operator]


def test_reflected_roll_arithmetic_rejects_events() -> None:
    roller = Roller(repetitions=2)
    roll = roller.d(6)
    event = roll >= 4

    with pytest.raises(TypeError, match="Event cannot be combined arithmetically"):
        event + roll
    with pytest.raises(TypeError, match="Event cannot be combined arithmetically"):
        event - roll
    with pytest.raises(TypeError, match="Event cannot be combined arithmetically"):
        event * roll


def test_direct_pool_roll_event_construction_is_validated() -> None:
    roller = Roller(repetitions=2)

    with pytest.raises(ValueError, match="repetitions must be >= 1"):
        Pool(np.empty((0, 2), dtype=np.int64), sides=6, roller=roller)
    with pytest.raises(ValueError, match="Pool must contain at least one die"):
        Pool(np.empty((2, 0), dtype=np.int64), sides=6, roller=roller)

    with pytest.raises(ValueError, match="repetitions must be >= 1"):
        Roll(np.empty((0,), dtype=np.int64))

    with pytest.raises(ValueError, match="repetitions must be >= 1"):
        Event(np.empty((0,), dtype=np.bool_))


def test_zero_sized_structural_shapes_are_supported() -> None:
    roller = Roller(repetitions=2)

    assert roller.d(6, shape=0).values.shape == (2, 0)
    assert roller.d(6, shape=(0, 2)).values.shape == (2, 0, 2)
    assert roller.d(6, shape=(2, 0)).values.shape == (2, 2, 0)
    assert roller.pool(2, d=6, shape=0).sum().values.shape == (2, 0)
    assert Event(np.empty((2, 0), dtype=np.bool_)).values.shape == (2, 0)


def test_empty_axis_reductions_follow_numpy_semantics() -> None:
    roll = Roll(np.empty((2, 0), dtype=np.int64))

    np.testing.assert_array_equal(roll.sum().values, np.zeros(2, dtype=np.int64))

    with pytest.warns(RuntimeWarning):
        mean = roll.mean()
    assert np.isnan(mean.values).all()

    with pytest.raises(ValueError, match="zero-size array"):
        roll.min()
    with pytest.raises(ValueError, match="zero-size array"):
        roll.max()

    nonempty_reduction_axis = Roll(np.empty((2, 0, 3), dtype=np.int64))
    assert nonempty_reduction_axis.max().values.shape == (2, 0)


@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        (1, (1,)),
        (-1, (2,)),
        ((1, -1), (1, 2)),
        ((), ()),
        (None, None),
    ],
)
def test_normalize_axis(
    axis: int | tuple[int, ...] | None,
    expected: tuple[int, ...] | None,
) -> None:
    assert _normalize_axis(axis, ndim=3) == expected


@pytest.mark.parametrize("axis", [3, -4, (1, 3)])
def test_normalize_axis_rejects_out_of_bounds_indices(
    axis: int | tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match=r"axis .* is out of bounds"):
        _normalize_axis(axis, ndim=3)


@pytest.mark.parametrize("axis", [0, -2, (1, 0), None])
def test_shape_reductions_reject_the_repetitions_axis(
    axis: int | tuple[int, ...] | None,
) -> None:
    roll = Roll(np.ones((2, 3), dtype=np.int64))
    event = Event(np.ones((2, 3), dtype=np.bool_))

    for reduction in (roll.sum, roll.mean, roll.min, roll.max, event.count):
        with pytest.raises(ValueError, match="cannot reduce the repetitions axis"):
            reduction(axis=axis)
