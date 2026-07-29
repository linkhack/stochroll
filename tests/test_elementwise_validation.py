import operator
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from stochroll import Event, Roll, where


def test_roll_rank_mismatch_cannot_mix_repetitions() -> None:
    scalar_per_repetition = Roll(np.array([1, 2, 3], dtype=np.int64))
    matrix = Roll(np.arange(9, dtype=np.int64).reshape(3, 3))

    with pytest.raises(ValueError, match=r"ranks.*add_axis\(\)"):
        scalar_per_repetition + matrix


def test_add_axis_expresses_per_repetition_structural_broadcasting() -> None:
    scalar_per_repetition = Roll(np.array([1, 2, 3], dtype=np.int64))
    matrix = Roll(np.arange(9, dtype=np.int64).reshape(3, 3))

    result = scalar_per_repetition.add_axis() + matrix

    np.testing.assert_array_equal(
        result.values,
        [[1, 2, 3], [5, 6, 7], [9, 10, 11]],
    )


def test_equal_rank_rolls_retain_structural_broadcasting() -> None:
    column = Roll(np.array([[1], [2], [3]], dtype=np.int64))
    matrix = Roll(np.arange(12, dtype=np.int64).reshape(3, 4))
    planes = Roll(np.arange(6, dtype=np.int64).reshape(3, 2, 1))
    rows = Roll(np.arange(12, dtype=np.int64).reshape(3, 1, 4))

    np.testing.assert_array_equal(
        (column + matrix).values,
        [[1, 2, 3, 4], [6, 7, 8, 9], [11, 12, 13, 14]],
    )
    np.testing.assert_array_equal(
        (planes + rows).values,
        [
            [[0, 1, 2, 3], [1, 2, 3, 4]],
            [[6, 7, 8, 9], [7, 8, 9, 10]],
            [[12, 13, 14, 15], [13, 14, 15, 16]],
        ],
    )


def test_roll_repetition_axis_never_broadcasts() -> None:
    one_repetition = Roll(np.array([[10, 20]], dtype=np.int64))
    many_repetitions = Roll(np.arange(6, dtype=np.int64).reshape(3, 2))

    with pytest.raises(ValueError, match="matching repetitions"):
        one_repetition + many_repetitions


def test_scalar_and_reflected_roll_arithmetic_is_unchanged() -> None:
    roll = Roll(np.array([[1, 2], [3, 4]], dtype=np.int16))

    np.testing.assert_array_equal((roll + 5).values, [[6, 7], [8, 9]])
    np.testing.assert_array_equal((5 + roll).values, [[6, 7], [8, 9]])
    np.testing.assert_array_equal((5 * roll).values, [[5, 10], [15, 20]])
    np.testing.assert_array_equal((5 - roll).values, [[4, 3], [2, 1]])
    assert (roll + 5).values.dtype == np.dtype(np.int16)
    assert isinstance(roll + 5, Roll)


@pytest.mark.parametrize(
    "operation",
    [
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    ],
)
@pytest.mark.parametrize(
    ("left_shape", "right_shape", "message"),
    [
        ((3,), (3, 3), r"ranks.*add_axis\(\)"),
        ((1, 2), (3, 2), "matching repetitions"),
    ],
)
def test_all_roll_comparisons_validate_wrapper_axes(
    operation: Callable[[Any, Any], Any],
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    message: str,
) -> None:
    left = Roll(np.zeros(left_shape, dtype=np.int64))
    right = Roll(np.ones(right_shape, dtype=np.int64))

    with pytest.raises(ValueError, match=message):
        operation(left, right)


def test_roll_arithmetic_with_event_preserves_existing_error() -> None:
    roll = Roll(np.array([1, 2], dtype=np.int64))
    event = Event(np.array([True, False], dtype=np.bool_))

    with pytest.raises(TypeError, match="Event cannot be combined arithmetically"):
        operator.add(roll, event)


@pytest.mark.parametrize("operation", [operator.or_, operator.and_])
def test_event_boolean_operations_validate_wrapper_axes(
    operation: Callable[[Any, Any], Any],
) -> None:
    rank_one = Event(np.array([True, False, True], dtype=np.bool_))
    rank_two = Event(np.ones((3, 3), dtype=np.bool_))
    one_repetition = Event(np.ones((1, 3), dtype=np.bool_))

    with pytest.raises(ValueError, match=r"ranks.*add_axis\(\)"):
        operation(rank_one, rank_two)
    with pytest.raises(ValueError, match="matching repetitions"):
        operation(one_repetition, rank_two)


def test_event_boolean_operations_retain_structural_broadcasting() -> None:
    column = Event(np.array([[True], [False]], dtype=np.bool_))
    matrix = Event(np.array([[True, False, True], [True, True, False]]))

    disjunction = column | matrix
    conjunction = column & matrix

    np.testing.assert_array_equal(
        disjunction.values,
        [[True, True, True], [True, True, False]],
    )
    np.testing.assert_array_equal(
        conjunction.values,
        [[True, False, True], [False, False, False]],
    )
    assert disjunction.values.dtype == np.dtype(np.bool_)
    assert isinstance(disjunction, Event)


@pytest.mark.parametrize("operation", [operator.or_, operator.and_])
def test_event_boolean_operations_reject_rolls_on_either_side(
    operation: Callable[[Any, Any], Any],
) -> None:
    event = Event(np.array([True, False], dtype=np.bool_))
    roll = Roll(np.array([1, 0], dtype=np.int64))

    with pytest.raises(TypeError):
        operation(event, roll)
    with pytest.raises(TypeError):
        operation(roll, event)


def test_event_has_no_reflected_boolean_operators() -> None:
    assert "__ror__" not in Event.__dict__
    assert "__rand__" not in Event.__dict__


@pytest.mark.parametrize(
    "branch",
    [
        Roll(np.ones((3,), dtype=np.int64)),
        Roll(np.ones((1, 2), dtype=np.int64)),
    ],
)
def test_where_validates_each_roll_branch(branch: Roll) -> None:
    event = Event(np.ones((3, 2), dtype=np.bool_))

    with pytest.raises(ValueError):
        where(event, branch, 0)
    with pytest.raises(ValueError):
        where(event, 0, branch)


def test_where_retains_scalar_and_equal_rank_structural_broadcasting() -> None:
    event = Event(np.array([[True], [False], [True]], dtype=np.bool_))
    yes = Roll(np.arange(6, dtype=np.float32).reshape(3, 2))

    result = where(event, yes, -1.5)

    np.testing.assert_allclose(result.values, [[0, 1], [-1.5, -1.5], [4, 5]])
    assert result.values.dtype == np.dtype(np.float32)
    assert isinstance(result, Roll)
