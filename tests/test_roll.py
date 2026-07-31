import operator
from typing import Any

import numpy as np
import pytest

from stochroll import Event, Roll


def test_all_arithmetic_operators_are_elementwise() -> None:
    left = Roll(np.array([[1, 2], [3, 4]], dtype=np.int64))
    right = Roll(np.array([[10, 20], [30, 40]], dtype=np.int64))

    np.testing.assert_array_equal((left + right).values, [[11, 22], [33, 44]])
    np.testing.assert_array_equal((left - right).values, [[-9, -18], [-27, -36]])
    np.testing.assert_array_equal((left * right).values, [[10, 40], [90, 160]])
    np.testing.assert_array_equal((5 + left).values, [[6, 7], [8, 9]])
    np.testing.assert_array_equal((5 - left).values, [[4, 3], [2, 1]])
    np.testing.assert_array_equal((5 * left).values, [[5, 10], [15, 20]])


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (operator.eq, [[False, True], [False, False]]),
        (operator.ne, [[True, False], [True, True]]),
        (operator.le, [[True, True], [False, False]]),
        (operator.lt, [[True, False], [False, False]]),
        (operator.ge, [[False, True], [True, True]]),
        (operator.gt, [[False, False], [True, True]]),
    ],
)
def test_all_comparison_operators_return_exact_events(
    operation: object, expected: list[list[bool]]
) -> None:
    roll = Roll(np.array([[1, 2], [3, 4]], dtype=np.int64))

    event = operation(roll, 2)  # type: ignore[operator]

    assert isinstance(event, Event)
    np.testing.assert_array_equal(event.values, expected)


def test_roll_broadcasting_uses_trailing_axes() -> None:
    left = Roll(np.array([[[1], [2]], [[3], [4]]], dtype=np.int64))
    right = Roll(np.array([[[10, 20, 30]], [[40, 50, 60]]], dtype=np.int64))

    np.testing.assert_array_equal(
        (left + right).values,
        [
            [[11, 21, 31], [12, 22, 32]],
            [[43, 53, 63], [44, 54, 64]],
        ],
    )


def test_broadcast_to_expands_each_simulation_sample() -> None:
    roll = Roll(np.array([1, 2], dtype=np.int64))

    np.testing.assert_array_equal(roll.broadcast_to(3).values, [[1, 1, 1], [2, 2, 2]])


def test_broadcast_to_expands_existing_structural_axes() -> None:
    roll = Roll(np.array([[1], [2]], dtype=np.int64))
    vector = Roll(
        np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ],
            dtype=np.int64,
        )
    )
    shaped = Roll(
        np.array(
            [
                [[1, 2, 3, 4]],
                [[5, 6, 7, 8]],
            ],
            dtype=np.int64,
        )
    )

    np.testing.assert_array_equal(
        roll.broadcast_to(3).values,
        [[1, 1, 1], [2, 2, 2]],
    )
    np.testing.assert_array_equal(
        vector.broadcast_to(3, 4).values,
        [
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            [[5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]],
        ],
    )
    np.testing.assert_array_equal(
        shaped.broadcast_to(3, 4).values,
        [
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            [[5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]],
        ],
    )


def test_broadcast_to_applies_scalar_sample_to_multidimensional_shape() -> None:
    roll = Roll(np.array([1, 2], dtype=np.int64))

    np.testing.assert_array_equal(
        roll.broadcast_to(2, 3).values,
        [
            [[1, 1, 1], [1, 1, 1]],
            [[2, 2, 2], [2, 2, 2]],
        ],
    )


def test_roll_reductions_are_exact() -> None:
    roll = Roll(np.array([[1, 6, 3], [4, 2, 5]], dtype=np.int64))

    np.testing.assert_array_equal(roll.sum().values, [10, 11])
    np.testing.assert_allclose(roll.mean().values, [10 / 3, 11 / 3])
    np.testing.assert_array_equal(roll.min().values, [1, 2])
    np.testing.assert_array_equal(roll.max().values, [6, 5])
    np.testing.assert_allclose(roll.expected(), [2.5, 4.0, 4.0])
    np.testing.assert_allclose(roll.probability_at_least(4), [0.5, 0.5, 0.5])


def test_distribution_statistics_match_numpy_for_scalar_and_shaped_rolls() -> None:
    values = np.array(
        [
            [[1, 2], [3, 4]],
            [[2, 4], [6, 8]],
            [[4, 8], [12, 16]],
            [[8, 16], [24, 32]],
        ],
        dtype=np.float32,
    )
    roll = Roll(values)

    np.testing.assert_allclose(
        roll.variance(ddof=1), np.var(values, axis=0, dtype=np.float64, ddof=1)
    )
    np.testing.assert_allclose(
        roll.standard_deviation(), np.std(values, axis=0, dtype=np.float64)
    )
    np.testing.assert_allclose(
        roll.quantile([0, 0.5, 1], method="linear"),
        np.quantile(values, [0, 0.5, 1], axis=0, method="linear"),
    )
    assert roll.variance().dtype == np.dtype(np.float64)
    assert roll.standard_deviation().dtype == np.dtype(np.float64)
    assert roll.quantile(0.5).dtype == np.dtype(roll.values.dtype)


def test_distribution_statistics_preserve_scalar_and_zero_structural_shapes() -> None:
    scalar = Roll(np.array([1, 2, 4], dtype=np.int64))
    assert scalar.variance().shape == ()
    assert scalar.standard_deviation().shape == ()
    assert scalar.quantile(0.5).shape == ()

    empty = Roll(np.empty((3, 2, 0), dtype=np.float64))
    assert empty.variance().shape == (2, 0)
    assert empty.standard_deviation().shape == (2, 0)
    assert empty.quantile(0.5).shape == (2, 0)


def test_probability_at_most_is_inclusive_and_preserves_shape() -> None:
    values = np.array([[1.0, 2.5], [2.0, 2.5], [3.0, 4.0]])
    roll = Roll(values)

    np.testing.assert_array_equal(
        roll.probability_at_most(2.5),
        np.count_nonzero(values <= 2.5, axis=0) / 3,
    )


@pytest.mark.parametrize("ddof", [-1, 3, 1.5, True])
def test_variance_and_standard_deviation_reject_invalid_ddof(ddof: object) -> None:
    roll = Roll(np.array([1, 2, 3], dtype=np.int64))

    with pytest.raises((TypeError, ValueError)):
        roll.variance(ddof=ddof)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        roll.standard_deviation(ddof=ddof)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "q", [-0.1, 1.1, np.nan, [0.25, np.inf], [True, False], [1, [1, 2]]]
)
def test_quantile_rejects_invalid_q(q: object) -> None:
    roll = Roll(np.array([1, 2, 3], dtype=np.int64))

    with pytest.raises((TypeError, ValueError)):
        roll.quantile(q)  # type: ignore[arg-type]


def test_quantile_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="unsupported quantile method"):
        Roll(np.array([1, 2, 3], dtype=np.int64)).quantile(0.5, method="unknown")


@pytest.mark.parametrize(
    "shape",
    [
        (10,),
        (10, 3),
        (10, 2, 3),
    ],
)
def test_expected_shape(shape: tuple[int, ...]) -> None:
    values = np.ones(shape, dtype=np.int64)
    result = Roll(values).expected()

    assert result.shape == shape[1:]
    assert result.dtype == np.float64


@pytest.mark.parametrize("dtype", [np.dtype(np.int64), np.dtype(np.float32)])
def test_scalar_roll_statistics_return_numpy_scalars(
    dtype: np.dtype[Any],
) -> None:
    roll = Roll(np.array([1, 2, 4], dtype=dtype))

    expected = roll.expected()
    probability = roll.probability_at_least(2)

    assert isinstance(expected, np.float64)
    assert isinstance(probability, np.float64)
    assert not isinstance(expected, np.ndarray)
    assert not isinstance(probability, np.ndarray)
    assert expected.shape == ()
    assert probability.shape == ()
    assert expected.dtype == np.dtype(np.float64)
    assert probability.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(expected, 7 / 3)
    np.testing.assert_allclose(probability, 2 / 3)


def test_shaped_roll_statistics_preserve_shape_and_float64_dtype() -> None:
    roll = Roll(
        np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[2, 3, 4], [5, 6, 7]],
                [[3, 4, 5], [6, 7, 8]],
            ],
            dtype=np.int64,
        )
    )

    expected = roll.expected()
    probability = roll.probability_at_least(5)

    assert isinstance(expected, np.ndarray)
    assert isinstance(probability, np.ndarray)
    assert expected.shape == (2, 3)
    assert probability.shape == (2, 3)
    assert expected.dtype == np.dtype(np.float64)
    assert probability.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(expected, [[2, 3, 4], [5, 6, 7]])
    np.testing.assert_allclose(
        probability,
        [[0, 0, 1 / 3], [2 / 3, 1, 1]],
    )


def test_zero_sized_roll_statistics_preserve_structural_shape() -> None:
    roll = Roll(np.empty((3, 2, 0), dtype=np.float64))

    expected = roll.expected()
    probability = roll.probability_at_least(1)

    assert isinstance(expected, np.ndarray)
    assert isinstance(probability, np.ndarray)
    assert expected.shape == (2, 0)
    assert probability.shape == (2, 0)
    assert expected.dtype == np.dtype(np.float64)
    assert probability.dtype == np.dtype(np.float64)


def test_roll_reductions_accept_structural_axes() -> None:
    roll = Roll(
        np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.int64,
        )
    )

    np.testing.assert_array_equal(roll.sum(axis=1).values, [[5, 7, 9], [17, 19, 21]])
    np.testing.assert_allclose(
        roll.mean(axis=1).values,
        [[2.5, 3.5, 4.5], [8.5, 9.5, 10.5]],
    )
    np.testing.assert_array_equal(roll.min(axis=1).values, [[1, 2, 3], [7, 8, 9]])
    np.testing.assert_array_equal(roll.max(axis=1).values, [[4, 5, 6], [10, 11, 12]])


def test_roll_unsigned_integer_boundaries_are_promoted_before_arithmetic() -> None:
    uint16_roll = Roll(np.array([[65535]], dtype=np.uint16))
    uint32_roll = Roll(np.array([[4_294_967_295]], dtype=np.uint32))

    np.testing.assert_array_equal((uint16_roll + 1).values, [[65536]])
    np.testing.assert_array_equal((uint32_roll + 1).values, [[4_294_967_296]])


def test_roll_arithmetic_does_not_overflow_at_uint8_boundary() -> None:
    roll = Roll(np.array([[255], [254]], dtype=np.uint8))

    np.testing.assert_array_equal((roll + 1).values, [[256], [255]])


@pytest.mark.parametrize("dtype", [np.dtype(np.float32), np.dtype(np.float64)])
def test_float_roll_arithmetic_and_fractional_comparisons(
    dtype: np.dtype[Any],
) -> None:
    roll = Roll(np.array([[1.25, -2.5], [3.5, 4.75]], dtype=dtype))

    np.testing.assert_allclose((roll + 0.5).values, [[1.75, -2.0], [4.0, 5.25]])
    np.testing.assert_allclose((roll - 0.5).values, [[0.75, -3.0], [3.0, 4.25]])
    np.testing.assert_allclose((roll * 0.5).values, [[0.625, -1.25], [1.75, 2.375]])
    np.testing.assert_allclose((0.5 + roll).values, [[1.75, -2.0], [4.0, 5.25]])
    np.testing.assert_allclose((0.5 - roll).values, [[-0.75, 3.0], [-3.0, -4.25]])
    np.testing.assert_allclose((0.5 * roll).values, [[0.625, -1.25], [1.75, 2.375]])

    np.testing.assert_array_equal(
        (roll >= 2.5).values,
        [[False, False], [True, True]],
    )
    np.testing.assert_array_equal(
        (roll < 2.5).values,
        [[True, True], [False, False]],
    )


@pytest.mark.parametrize("dtype", [np.dtype(np.float32), np.dtype(np.float64)])
def test_float_roll_reductions_and_statistics_are_exact(
    dtype: np.dtype[Any],
) -> None:
    roll = Roll(
        np.array(
            [
                [0.1, 0.2, 0.3],
                [1.0, 1.1, 1.2],
                [2.0, 2.1, 2.2],
            ],
            dtype=dtype,
        )
    )

    np.testing.assert_allclose(roll.sum().values, [0.6, 3.3, 6.3])
    np.testing.assert_allclose(roll.mean().values, [0.2, 1.1, 2.1])
    np.testing.assert_allclose(roll.min().values, [0.1, 1.0, 2.0])
    np.testing.assert_allclose(roll.max().values, [0.3, 1.2, 2.2])
    np.testing.assert_allclose(roll.expected(), [1.0333333, 1.1333333, 1.2333333])
    np.testing.assert_allclose(
        roll.probability_at_least(1.05),
        [1 / 3, 2 / 3, 2 / 3],
    )


def test_broadcast_comparisons_work_between_rolls() -> None:
    left = Roll(np.array([[[1], [2]], [[3], [4]]], dtype=np.int64))
    right = Roll(np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.int64))

    np.testing.assert_array_equal(
        (left < right).values,
        [
            [[False, True, True], [False, False, True]],
            [[True, True, True], [False, True, True]],
        ],
    )
