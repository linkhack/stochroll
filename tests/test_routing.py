import numpy as np
import pytest

from stochroll import Event, Roll
from stochroll._reductions import _default_sum_dtype
from stochroll._routing import (
    _route_any_indexed,
    _route_reference_any,
    _route_reference_sum,
    _route_sum_indexed,
)
from stochroll.core import _prepare_route_inputs

SUM_IMPLEMENTATIONS = (_route_sum_indexed, _route_reference_sum)
ANY_IMPLEMENTATIONS = (_route_any_indexed, _route_reference_any)

ROUTE_CASES = (
    pytest.param(
        np.array([5, 7, 11], dtype=np.int16),
        np.array([2, 0, 2], dtype=np.int8),
        4,
        -1,
        id="scalar",
    ),
    pytest.param(
        np.arange(12, dtype=np.int16).reshape(2, 2, 3) + 1,
        np.array(
            [
                [[0, 1, 0], [2, 2, 1]],
                [[1, 0, 2], [0, 1, 1]],
            ],
            dtype=np.int8,
        ),
        3,
        -1,
        id="shaped-trailing-axis",
    ),
    pytest.param(
        np.arange(12, dtype=np.int16).reshape(2, 2, 3) + 1,
        np.array(
            [
                [[0], [2]],
                [[1], [0]],
            ],
            dtype=np.int8,
        ),
        3,
        1,
        id="interior-axis",
    ),
    pytest.param(
        np.arange(12, dtype=np.int16).reshape(2, 2, 3) + 1,
        np.array([[[1], [2]]], dtype=np.int8),
        3,
        -1,
        id="shared-repetitions-and-repeated-write-destination",
    ),
    pytest.param(
        np.empty((2, 0, 3), dtype=np.int16),
        np.empty((2, 0, 1), dtype=np.int8),
        4,
        1,
        id="empty-write-axis",
    ),
)


def test_scalar_route_inserts_a_destination_axis() -> None:
    values = Roll(np.array([10, 20], dtype=np.int64))
    destinations = np.array([2, 0], dtype=np.int8)

    positive = values.route_sum(destinations, size=3, axis=1)
    negative = values.route_sum(destinations, size=3, axis=-1)

    assert positive.values.shape == (2, 3)
    np.testing.assert_array_equal(positive.values, [[0, 0, 10], [20, 0, 0]])
    np.testing.assert_array_equal(positive.values, negative.values)


def test_shaped_numeric_route_sums_duplicate_destinations() -> None:
    values = Roll(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16))
    destinations = np.array([[0, 1, 0], [1, 1, 2]], dtype=np.int8)

    result = values.route_sum(destinations, size=3)

    assert result.values.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(result.values, [[4, 2, 0], [0, 9, 6]])


def test_shaped_route_supports_explicit_singleton_broadcasting() -> None:
    values = Roll(np.arange(12, dtype=np.int64).reshape(2, 2, 3) + 1)
    independent = np.array(
        [
            [[0, 1, 0], [2, 2, 1]],
            [[1, 0, 2], [0, 1, 1]],
        ],
        dtype=np.int8,
    )
    shared_teams = independent[:, :1, :]
    repeated_attacks = independent[:, :, :1]

    independent_result = values.route_sum(independent, size=3, axis=-1)
    shared_result = values.route_sum(shared_teams, size=3, axis=-1)
    repeated_result = values.route_sum(repeated_attacks, size=3, axis=-1)

    np.testing.assert_array_equal(
        independent_result.values,
        _reference_sum(values.values, independent, size=3, axis=2),
    )
    np.testing.assert_array_equal(
        shared_result.values,
        _reference_sum(values.values, shared_teams, size=3, axis=2),
    )
    np.testing.assert_array_equal(
        repeated_result.values,
        _reference_sum(values.values, repeated_attacks, size=3, axis=2),
    )


def test_route_can_replace_an_interior_structural_axis() -> None:
    values = Roll(np.arange(12, dtype=np.int64).reshape(2, 2, 3) + 1)
    destinations = np.array(
        [
            [[0, 1, 0], [1, 2, 1]],
            [[2, 0, 1], [0, 1, 2]],
        ],
        dtype=np.int8,
    )

    result = values.route_sum(destinations, size=4, axis=1)

    assert result.values.shape == (2, 4, 3)
    np.testing.assert_array_equal(
        result.values,
        _reference_sum(values.values, destinations, size=4, axis=1),
    )


def test_event_route_combines_duplicate_destinations_with_or() -> None:
    events = Event(
        np.array(
            [
                [True, False, True],
                [False, False, True],
            ],
            dtype=np.bool_,
        )
    )
    destinations = np.array([[1, 1, 0], [2, 2, 2]], dtype=np.int8)

    result = events.route_any(destinations, size=3)

    assert isinstance(result, Event)
    np.testing.assert_array_equal(
        result.values, [[True, True, False], [False, False, True]]
    )


def test_scalar_event_route_preserves_boolean_identity() -> None:
    events = Event(np.array([True, False, True], dtype=np.bool_))
    destinations = np.array([1, 0, 1], dtype=np.int8)

    result = events.route_any(destinations, size=2)

    np.testing.assert_array_equal(
        result.values, [[False, True], [False, False], [False, True]]
    )


@pytest.mark.parametrize("values,destinations,size,axis", ROUTE_CASES)
@pytest.mark.parametrize("implementation", SUM_IMPLEMENTATIONS)
def test_sum_implementations_agree_after_shared_public_validation(
    implementation: object,
    values: np.ndarray,
    destinations: np.ndarray,
    size: int,
    axis: int,
) -> None:
    canonical_values, canonical_destinations, normalized_axis, normalized_size = (
        _prepare_route_inputs(
            values,
            destinations,
            size=size,
            axis=axis,
        )
    )

    result = implementation(  # type: ignore[operator]
        canonical_values,
        canonical_destinations,
        size=normalized_size,
        axis=normalized_axis,
        dtype=_default_sum_dtype(values.dtype),
    )

    np.testing.assert_array_equal(
        result,
        Roll(values).route_sum(destinations, size=size, axis=axis).values,
    )


@pytest.mark.parametrize("values,destinations,size,axis", ROUTE_CASES)
@pytest.mark.parametrize("implementation", ANY_IMPLEMENTATIONS)
def test_any_implementations_agree_after_shared_public_validation(
    implementation: object,
    values: np.ndarray,
    destinations: np.ndarray,
    size: int,
    axis: int,
) -> None:
    events = values % 3 == 0
    canonical_values, canonical_destinations, normalized_axis, normalized_size = (
        _prepare_route_inputs(
            events,
            destinations,
            size=size,
            axis=axis,
        )
    )

    result = implementation(  # type: ignore[operator]
        canonical_values,
        canonical_destinations,
        size=normalized_size,
        axis=normalized_axis,
    )

    np.testing.assert_array_equal(
        result,
        Event(events).route_any(destinations, size=size, axis=axis).values,
    )


@pytest.mark.parametrize("values,destinations,size,axis", ROUTE_CASES)
def test_route_conservation_across_contract_cases(
    values: np.ndarray,
    destinations: np.ndarray,
    size: int,
    axis: int,
) -> None:
    roll = Roll(values)
    events = Event(values % 3 == 0)
    routed_values = roll.route_sum(destinations, size=size, axis=axis)
    routed_events = events.route_any(destinations, size=size, axis=axis)

    if values.ndim == 1:
        expected_values = values
        expected_events = events.values
    else:
        normalized_axis = axis % values.ndim
        expected_values = values.sum(axis=normalized_axis, dtype=np.int64)
        expected_events = events.values.any(axis=normalized_axis)

    normalized_result_axis = axis % routed_values.values.ndim
    np.testing.assert_array_equal(
        routed_values.values.sum(axis=normalized_result_axis, dtype=np.int64),
        expected_values,
    )
    np.testing.assert_array_equal(
        routed_events.values.any(axis=normalized_result_axis),
        expected_events,
    )


def test_floating_route_conservation_uses_tolerance() -> None:
    values = Roll(np.array([[1.5, 2.0, 3.25], [4.0, 5.5, 6.0]], dtype=np.float32))
    events = Event(values.values > 3)
    destinations = np.array([[0, 1, 0], [1, 1, 2]], dtype=np.int8)

    routed_values = values.route_sum(destinations, size=3)
    routed_events = events.route_any(destinations, size=3)

    np.testing.assert_allclose(
        routed_values.values.sum(axis=1), values.values.sum(axis=1)
    )
    np.testing.assert_array_equal(
        routed_events.values.any(axis=1),
        events.values.any(axis=1),
    )


def test_empty_write_axes_use_reduction_identities() -> None:
    values = Roll(np.empty((2, 0, 3), dtype=np.int64))
    events = Event(np.empty((2, 0, 3), dtype=np.bool_))
    destinations = np.empty((2, 0, 1), dtype=np.int8)

    routed_values = values.route_sum(destinations, size=4, axis=1)
    routed_events = events.route_any(destinations, size=4, axis=1)

    assert routed_values.values.shape == (2, 4, 3)
    assert routed_events.values.shape == (2, 4, 3)
    np.testing.assert_array_equal(routed_values.values, 0)
    np.testing.assert_array_equal(routed_events.values, False)


def test_zero_non_write_axes_require_zero_destination_axes() -> None:
    values = Roll(np.empty((2, 0, 3), dtype=np.int64))

    with pytest.raises(ValueError, match="dimensions must be zero"):
        values.route_sum(np.zeros((2, 1, 3), dtype=np.int8), size=2, axis=2)

    result = values.route_sum(np.empty((2, 0, 3), dtype=np.int8), size=2, axis=2)
    assert result.values.shape == (2, 0, 2)


@pytest.mark.parametrize("size", [0, -1])
def test_route_rejects_non_positive_sizes(size: int) -> None:
    values = Roll(np.ones((2, 3), dtype=np.int64))

    with pytest.raises(ValueError, match="size must be positive"):
        values.route_sum(np.zeros((2, 3), dtype=np.int8), size=size)


@pytest.mark.parametrize("size", [True, 1.5, "3"])
def test_route_rejects_non_integer_sizes(size: object) -> None:
    values = Roll(np.ones((2, 3), dtype=np.int64))

    with pytest.raises(TypeError, match="size must be an integer"):
        values.route_sum(np.zeros((2, 3), dtype=np.int8), size=size)  # type: ignore[arg-type]


def test_route_accepts_numpy_integer_sizes() -> None:
    values = Roll(np.ones((2, 3), dtype=np.int64))

    result = values.route_sum(np.zeros((2, 3), dtype=np.int8), size=np.int64(2))

    assert result.values.shape == (2, 2)


@pytest.mark.parametrize(
    "destinations",
    [
        np.array([[0.0, 1.0]], dtype=np.float64),
        np.array([[True, False]], dtype=np.bool_),
        np.array([[-1, 0]], dtype=np.int8),
        np.array([[0, 3]], dtype=np.int8),
    ],
)
def test_route_rejects_invalid_destination_values(destinations: np.ndarray) -> None:
    values = Roll(np.ones((1, 2), dtype=np.int64))
    expected = TypeError if destinations.dtype.kind in "fb" else IndexError

    with pytest.raises(expected):
        values.route_sum(destinations, size=3)


def test_route_rejects_invalid_destination_shapes_and_axes() -> None:
    values = Roll(np.ones((2, 3, 4), dtype=np.int64))

    with pytest.raises(ValueError, match="same rank"):
        values.route_sum(np.zeros((2, 3), dtype=np.int8), size=2)
    with pytest.raises(ValueError, match="dimensions must be 1 or match"):
        values.route_sum(np.zeros((2, 2, 4), dtype=np.int8), size=2)
    with pytest.raises(ValueError, match="repetitions dimension"):
        values.route_sum(np.zeros((3, 3, 4), dtype=np.int8), size=2)

    for axis in (0, -3):
        with pytest.raises(ValueError, match="repetitions axis"):
            values.route_sum(np.zeros((2, 3, 4), dtype=np.int8), size=2, axis=axis)


def test_scalar_route_requires_exact_destination_shape() -> None:
    values = Roll(np.ones(2, dtype=np.int64))

    with pytest.raises(ValueError, match="same shape"):
        values.route_sum(np.zeros(1, dtype=np.int8), size=2)


def test_scalar_route_rejects_repetitions_axis() -> None:
    values = Roll(np.ones(2, dtype=np.int64))

    with pytest.raises(ValueError, match="repetitions axis"):
        values.route_sum(np.zeros(2, dtype=np.int8), size=2, axis=0)


def test_route_does_not_mutate_inputs() -> None:
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    destinations = np.array([[0, 1, 0], [1, 1, 2]], dtype=np.int8)
    values_before = values.copy()
    destinations_before = destinations.copy()

    Roll(values).route_sum(destinations, size=3)

    np.testing.assert_array_equal(values, values_before)
    np.testing.assert_array_equal(destinations, destinations_before)


def _reference_sum(
    values: np.ndarray,
    destinations: np.ndarray,
    *,
    size: int,
    axis: int,
) -> np.ndarray:
    canonical_values, canonical_destinations, normalized_axis, normalized_size = (
        _prepare_route_inputs(values, destinations, size=size, axis=axis)
    )
    moved_values = np.moveaxis(canonical_values, normalized_axis, -1)
    moved_destinations = np.moveaxis(canonical_destinations, normalized_axis, -1)
    output = np.zeros((*moved_values.shape[:-1], normalized_size), dtype=np.int64)
    for row in np.ndindex(output.shape[:-1]):
        for source_index in range(moved_values.shape[-1]):
            destination_index = moved_destinations[(*row, source_index)]
            output[(*row, destination_index)] += moved_values[(*row, source_index)]
    return np.moveaxis(output, -1, normalized_axis)
