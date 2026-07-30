import numpy as np
import pytest

from stochroll import Event, Roll
from stochroll._reductions import _default_sum_dtype
from stochroll._routing import (
    _route_all_indexed,
    _route_any_indexed,
    _route_multiply_indexed,
    _route_reference_all,
    _route_reference_any,
    _route_reference_multiply,
    _route_reference_sum,
    _route_sum_indexed,
)
from stochroll.core import _prepare_route_inputs

ROLL_ROUTES = (
    pytest.param(
        ("route_sum", _route_sum_indexed, _route_reference_sum, 0),
        id="sum",
    ),
    pytest.param(
        ("route_multiply", _route_multiply_indexed, _route_reference_multiply, 1),
        id="multiply",
    ),
)
EVENT_ROUTES = (
    pytest.param(
        ("route_any", _route_any_indexed, _route_reference_any, False),
        id="any",
    ),
    pytest.param(
        ("route_all", _route_all_indexed, _route_reference_all, True),
        id="all",
    ),
)

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


def _route_roll(
    route: tuple[str, object, object, int],
    values: np.ndarray,
    destinations: np.ndarray,
    *,
    size: int,
    axis: int = -1,
) -> Roll:
    method, _, _, _ = route
    return getattr(Roll(values), method)(destinations, size=size, axis=axis)


def _route_event(
    route: tuple[str, object, object, bool],
    values: np.ndarray,
    destinations: np.ndarray,
    *,
    size: int,
    axis: int = -1,
) -> Event:
    method, _, _, _ = route
    return getattr(Event(values), method)(destinations, size=size, axis=axis)


def _reference_route(
    route: tuple[str, object, object, int | bool],
    values: np.ndarray,
    destinations: np.ndarray,
    *,
    size: int,
    axis: int,
) -> np.ndarray:
    _, _, reference, _ = route
    canonical_values, canonical_destinations, normalized_axis, normalized_size = (
        _prepare_route_inputs(values, destinations, size=size, axis=axis)
    )
    if values.dtype == np.bool_:
        return reference(  # type: ignore[operator]
            canonical_values,
            canonical_destinations,
            size=normalized_size,
            axis=normalized_axis,
        )
    dtype = _default_sum_dtype(values.dtype)
    return reference(  # type: ignore[operator]
        canonical_values,
        canonical_destinations,
        size=normalized_size,
        axis=normalized_axis,
        dtype=dtype,
    )


@pytest.mark.parametrize("route", ROLL_ROUTES)
def test_scalar_roll_route_inserts_a_destination_axis(
    route: tuple[str, object, object, int],
) -> None:
    values = Roll(np.array([10, 20], dtype=np.int64))
    destinations = np.array([2, 0], dtype=np.int8)

    positive = _route_roll(route, values.values, destinations, size=3, axis=1)
    negative = _route_roll(route, values.values, destinations, size=3, axis=-1)

    assert positive.values.shape == (2, 3)
    _, _, _, identity = route
    expected = np.full((2, 3), identity, dtype=np.int64)
    expected[0, 2] = 10
    expected[1, 0] = 20
    np.testing.assert_array_equal(positive.values, expected)
    np.testing.assert_array_equal(positive.values, negative.values)


@pytest.mark.parametrize("route", ROLL_ROUTES)
def test_shaped_roll_route_combines_duplicate_destinations(
    route: tuple[str, object, object, int],
) -> None:
    values = Roll(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16))
    destinations = np.array([[0, 1, 0], [1, 1, 2]], dtype=np.int8)

    result = _route_roll(route, values.values, destinations, size=3)

    assert result.values.dtype == np.dtype(np.int64)
    _, _, _, identity = route
    expected = (
        [[4, 2, identity], [identity, 9, 6]]
        if identity == 0
        else [[3, 2, identity], [identity, 20, 6]]
    )
    np.testing.assert_array_equal(result.values, expected)


@pytest.mark.parametrize("route", ROLL_ROUTES)
def test_shaped_roll_route_supports_explicit_singleton_broadcasting(
    route: tuple[str, object, object, int],
) -> None:
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

    independent_result = _route_roll(route, values.values, independent, size=3, axis=-1)
    shared_result = _route_roll(route, values.values, shared_teams, size=3, axis=-1)
    repeated_result = _route_roll(
        route, values.values, repeated_attacks, size=3, axis=-1
    )

    np.testing.assert_array_equal(
        independent_result.values,
        _reference_route(route, values.values, independent, size=3, axis=2),
    )
    np.testing.assert_array_equal(
        shared_result.values,
        _reference_route(route, values.values, shared_teams, size=3, axis=2),
    )
    np.testing.assert_array_equal(
        repeated_result.values,
        _reference_route(route, values.values, repeated_attacks, size=3, axis=2),
    )


@pytest.mark.parametrize("route", ROLL_ROUTES)
def test_roll_route_can_replace_an_interior_structural_axis(
    route: tuple[str, object, object, int],
) -> None:
    values = Roll(np.arange(12, dtype=np.int64).reshape(2, 2, 3) + 1)
    destinations = np.array(
        [
            [[0, 1, 0], [1, 2, 1]],
            [[2, 0, 1], [0, 1, 2]],
        ],
        dtype=np.int8,
    )

    result = _route_roll(route, values.values, destinations, size=4, axis=1)

    assert result.values.shape == (2, 4, 3)
    np.testing.assert_array_equal(
        result.values,
        _reference_route(route, values.values, destinations, size=4, axis=1),
    )


@pytest.mark.parametrize("route", EVENT_ROUTES)
def test_event_route_combines_duplicate_destinations(
    route: tuple[str, object, object, bool],
) -> None:
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

    result = _route_event(route, events.values, destinations, size=3)

    assert isinstance(result, Event)
    _, _, _, identity = route
    expected = (
        [[True, True, False], [False, False, True]]
        if not identity
        else [[True, False, True], [True, True, False]]
    )
    np.testing.assert_array_equal(result.values, expected)


@pytest.mark.parametrize("route", EVENT_ROUTES)
def test_scalar_event_route_preserves_boolean_identity(
    route: tuple[str, object, object, bool],
) -> None:
    events = Event(np.array([True, False, True], dtype=np.bool_))
    destinations = np.array([1, 0, 1], dtype=np.int8)

    result = _route_event(route, events.values, destinations, size=2)

    _, _, _, identity = route
    expected = (
        [[False, True], [False, False], [False, True]]
        if not identity
        else [[True, True], [False, True], [True, True]]
    )
    np.testing.assert_array_equal(result.values, expected)


@pytest.mark.parametrize("values,destinations,size,axis", ROUTE_CASES)
@pytest.mark.parametrize("route", ROLL_ROUTES)
def test_roll_route_implementations_agree_after_shared_public_validation(
    route: tuple[str, object, object, int],
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

    _, implementation, _, _ = route
    result = implementation(  # type: ignore[operator]
        canonical_values,
        canonical_destinations,
        size=normalized_size,
        axis=normalized_axis,
        dtype=_default_sum_dtype(values.dtype),
    )

    np.testing.assert_array_equal(
        result,
        _route_roll(route, values, destinations, size=size, axis=axis).values,
    )


@pytest.mark.parametrize("values,destinations,size,axis", ROUTE_CASES)
@pytest.mark.parametrize("route", EVENT_ROUTES)
def test_event_route_implementations_agree_after_shared_public_validation(
    route: tuple[str, object, object, bool],
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

    _, implementation, _, _ = route
    result = implementation(  # type: ignore[operator]
        canonical_values,
        canonical_destinations,
        size=normalized_size,
        axis=normalized_axis,
    )

    np.testing.assert_array_equal(
        result,
        _reference_route(route, events, destinations, size=size, axis=axis),
    )
    np.testing.assert_array_equal(
        result,
        _route_event(route, events, destinations, size=size, axis=axis).values,
    )


@pytest.mark.parametrize("values,destinations,size,axis", ROUTE_CASES)
@pytest.mark.parametrize("roll_route", ROLL_ROUTES)
@pytest.mark.parametrize("event_route", EVENT_ROUTES)
def test_routes_preserve_reduction_across_contract_cases(
    roll_route: tuple[str, object, object, int],
    event_route: tuple[str, object, object, bool],
    values: np.ndarray,
    destinations: np.ndarray,
    size: int,
    axis: int,
) -> None:
    roll = Roll(values)
    events = Event(values % 3 == 0)
    routed_values = _route_roll(
        roll_route, roll.values, destinations, size=size, axis=axis
    )
    routed_events = _route_event(
        event_route, events.values, destinations, size=size, axis=axis
    )

    if values.ndim == 1:
        expected_values = values
        expected_events = events.values
    else:
        normalized_axis = axis % values.ndim
        expected_values = values.sum(axis=normalized_axis, dtype=np.int64)
        expected_events = events.values.any(axis=normalized_axis)

    normalized_result_axis = axis % routed_values.values.ndim
    _, _, _, roll_identity = roll_route
    roll_reduction = (
        routed_values.values.sum(axis=normalized_result_axis, dtype=np.int64)
        if roll_identity == 0
        else routed_values.values.prod(axis=normalized_result_axis, dtype=np.int64)
    )
    expected_roll_reduction = (
        expected_values
        if values.ndim == 1 or roll_identity == 0
        else values.prod(axis=normalized_result_axis, dtype=np.int64)
    )
    np.testing.assert_array_equal(roll_reduction, expected_roll_reduction)

    _, _, _, event_identity = event_route
    event_reduction = (
        routed_events.values.any(axis=normalized_result_axis)
        if not event_identity
        else routed_events.values.all(axis=normalized_result_axis)
    )
    expected_event_reduction = (
        expected_events
        if values.ndim == 1 or not event_identity
        else events.values.all(axis=normalized_result_axis)
    )
    np.testing.assert_array_equal(event_reduction, expected_event_reduction)


@pytest.mark.parametrize("roll_route", ROLL_ROUTES)
@pytest.mark.parametrize("event_route", EVENT_ROUTES)
def test_floating_routes_conserve_with_their_reduction_identities(
    roll_route: tuple[str, object, object, int],
    event_route: tuple[str, object, object, bool],
) -> None:
    values = Roll(np.array([[1.5, 2.0, 3.25], [4.0, 5.5, 6.0]], dtype=np.float32))
    events = Event(values.values > 3)
    destinations = np.array([[0, 1, 0], [1, 1, 2]], dtype=np.int8)

    routed_values = _route_roll(roll_route, values.values, destinations, size=3)
    routed_events = _route_event(event_route, events.values, destinations, size=3)

    _, _, _, roll_identity = roll_route
    if roll_identity == 0:
        np.testing.assert_allclose(
            routed_values.values.sum(axis=1), values.values.sum(axis=1)
        )
    else:
        np.testing.assert_allclose(
            routed_values.values.prod(axis=1), values.values.prod(axis=1)
        )

    _, _, _, event_identity = event_route
    if event_identity:
        np.testing.assert_array_equal(
            routed_events.values.all(axis=1), events.values.all(axis=1)
        )
    else:
        np.testing.assert_array_equal(
            routed_events.values.any(axis=1), events.values.any(axis=1)
        )


@pytest.mark.parametrize("roll_route", ROLL_ROUTES)
@pytest.mark.parametrize("event_route", EVENT_ROUTES)
def test_empty_write_axes_use_reduction_identities(
    roll_route: tuple[str, object, object, int],
    event_route: tuple[str, object, object, bool],
) -> None:
    values = Roll(np.empty((2, 0, 3), dtype=np.int64))
    events = Event(np.empty((2, 0, 3), dtype=np.bool_))
    destinations = np.empty((2, 0, 1), dtype=np.int8)

    routed_values = _route_roll(roll_route, values.values, destinations, size=4, axis=1)
    routed_events = _route_event(
        event_route, events.values, destinations, size=4, axis=1
    )

    assert routed_values.values.shape == (2, 4, 3)
    assert routed_events.values.shape == (2, 4, 3)
    _, _, _, roll_identity = roll_route
    _, _, _, event_identity = event_route
    np.testing.assert_array_equal(routed_values.values, roll_identity)
    np.testing.assert_array_equal(routed_events.values, event_identity)


@pytest.mark.parametrize("route", ROLL_ROUTES)
def test_roll_routes_require_zero_destination_axes_for_zero_source_axes(
    route: tuple[str, object, object, int],
) -> None:
    values = Roll(np.empty((2, 0, 3), dtype=np.int64))

    with pytest.raises(ValueError, match="dimensions must be zero"):
        _route_roll(
            route,
            values.values,
            np.zeros((2, 1, 3), dtype=np.int8),
            size=2,
            axis=2,
        )

    result = _route_roll(
        route,
        values.values,
        np.empty((2, 0, 3), dtype=np.int8),
        size=2,
        axis=2,
    )
    assert result.values.shape == (2, 0, 2)


@pytest.mark.parametrize("size", [0, -1])
@pytest.mark.parametrize("route", ROLL_ROUTES)
def test_roll_routes_reject_non_positive_sizes(
    route: tuple[str, object, object, int], size: int
) -> None:
    values = Roll(np.ones((2, 3), dtype=np.int64))

    with pytest.raises(ValueError, match="size must be positive"):
        _route_roll(route, values.values, np.zeros((2, 3), dtype=np.int8), size=size)


@pytest.mark.parametrize("size", [True, 1.5, "3"])
@pytest.mark.parametrize("route", ROLL_ROUTES)
def test_roll_routes_reject_non_integer_sizes(
    route: tuple[str, object, object, int], size: object
) -> None:
    values = Roll(np.ones((2, 3), dtype=np.int64))

    with pytest.raises(TypeError, match="size must be an integer"):
        _route_roll(  # type: ignore[arg-type]
            route, values.values, np.zeros((2, 3), dtype=np.int8), size=size
        )


@pytest.mark.parametrize("route", ROLL_ROUTES)
def test_roll_routes_accept_numpy_integer_sizes(
    route: tuple[str, object, object, int],
) -> None:
    values = Roll(np.ones((2, 3), dtype=np.int64))

    result = _route_roll(
        route, values.values, np.zeros((2, 3), dtype=np.int8), size=np.int64(2)
    )

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
@pytest.mark.parametrize("route", ROLL_ROUTES)
def test_roll_routes_reject_invalid_destination_values(
    route: tuple[str, object, object, int], destinations: np.ndarray
) -> None:
    values = Roll(np.ones((1, 2), dtype=np.int64))
    expected = TypeError if destinations.dtype.kind in "fb" else IndexError

    with pytest.raises(expected):
        _route_roll(route, values.values, destinations, size=3)


@pytest.mark.parametrize("route", ROLL_ROUTES)
def test_roll_routes_reject_invalid_destination_shapes_and_axes(
    route: tuple[str, object, object, int],
) -> None:
    values = Roll(np.ones((2, 3, 4), dtype=np.int64))

    with pytest.raises(ValueError, match="same rank"):
        _route_roll(route, values.values, np.zeros((2, 3), dtype=np.int8), size=2)
    with pytest.raises(ValueError, match="dimensions must be 1 or match"):
        _route_roll(route, values.values, np.zeros((2, 2, 4), dtype=np.int8), size=2)
    with pytest.raises(ValueError, match="repetitions dimension"):
        _route_roll(route, values.values, np.zeros((3, 3, 4), dtype=np.int8), size=2)

    for axis in (0, -3):
        with pytest.raises(ValueError, match="repetitions axis"):
            _route_roll(
                route,
                values.values,
                np.zeros((2, 3, 4), dtype=np.int8),
                size=2,
                axis=axis,
            )


@pytest.mark.parametrize("route", ROLL_ROUTES)
def test_scalar_roll_routes_require_exact_destination_shape(
    route: tuple[str, object, object, int],
) -> None:
    values = Roll(np.ones(2, dtype=np.int64))

    with pytest.raises(ValueError, match="same shape"):
        _route_roll(route, values.values, np.zeros(1, dtype=np.int8), size=2)


@pytest.mark.parametrize("route", ROLL_ROUTES)
def test_scalar_roll_routes_reject_repetitions_axis(
    route: tuple[str, object, object, int],
) -> None:
    values = Roll(np.ones(2, dtype=np.int64))

    with pytest.raises(ValueError, match="repetitions axis"):
        _route_roll(route, values.values, np.zeros(2, dtype=np.int8), size=2, axis=0)


@pytest.mark.parametrize("route", ROLL_ROUTES)
def test_roll_routes_do_not_mutate_inputs(
    route: tuple[str, object, object, int],
) -> None:
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    destinations = np.array([[0, 1, 0], [1, 1, 2]], dtype=np.int8)
    values_before = values.copy()
    destinations_before = destinations.copy()

    _route_roll(route, values, destinations, size=3)

    np.testing.assert_array_equal(values, values_before)
    np.testing.assert_array_equal(destinations, destinations_before)
