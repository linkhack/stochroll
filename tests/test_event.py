import numpy as np

from stochroll import Event, Roll, where


def test_vector_event_probability() -> None:
    event = Event(
        np.array(
            [
                [True, False],
                [True, True],
                [False, True],
                [True, False],
            ],
            dtype=np.bool_,
        )
    )

    np.testing.assert_allclose(event.probability(), [0.75, 0.5])


def test_event_boolean_operators_are_elementwise() -> None:
    left = Event(np.array([[True, False], [False, True]], dtype=np.bool_))
    right = Event(np.array([[True, True], [False, False]], dtype=np.bool_))

    np.testing.assert_array_equal(
        (left | right).values,
        [[True, True], [False, True]],
    )
    np.testing.assert_array_equal(
        (left & right).values,
        [[True, False], [False, False]],
    )
    np.testing.assert_array_equal(
        (~left).values,
        [[False, True], [True, False]],
    )


def test_event_count_and_probability_reduce_repetitions() -> None:
    event = Event(
        np.array(
            [
                [[True, False], [False, True]],
                [[True, True], [False, False]],
                [[False, True], [True, True]],
            ],
            dtype=np.bool_,
        )
    )

    np.testing.assert_array_equal(event.count().values, [[1, 1], [2, 0], [1, 2]])
    np.testing.assert_allclose(event.probability(), [[2 / 3, 2 / 3], [1 / 3, 2 / 3]])


def test_event_indicator_preserves_scalar_per_repetition_shape() -> None:
    event = Event(np.array([True, False, True], dtype=np.bool_))

    indicator: Roll = event.indicator()

    np.testing.assert_array_equal(indicator.values, [1, 0, 1])
    assert indicator.values.shape == event.values.shape
    assert np.issubdtype(indicator.values.dtype, np.signedinteger)


def test_event_indicator_preserves_shape_and_supports_arithmetic() -> None:
    values = np.array(
        [
            [[True, False], [False, True]],
            [[False, False], [True, True]],
        ],
        dtype=np.bool_,
    )
    event = Event(values)

    indicator = event.indicator()

    np.testing.assert_array_equal(
        indicator.values,
        [[[1, 0], [0, 1]], [[0, 0], [1, 1]]],
    )
    assert indicator.values.shape == event.values.shape
    np.testing.assert_array_equal(
        (indicator * 10 + 1).values,
        [[[11, 1], [1, 11]], [[1, 1], [11, 11]]],
    )


def test_event_indicator_handles_uniform_and_empty_structural_events() -> None:
    all_true = Event(np.ones((3, 2), dtype=np.bool_))
    all_false = Event(np.zeros((3, 2), dtype=np.bool_))
    empty = Event(np.empty((3, 0, 2), dtype=np.bool_))

    np.testing.assert_array_equal(
        all_true.indicator().values, np.ones((3, 2), dtype=np.int8)
    )
    np.testing.assert_array_equal(
        all_false.indicator().values, np.zeros((3, 2), dtype=np.int8)
    )
    assert empty.indicator().values.shape == (3, 0, 2)


def test_event_indicator_does_not_mutate_event() -> None:
    values = np.array([[True, False], [False, True]], dtype=np.bool_)
    event = Event(values)

    event.indicator()

    np.testing.assert_array_equal(event.values, values)
    assert event.values.dtype == np.bool_


def test_where_selects_rolls_and_scalars() -> None:
    event = Event(np.array([[True, False], [False, True]], dtype=np.bool_))
    yes = Roll(np.array([[10, 20], [30, 40]], dtype=np.int64))
    no = Roll(np.array([[1, 2], [3, 4]], dtype=np.int64))

    np.testing.assert_array_equal(
        where(event, yes, no).values,
        [[10, 2], [3, 40]],
    )
    np.testing.assert_array_equal(where(event, 9, 0).values, [[9, 0], [0, 9]])


def test_where_broadcasts_structural_axes() -> None:
    event = Event(np.array([[[True], [False]], [[False], [True]]], dtype=np.bool_))
    yes = Roll(np.array([[[10, 20, 30]], [[40, 50, 60]]], dtype=np.int64))

    np.testing.assert_array_equal(
        where(event, yes, 0).values,
        [
            [[10, 20, 30], [0, 0, 0]],
            [[0, 0, 0], [40, 50, 60]],
        ],
    )


def test_event_broadcast_to_expands_each_sample() -> None:
    event = Event(np.array([True, False], dtype=np.bool_))

    np.testing.assert_array_equal(
        event.broadcast_to(3).values,
        [[True, True, True], [False, False, False]],
    )


def test_event_broadcast_to_expands_existing_structural_axes() -> None:
    event = Event(np.array([[True], [False]], dtype=np.bool_))
    vector = Event(
        np.array(
            [
                [True, False, True, False],
                [False, True, False, True],
            ],
            dtype=np.bool_,
        )
    )
    shaped = Event(
        np.array(
            [
                [[True, False, True, False]],
                [[False, True, False, True]],
            ],
            dtype=np.bool_,
        )
    )

    np.testing.assert_array_equal(
        event.broadcast_to(3).values,
        [[True, True, True], [False, False, False]],
    )
    np.testing.assert_array_equal(
        vector.broadcast_to(3, 4).values,
        [
            [[True, False, True, False]] * 3,
            [[False, True, False, True]] * 3,
        ],
    )
    np.testing.assert_array_equal(
        shaped.broadcast_to(3, 4).values,
        [
            [[True, False, True, False]] * 3,
            [[False, True, False, True]] * 3,
        ],
    )


def test_where_supports_float_rolls_and_mixed_numeric_values() -> None:
    event = Event(np.array([[True, False], [False, True]], dtype=np.bool_))
    floats = Roll(np.array([[1.25, 2.5], [3.75, 4.5]], dtype=np.float32))
    integers = Roll(np.array([[1, 2], [3, 4]], dtype=np.int64))

    np.testing.assert_allclose(
        where(event, floats, 0.5).values,
        [[1.25, 0.5], [0.5, 4.5]],
    )
    np.testing.assert_allclose(
        where(event, integers, 0.5).values,
        [[1.0, 0.5], [0.5, 4.0]],
    )


def test_where_broadcasts_float_operands() -> None:
    event = Event(np.array([[[True], [False]], [[False], [True]]], dtype=np.bool_))
    floats = Roll(
        np.array(
            [
                [[1.25, 2.5, 3.75]],
                [[4.5, 5.75, 6.25]],
            ],
            dtype=np.float64,
        )
    )

    np.testing.assert_allclose(
        where(event, floats, -0.5).values,
        [
            [[1.25, 2.5, 3.75], [-0.5, -0.5, -0.5]],
            [[-0.5, -0.5, -0.5], [4.5, 5.75, 6.25]],
        ],
    )
