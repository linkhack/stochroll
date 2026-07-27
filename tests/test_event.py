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
