import numpy as np
import pytest

from stochroll import Event, Pool, Roll, Roller


def test_fixed_selection_preserves_each_wrapper_type_and_semantics() -> None:
    numeric = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    boolean = numeric % 2 == 0
    roller = Roller(repetitions=2, seed=42)
    pool = Pool(
        np.arange(48, dtype=np.int64).reshape(2, 3, 4, 2),
        sides=48,
        roller=roller,
    )

    roll_scalar = Roll(numeric).select(1, axis=1)
    event_slice = Event(boolean).select(slice(None, None, 2), axis=-1)
    pool_array = pool.select(np.array([2, 0], dtype=np.int16), axis=-3)

    assert isinstance(roll_scalar, Roll)
    assert isinstance(event_slice, Event)
    assert isinstance(pool_array, Pool)
    np.testing.assert_array_equal(roll_scalar.values, numeric[:, 1, :])
    np.testing.assert_array_equal(event_slice.values, boolean[..., ::2])
    np.testing.assert_array_equal(pool_array.values, pool.values[:, [2, 0], :, :])
    assert pool_array.sides == pool.sides
    assert pool_array.roller is pool.roller


def test_scalar_slice_and_sequence_selection_work_for_every_wrapper() -> None:
    numeric = np.arange(12, dtype=np.int64).reshape(2, 3, 2)
    roller = Roller(repetitions=2, seed=42)
    sources = (
        Roll(numeric),
        Event(numeric % 2 == 0),
        Pool(numeric + 1, sides=12, roller=roller),
    )

    for source in sources:
        scalar = source.select(1, axis=1)
        sliced = source.select(slice(-2, None), axis=-2)
        sequenced = source.select([2, 0], axis=1)

        assert isinstance(scalar, type(source))
        assert isinstance(sliced, type(source))
        assert isinstance(sequenced, type(source))
        np.testing.assert_array_equal(scalar.values, source.values[:, 1, ...])
        np.testing.assert_array_equal(sliced.values, source.values[:, -2:, ...])
        np.testing.assert_array_equal(
            sequenced.values,
            source.values[:, [2, 0], ...],
        )


def test_fixed_array_selection_replaces_axis_with_index_shape() -> None:
    values = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    indices = np.array([[2, 0]], dtype=np.int8)

    result = Roll(values).select(indices, axis=1)

    assert result.values.shape == (2, 1, 2, 4)
    np.testing.assert_array_equal(result.values, np.take(values, indices, axis=1))


def test_fixed_selection_supports_repeated_and_empty_indices() -> None:
    roll = Roll(np.arange(12, dtype=np.int64).reshape(2, 6))

    np.testing.assert_array_equal(
        roll.select([3, 3, 1]).values,
        roll.values[:, [3, 3, 1]],
    )
    assert roll.select(np.array([], dtype=np.int64)).values.shape == (2, 0)
    assert roll.select(slice(2, 2)).values.shape == (2, 0)


def test_one_axis_lookup_accepts_raw_and_roll_shorthand() -> None:
    values = np.array(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
        ],
        dtype=np.int64,
    )
    roll = Roll(values)

    scalar = roll.lookup(np.array([3, 0, 2], dtype=np.int8))
    multiple = roll.lookup(Roll(np.array([[0, 2], [1, 3], [3, 0]], dtype=np.int16)))
    shared = roll.lookup(np.array([[1, 3]], dtype=np.int32))
    one_shared = roll.lookup(np.array([2], dtype=np.uint8))

    np.testing.assert_array_equal(scalar.values, [[13], [20], [32]])
    np.testing.assert_array_equal(
        multiple.values,
        [[10, 12], [21, 23], [33, 30]],
    )
    np.testing.assert_array_equal(
        shared.values,
        [[11, 13], [21, 23], [31, 33]],
    )
    np.testing.assert_array_equal(one_shared.values, [[12], [22], [32]])


def test_event_lookup_preserves_boolean_wrapper() -> None:
    event = Event(
        np.array(
            [[True, False, False], [False, True, False]],
            dtype=np.bool_,
        )
    )

    result = event.lookup(np.array([0, 1], dtype=np.int64))

    assert isinstance(result, Event)
    np.testing.assert_array_equal(result.values, [[True], [True]])


def test_full_rank_lookup_broadcasts_explicit_singleton_dimensions() -> None:
    values = np.arange(2 * 3 * 4, dtype=np.int64).reshape(2, 3, 4)
    roll = Roll(values)

    shared_players = roll.lookup(
        np.array([[[1, 3]], [[0, 2]]], dtype=np.int8),
        axis=-1,
    )
    per_team = roll.lookup(
        Roll(
            np.array(
                [
                    [[0], [1], [2]],
                    [[3], [2], [1]],
                ],
                dtype=np.int8,
            )
        ),
        axis=2,
    )

    np.testing.assert_array_equal(
        shared_players.values,
        [
            [[1, 3], [5, 7], [9, 11]],
            [[12, 14], [16, 18], [20, 22]],
        ],
    )
    np.testing.assert_array_equal(
        per_team.values,
        [[[0], [5], [10]], [[15], [18], [21]]],
    )


def test_pool_structural_lookup_preserves_dice_and_metadata() -> None:
    roller = Roller(repetitions=2, seed=42)
    values = np.arange(2 * 3 * 2, dtype=np.int64).reshape(2, 3, 2) + 1
    pool = Pool(values, sides=12, roller=roller)

    shorthand = pool.lookup(np.array([2, 0], dtype=np.int8))
    shared = pool.lookup(np.array([[1, 2]], dtype=np.int16))
    full_rank = pool.lookup(
        Roll(np.array([[[0]], [[2]]], dtype=np.int64)),
        axis=1,
    )

    assert shorthand.values.shape == (2, 1, 2)
    assert shared.values.shape == (2, 2, 2)
    assert full_rank.values.shape == (2, 1, 2)
    np.testing.assert_array_equal(shorthand.values, values[[0, 1], [2, 0], None, :])
    np.testing.assert_array_equal(
        shared.values,
        np.stack((values[0, [1, 2]], values[1, [1, 2]])),
    )
    np.testing.assert_array_equal(
        full_rank.values,
        np.stack((values[0, [0]], values[1, [2]])),
    )
    assert shorthand.sides == pool.sides
    assert shorthand.roller is pool.roller


@pytest.mark.parametrize(
    "indices",
    [
        np.array([0.0, 1.0]),
        np.array([True, False]),
        Roll(np.array([0.0, 1.0])),
    ],
)
def test_selection_and_lookup_reject_noninteger_indices(indices: object) -> None:
    roll = Roll(np.arange(6, dtype=np.int64).reshape(2, 3))

    with pytest.raises(TypeError, match="integer dtype"):
        roll.select(indices)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer dtype"):
        roll.lookup(indices)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "indices",
    [
        -1,
        [-1, 0],
        np.array([0, 3], dtype=np.int64),
    ],
)
def test_selection_rejects_negative_and_out_of_bounds_indices(
    indices: object,
) -> None:
    roll = Roll(np.arange(6, dtype=np.int64).reshape(2, 3))

    with pytest.raises(IndexError):
        roll.select(indices)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "indices",
    [
        np.array([-1, 0], dtype=np.int64),
        np.array([0, 3], dtype=np.int64),
    ],
)
def test_lookup_rejects_negative_and_out_of_bounds_indices(
    indices: np.ndarray[tuple[int, ...], np.dtype[np.int64]],
) -> None:
    roll = Roll(np.arange(6, dtype=np.int64).reshape(2, 3))

    with pytest.raises(IndexError):
        roll.lookup(indices)


def test_lookup_requires_explicit_valid_ranks_and_dimensions() -> None:
    one_axis = Roll(np.arange(12, dtype=np.int64).reshape(3, 4))
    multi_axis = Roll(np.arange(24, dtype=np.int64).reshape(2, 3, 4))

    with pytest.raises(ValueError, match="same rank"):
        one_axis.lookup(np.array(1, dtype=np.int64))
    with pytest.raises(ValueError, match="shorthand is only supported"):
        multi_axis.lookup(np.array([1, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="shorthand is only supported"):
        multi_axis.lookup(np.ones((2, 2), dtype=np.int64))
    with pytest.raises(ValueError, match="repetitions dimension"):
        multi_axis.lookup(np.zeros((3, 1, 1), dtype=np.int64))
    with pytest.raises(ValueError, match="dimensions must be 1 or match"):
        multi_axis.lookup(np.zeros((2, 2, 1), dtype=np.int64))


def test_lookup_rejects_omitted_repetitions_dimension() -> None:
    roll = Roll(np.arange(24, dtype=np.int64).reshape(2, 3, 4))

    with pytest.raises(ValueError, match="shorthand is only supported"):
        roll.lookup(np.zeros((3, 4), dtype=np.int64), axis=-1)


@pytest.mark.parametrize("axis", [0, -2])
def test_roll_and_event_reject_repetitions_axis(axis: int) -> None:
    roll = Roll(np.arange(6, dtype=np.int64).reshape(2, 3))
    event = Event(roll.values > 2)

    for value in (roll, event):
        with pytest.raises(ValueError, match="repetitions axis"):
            value.select(0, axis=axis)
        with pytest.raises(ValueError, match="repetitions axis"):
            value.lookup(np.zeros((2, 1), dtype=np.int64), axis=axis)


def test_pool_rejects_repetitions_and_dice_axes() -> None:
    shaped = Roller(repetitions=2, seed=42).pool(2, d=6, shape=3)
    bare = Roller(repetitions=2, seed=42).pool(2, d=6)

    for axis in (-1, 2):
        with pytest.raises(ValueError, match="Pool dice axis"):
            shaped.select(0, axis=axis)
        with pytest.raises(ValueError, match="Pool dice axis"):
            shaped.lookup(np.zeros((2, 1, 1), dtype=np.int64), axis=axis)

    with pytest.raises(ValueError, match="repetitions axis"):
        bare.select(0)
    with pytest.raises(ValueError, match="repetitions axis"):
        bare.lookup(np.zeros((2, 1), dtype=np.int64))


@pytest.mark.parametrize("axis", [3, -4])
def test_selection_and_lookup_reject_out_of_range_axes(axis: int) -> None:
    roll = Roll(np.arange(6, dtype=np.int64).reshape(2, 3))

    with pytest.raises(ValueError, match="out of bounds"):
        roll.select(0, axis=axis)
    with pytest.raises(ValueError, match="out of bounds"):
        roll.lookup(np.zeros((2, 1), dtype=np.int64), axis=axis)


def test_selection_and_lookup_handle_empty_source_axes() -> None:
    roll = Roll(np.empty((2, 0, 3), dtype=np.int64))

    assert roll.select(slice(None), axis=1).values.shape == (2, 0, 3)
    result = roll.lookup(np.empty((2, 0, 1), dtype=np.int64), axis=1)
    assert result.values.shape == (2, 0, 3)

    with pytest.raises(IndexError):
        roll.select(0, axis=1)
    with pytest.raises(IndexError):
        roll.lookup(np.zeros((2, 1, 1), dtype=np.int64), axis=1)


def test_add_axis_supports_leading_interior_trailing_and_negative_axes() -> None:
    roll = Roll(np.arange(24, dtype=np.int64).reshape(2, 3, 4))
    event = Event(roll.values % 2 == 0)

    for value in (roll, event):
        assert value.add_axis(axis=1).values.shape == (2, 1, 3, 4)
        assert value.add_axis(axis=2).values.shape == (2, 3, 1, 4)
        assert value.add_axis().values.shape == (2, 3, 4, 1)
        assert value.add_axis(axis=-2).values.shape == (2, 3, 1, 4)
    np.testing.assert_array_equal(
        roll.add_axis(axis=2).values[:, :, 0, :],
        roll.values,
    )


@pytest.mark.parametrize("axis", [0, -3])
def test_add_axis_rejects_positions_before_repetitions(axis: int) -> None:
    roll = Roll(np.arange(6, dtype=np.int64).reshape(2, 3))
    event = Event(roll.values > 2)

    for value in (roll, event):
        with pytest.raises(ValueError, match="before the repetitions axis"):
            value.add_axis(axis=axis)


@pytest.mark.parametrize("axis", [3, -4])
def test_add_axis_rejects_out_of_range_positions(axis: int) -> None:
    roll = Roll(np.arange(6, dtype=np.int64).reshape(2, 3))

    with pytest.raises(ValueError, match="out of bounds"):
        roll.add_axis(axis=axis)


@pytest.mark.parametrize("axis", [True, False])
def test_structural_operations_reject_boolean_axes(axis: bool) -> None:
    roll = Roll(np.arange(6, dtype=np.int64).reshape(2, 3))

    with pytest.raises(TypeError, match="integer, not bool"):
        roll.select(0, axis=axis)
    with pytest.raises(TypeError, match="integer, not bool"):
        roll.lookup(np.zeros((2, 1), dtype=np.int64), axis=axis)
    with pytest.raises(TypeError, match="integer, not bool"):
        roll.add_axis(axis=axis)
