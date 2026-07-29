import numpy as np
import pytest

from stochroll import Event, Pool, Roll, Roller, concatenate, stack


def test_stack_rolls_on_default_structural_axis() -> None:
    first = Roll(np.array([1, 2], dtype=np.int64))
    second = Roll(np.array([3, 4], dtype=np.int64))

    result = stack([first, second])

    assert isinstance(result, Roll)
    np.testing.assert_array_equal(result.values, [[1, 3], [2, 4]])


def test_stack_events_on_negative_output_axis() -> None:
    first = Event(np.array([[True, False], [False, True]], dtype=np.bool_))
    second = Event(np.array([[False, True], [True, False]], dtype=np.bool_))

    result = stack([first, second], axis=-1)

    assert isinstance(result, Event)
    np.testing.assert_array_equal(
        result.values,
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
        ],
    )


def test_stack_pools_preserves_dice_axis_and_metadata() -> None:
    roller = Roller(repetitions=2, seed=42)
    first = Pool(
        np.array([[1, 2], [3, 4]], dtype=np.int8),
        sides=6,
        roller=roller,
    )
    second = Pool(
        np.array([[5, 6], [2, 1]], dtype=np.int8),
        sides=6,
        roller=roller,
    )

    result = stack([first, second])

    assert isinstance(result, Pool)
    assert result.values.shape == (2, 2, 2)
    np.testing.assert_array_equal(
        result.values,
        [[[1, 2], [5, 6]], [[3, 4], [2, 1]]],
    )
    assert result.sides == 6
    assert result.roller is roller


def test_concatenate_rolls_on_structural_axis() -> None:
    first = Roll(np.array([[1], [2]], dtype=np.int16))
    second = Roll(np.array([[3, 4], [5, 6]], dtype=np.int16))

    result = concatenate([first, second], axis=1)

    assert isinstance(result, Roll)
    np.testing.assert_array_equal(result.values, [[1, 3, 4], [2, 5, 6]])


def test_concatenate_events_on_negative_axis() -> None:
    first = Event(np.array([[True], [False]], dtype=np.bool_))
    second = Event(np.array([[False, True], [True, False]], dtype=np.bool_))

    result = concatenate([first, second], axis=-1)

    assert isinstance(result, Event)
    np.testing.assert_array_equal(
        result.values,
        [[True, False, True], [False, True, False]],
    )


def test_concatenate_pools_preserves_metadata_and_dice_extent() -> None:
    roller = Roller(repetitions=2, seed=42)
    first = Pool(
        np.array([[[1, 2]], [[3, 4]]], dtype=np.int8),
        sides=6,
        roller=roller,
    )
    second = Pool(
        np.array(
            [[[5, 6], [2, 1]], [[4, 3], [6, 5]]],
            dtype=np.int8,
        ),
        sides=6,
        roller=roller,
    )

    result = concatenate([first, second])

    assert isinstance(result, Pool)
    assert result.values.shape == (2, 3, 2)
    np.testing.assert_array_equal(
        result.values,
        [
            [[1, 2], [5, 6], [2, 1]],
            [[3, 4], [4, 3], [6, 5]],
        ],
    )
    assert result.sides == 6
    assert result.roller is roller


def test_stack_uses_output_coordinates_for_positive_and_negative_axes() -> None:
    first = Roll(np.arange(12, dtype=np.int64).reshape(2, 2, 3))
    second = Roll(first.values + 20)

    positive = stack([first, second], axis=2)
    negative = stack([first, second], axis=-2)

    assert positive.values.shape == (2, 2, 2, 3)
    np.testing.assert_array_equal(positive.values, negative.values)
    np.testing.assert_array_equal(
        positive.values,
        np.stack([first.values, second.values], axis=2),
    )


def test_concatenate_uses_input_coordinates_for_positive_and_negative_axes() -> None:
    first = Roll(np.arange(12, dtype=np.int64).reshape(2, 2, 3))
    second = Roll(first.values + 20)

    positive = concatenate([first, second], axis=2)
    negative = concatenate([first, second], axis=-1)

    assert positive.values.shape == (2, 2, 6)
    np.testing.assert_array_equal(positive.values, negative.values)
    np.testing.assert_array_equal(
        positive.values,
        np.concatenate([first.values, second.values], axis=2),
    )


def test_assembly_supports_zero_sized_structural_dimensions() -> None:
    first = Roll(np.empty((2, 0, 3), dtype=np.int64))
    second = Roll(np.empty((2, 0, 3), dtype=np.int64))

    stacked = stack([first, second], axis=1)
    concatenated = concatenate([first, second], axis=1)

    assert stacked.values.shape == (2, 2, 0, 3)
    assert concatenated.values.shape == (2, 0, 3)


def test_assembly_preserves_numpy_dtype_promotion() -> None:
    integers = Roll(np.array([[1], [2]], dtype=np.int16))
    floats = Roll(np.array([[0.5], [1.5]], dtype=np.float32))

    stacked = stack([integers, floats])
    concatenated = concatenate([integers, floats])

    assert stacked.values.dtype == np.result_type(
        integers.values.dtype,
        floats.values.dtype,
    )
    assert concatenated.values.dtype == np.result_type(
        integers.values.dtype,
        floats.values.dtype,
    )


def test_assembly_does_not_mutate_inputs() -> None:
    first_values = np.array([[1, 2], [3, 4]], dtype=np.int64)
    second_values = np.array([[5, 6], [7, 8]], dtype=np.int64)
    first = Roll(first_values.copy())
    second = Roll(second_values.copy())

    stack([first, second])
    concatenate([first, second])

    np.testing.assert_array_equal(first.values, first_values)
    np.testing.assert_array_equal(second.values, second_values)


@pytest.mark.parametrize("operation", [stack, concatenate])
def test_assembly_rejects_empty_sequences(operation: object) -> None:
    with pytest.raises(ValueError, match="at least one wrapper"):
        operation([])  # type: ignore[operator]


@pytest.mark.parametrize("operation", [stack, concatenate])
def test_assembly_rejects_mixed_wrapper_types(operation: object) -> None:
    roll = Roll(np.ones((2, 1), dtype=np.int64))
    event = Event(np.ones((2, 1), dtype=np.bool_))

    with pytest.raises(TypeError, match="homogeneous wrapper types"):
        operation([roll, event])  # type: ignore[operator]


@pytest.mark.parametrize("operation", [stack, concatenate])
def test_assembly_rejects_non_wrapper_inputs(operation: object) -> None:
    with pytest.raises(TypeError, match="Roll, Event, or Pool"):
        operation([np.ones((2, 1))])  # type: ignore[operator]


@pytest.mark.parametrize("operation", [stack, concatenate])
def test_assembly_rejects_repetition_mismatches(operation: object) -> None:
    first = Roll(np.ones((2, 1), dtype=np.int64))
    second = Roll(np.ones((3, 1), dtype=np.int64))

    with pytest.raises(ValueError, match="matching repetitions"):
        operation([first, second])  # type: ignore[operator]


@pytest.mark.parametrize("operation", [stack, concatenate])
def test_assembly_rejects_rank_mismatches(operation: object) -> None:
    first = Roll(np.ones((2, 1), dtype=np.int64))
    second = Roll(np.ones((2, 1, 1), dtype=np.int64))

    with pytest.raises(ValueError, match="matching ranks"):
        operation([first, second])  # type: ignore[operator]


def test_stack_rejects_shape_mismatches_without_broadcasting() -> None:
    first = Roll(np.ones((2, 1), dtype=np.int64))
    second = Roll(np.ones((2, 2), dtype=np.int64))

    with pytest.raises(ValueError, match="matching shapes"):
        stack([first, second])


def test_concatenate_rejects_non_axis_shape_mismatches() -> None:
    first = Roll(np.ones((2, 1, 3), dtype=np.int64))
    second = Roll(np.ones((2, 2, 4), dtype=np.int64))

    with pytest.raises(ValueError, match="non-concatenated dimensions"):
        concatenate([first, second], axis=1)


@pytest.mark.parametrize("operation", [stack, concatenate])
def test_pool_assembly_rejects_side_mismatches(operation: object) -> None:
    roller = Roller(repetitions=2, seed=42)
    first = Pool(np.ones((2, 1, 2), dtype=np.int8), sides=6, roller=roller)
    second = Pool(np.ones((2, 1, 2), dtype=np.int8), sides=8, roller=roller)

    with pytest.raises(ValueError, match="matching sides"):
        operation([first, second])  # type: ignore[operator]


@pytest.mark.parametrize("operation", [stack, concatenate])
def test_pool_assembly_rejects_roller_identity_mismatches(
    operation: object,
) -> None:
    first = Pool(
        np.ones((2, 1, 2), dtype=np.int8),
        sides=6,
        roller=Roller(repetitions=2, seed=42),
    )
    second = Pool(
        np.ones((2, 1, 2), dtype=np.int8),
        sides=6,
        roller=Roller(repetitions=2, seed=42),
    )

    with pytest.raises(ValueError, match="same Roller"):
        operation([first, second])  # type: ignore[operator]


@pytest.mark.parametrize("operation", [stack, concatenate])
def test_pool_assembly_rejects_dice_extent_mismatches(
    operation: object,
) -> None:
    roller = Roller(repetitions=2, seed=42)
    first = Pool(np.ones((2, 1, 2), dtype=np.int8), sides=6, roller=roller)
    second = Pool(np.ones((2, 1, 3), dtype=np.int8), sides=6, roller=roller)

    with pytest.raises(ValueError, match="matching dice extents"):
        operation([first, second])  # type: ignore[operator]


@pytest.mark.parametrize("axis", [0, -3])
def test_stack_rejects_repetitions_axis(axis: int) -> None:
    values = [Roll(np.ones((2, 1), dtype=np.int64))]

    with pytest.raises(ValueError, match="repetitions axis"):
        stack(values, axis=axis)


@pytest.mark.parametrize("axis", [0, -2])
def test_concatenate_rejects_repetitions_axis(axis: int) -> None:
    values = [Roll(np.ones((2, 1), dtype=np.int64))]

    with pytest.raises(ValueError, match="repetitions axis"):
        concatenate(values, axis=axis)


@pytest.mark.parametrize("axis", [3, -4])
def test_stack_rejects_out_of_range_axes(axis: int) -> None:
    values = [Roll(np.ones((2, 1), dtype=np.int64))]

    with pytest.raises(ValueError, match="out of bounds"):
        stack(values, axis=axis)


@pytest.mark.parametrize("axis", [2, -3])
def test_concatenate_rejects_out_of_range_axes(axis: int) -> None:
    values = [Roll(np.ones((2, 1), dtype=np.int64))]

    with pytest.raises(ValueError, match="out of bounds"):
        concatenate(values, axis=axis)


def test_pool_stack_allows_only_axes_before_dice() -> None:
    roller = Roller(repetitions=2, seed=42)
    pool = Pool(np.ones((2, 3, 2), dtype=np.int8), sides=6, roller=roller)

    before_dice = stack([pool, pool], axis=-2)

    assert before_dice.values.shape == (2, 3, 2, 2)
    for axis in (-1, 3):
        with pytest.raises(ValueError, match="after the Pool dice axis"):
            stack([pool, pool], axis=axis)


def test_pool_concatenate_rejects_dice_axis_and_bare_pool() -> None:
    roller = Roller(repetitions=2, seed=42)
    shaped = Pool(np.ones((2, 3, 2), dtype=np.int8), sides=6, roller=roller)
    bare = Pool(np.ones((2, 2), dtype=np.int8), sides=6, roller=roller)

    for axis in (-1, 2):
        with pytest.raises(ValueError, match="Pool dice axis"):
            concatenate([shaped, shaped], axis=axis)
    with pytest.raises(ValueError, match="Pool dice axis"):
        concatenate([bare, bare])


@pytest.mark.parametrize("axis", [True, False])
def test_assembly_rejects_boolean_axes(axis: bool) -> None:
    roll = Roll(np.ones((2, 1), dtype=np.int64))

    with pytest.raises(TypeError, match="integer, not bool"):
        stack([roll, roll], axis=axis)
    with pytest.raises(TypeError, match="integer, not bool"):
        concatenate([roll, roll], axis=axis)
