from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from stochroll import Event, Pool, Roll
from stochroll._prototypes.stateful_simulation._shared import RecordingRNG, Roller
from stochroll._prototypes.stateful_simulation.wp012_event_sampling import (
    dragon_targeting,
)
from stochroll._prototypes.stateful_simulation.wp012_event_sampling.dragon_hunt import (
    ROUNDS,
    dragon_hunt_event_masked,
)
from stochroll._prototypes.stateful_simulation.wp012_event_sampling.sampling import (
    prepare_sampling,
    sample_reference,
    sample_vectorized,
)


def _batch(
    repetitions: int,
    *,
    seed: int = 42,
) -> tuple[Any, RecordingRNG]:
    roller = Roller(repetitions=repetitions, seed=seed)
    recording = RecordingRNG(seed)
    roller.rng = recording  # type: ignore[assignment]
    batch = roller.active_batch(Event(np.ones(repetitions, dtype=np.bool_)))
    assert batch is not None
    return batch, recording


def test_one_axis_samples_one_or_many_valid_indices_with_replacement() -> None:
    batch, _ = _batch(3, seed=7)
    eligible = Event(
        np.array([[True, False, False, False], [False, True, False, True], [True] * 4])
    )

    one = batch.sample_indices(eligible)
    many = batch.sample_indices(eligible, size=8)

    assert one.values.shape == (3, 1)
    assert many.values.shape == (3, 8)
    assert many.values.dtype == np.dtype(np.intp)
    assert np.all(many.values >= 0)
    assert np.all(many.values < 4)
    assert np.all(eligible.lookup(many).values)
    np.testing.assert_array_equal(many.values[0], np.zeros(8, dtype=np.intp))


@pytest.mark.parametrize(
    ("shape", "axis", "size", "expected_shape"),
    [
        ((2, 3, 4), -1, 2, (2, 3, 2)),
        ((2, 3, 4), 1, 5, (2, 5, 4)),
        ((2, 2, 3, 4), 2, 2, (2, 2, 2, 4)),
    ],
)
def test_multidimensional_sampling_replaces_only_selected_axis(
    shape: tuple[int, ...],
    axis: int,
    size: int,
    expected_shape: tuple[int, ...],
) -> None:
    batch, _ = _batch(2)
    eligible = Event(np.ones(shape, dtype=np.bool_))

    result = batch.sample_indices(eligible, size=size, axis=axis)

    assert result.values.shape == expected_shape
    assert np.all(eligible.lookup(result, axis=axis).values)


def test_vectorized_and_reference_mapping_share_exact_ordinals() -> None:
    values = np.array(
        [
            [[False, True, False, True, True], [True, False, True, False, False]],
            [[True, False, False, False, True], [False, True, True, False, True]],
        ]
    )
    prepared = prepare_sampling(values, repetitions=2, size=3, axis=-1)
    ordinals = np.array(
        [[[0, 1, 2], [1, 0, 1]], [[1, 0, 1], [2, 1, 0]]],
        dtype=np.intp,
    )
    expected = np.array(
        [[[1, 3, 4], [2, 0, 2]], [[4, 0, 4], [4, 2, 1]]],
        dtype=np.intp,
    )

    np.testing.assert_array_equal(sample_vectorized(prepared, ordinals), expected)
    np.testing.assert_array_equal(sample_reference(prepared, ordinals), expected)


def test_every_eligible_index_has_equal_ordinal_multiplicity() -> None:
    values = np.array(
        [
            [False, True, False, False, False],
            [True, False, False, True, False],
            [False, True, True, False, True],
            [True, True, False, True, True],
        ]
    )
    counts = np.count_nonzero(values, axis=-1)
    prepared = prepare_sampling(values, repetitions=4, size=12, axis=-1)
    ordinals = np.arange(12, dtype=np.intp) % counts[:, None]
    expected = np.stack(
        [np.flatnonzero(row)[ordinals[index]] for index, row in enumerate(values)]
    )

    vectorized = sample_vectorized(prepared, ordinals)
    reference = sample_reference(prepared, ordinals)

    np.testing.assert_array_equal(vectorized, expected)
    np.testing.assert_array_equal(reference, expected)
    for index, row in enumerate(values):
        eligible = np.flatnonzero(row)
        frequencies = np.count_nonzero(vectorized[index, :, None] == eligible, axis=0)
        np.testing.assert_array_equal(frequencies, 12 // len(eligible))


def test_all_eligible_mapping_is_direct_and_returns_intp() -> None:
    values = np.ones((2, 3, 4), dtype=np.bool_)
    prepared = prepare_sampling(values, repetitions=2, size=2, axis=1)
    ordinals = np.array(
        [
            [[0, 2], [1, 1], [2, 0], [0, 2]],
            [[1, 0], [2, 2], [0, 1], [1, 0]],
        ],
        dtype=np.int16,
    )

    result = sample_vectorized(prepared, ordinals)

    assert prepared.all_eligible
    assert prepared.eligible_positions.size == 0
    assert result.dtype == np.dtype(np.intp)
    np.testing.assert_array_equal(result, np.moveaxis(ordinals, -1, 1))


def test_uniform_prefix_mapping_needs_no_compressed_positions() -> None:
    values = np.zeros((2, 3, 8), dtype=np.bool_)
    values[..., :3] = True
    prepared = prepare_sampling(values, repetitions=2, size=2, axis=-1)
    ordinals = np.array(
        [[[0, 2], [1, 0], [2, 1]], [[2, 0], [0, 1], [1, 2]]],
        dtype=np.intp,
    )

    result = sample_vectorized(prepared, ordinals)

    assert prepared.prefix_eligible
    assert not prepared.all_eligible
    assert prepared.offsets.size == 0
    assert prepared.eligible_positions.size == 0
    np.testing.assert_array_equal(result, ordinals)


def test_uniform_nonprefix_mapping_keeps_compressed_positions() -> None:
    values = np.array(
        [
            [True, False, True, False],
            [False, True, False, True],
        ]
    )
    prepared = prepare_sampling(values, repetitions=2, size=2, axis=-1)
    ordinals = np.array([[0, 1], [1, 0]], dtype=np.intp)

    result = sample_vectorized(prepared, ordinals)

    assert not prepared.prefix_eligible
    assert prepared.offsets.shape == (2,)
    np.testing.assert_array_equal(result, [[0, 2], [3, 1]])


def test_indices_compose_with_lookup_and_routing() -> None:
    batch, _ = _batch(2, seed=11)
    eligible = Event(np.array([[True, False, True, False], [False, True, True, True]]))
    targets = batch.sample_indices(eligible, size=3)
    values = Roll(np.array([[10, 20, 30, 40], [50, 60, 70, 80]]))
    attacks = Roll(np.array([[2, 3, 5], [7, 11, 13]]))
    hits = Event(np.array([[True, False, True], [False, True, True]]))

    assert np.all(eligible.lookup(targets).values)
    assert values.lookup(targets).values.shape == (2, 3)
    assert attacks.route_sum(targets, size=4).values.shape == (2, 4)
    assert hits.route_any(targets, size=4).values.shape == (2, 4)


def test_indices_compose_with_pool_lookup_after_explicit_dice_axis() -> None:
    batch, _ = _batch(2)
    targets = batch.sample_indices(Event(np.ones((2, 4), dtype=np.bool_)), size=3)
    roller = Roller(repetitions=2, seed=1)
    pool = Pool(
        np.arange(1, 17, dtype=np.int64).reshape(2, 4, 2),
        sides=20,
        roller=roller,
    )

    selected = pool.lookup(targets.add_axis(), axis=1)

    assert selected.values.shape == (2, 3, 2)
    np.testing.assert_array_equal(
        selected.values,
        np.take_along_axis(pool.values, targets.values[..., None], axis=1),
    )


@pytest.mark.parametrize("alive_count", [1, 2, 3, 4])
def test_dragon_targets_only_players_alive_at_phase_start(alive_count: int) -> None:
    batch, _ = _batch(2, seed=alive_count)
    hp = np.zeros((2, 4), dtype=np.int32)
    hp[0, :alive_count] = 10
    hp[1, 4 - alive_count :] = 10
    player_hp = Roll(hp)

    targets = dragon_targeting.dragon_targets(batch, player_hp)

    assert targets.values.shape == (2, 3)
    assert np.all((player_hp > 0).lookup(targets).values)


def test_dragon_targets_rejects_a_zero_player_row_before_drawing() -> None:
    batch, recording = _batch(2)
    player_hp = Roll(np.array([[10, 0, 0, 0], [0, 0, 0, 0]]))

    with pytest.raises(ValueError, match="every eligibility slice"):
        dragon_targeting.dragon_targets(batch, player_hp)
    assert recording.calls == []


def test_full_dragon_hunt_is_seeded_bounded_and_shape_stable() -> None:
    first = dragon_hunt_event_masked(256, seed=20260803)
    second = dragon_hunt_event_masked(256, seed=20260803)

    np.testing.assert_array_equal(first.dragon_hp.values, second.dragon_hp.values)
    np.testing.assert_array_equal(first.player_hp.values, second.player_hp.values)
    np.testing.assert_array_equal(
        first.termination_step.values,
        second.termination_step.values,
    )
    assert first.dragon_hp.values.shape == (256,)
    assert first.player_hp.values.shape == (256, 4)
    assert first.termination_step.values.shape == (256,)
    assert np.all(first.dragon_hp.values <= 80)
    assert np.all(first.player_hp.values <= 30)
    assert np.all(first.termination_step.values >= 1)
    assert np.all(first.termination_step.values <= ROUNDS)
    assert 0 < first.transitions <= 256 * ROUNDS
    assert first.draws > 0
    assert first.draws == second.draws


def test_full_dragon_hunt_respects_explicit_horizon() -> None:
    result = dragon_hunt_event_masked(32, seed=9, max_steps=1)

    assert result.transitions == 32
    np.testing.assert_array_equal(
        result.termination_step.values,
        np.ones(32, dtype=np.int32),
    )


@pytest.mark.parametrize("max_steps", [True, 1.5, 0, -1])
def test_full_dragon_hunt_validates_max_steps(max_steps: object) -> None:
    expected = TypeError if isinstance(max_steps, (bool, float)) else ValueError
    with pytest.raises(expected):
        dragon_hunt_event_masked(2, max_steps=max_steps)  # type: ignore[arg-type]


def test_empty_slice_rejected_before_rng_advancement() -> None:
    batch, recording = _batch(2)
    masks = (
        Event(np.array([[True, False], [False, False]])),
        Event(np.array([[[True, False], [False, False]], [[True, False]] * 2])),
    )

    for eligible in masks:
        with pytest.raises(ValueError, match="every eligibility slice"):
            batch.sample_indices(eligible)
    assert recording.calls == []


def test_zero_preserved_axis_returns_empty_without_drawing() -> None:
    batch, recording = _batch(2)

    result = batch.sample_indices(
        Event(np.empty((2, 0, 4), dtype=np.bool_)),
        size=3,
        axis=-1,
    )

    assert result.values.shape == (2, 0, 3)
    assert result.values.dtype == np.dtype(np.intp)
    assert recording.calls == []


def test_zero_selected_axis_rejected_when_slices_exist() -> None:
    batch, recording = _batch(2)
    with pytest.raises(ValueError, match="every eligibility slice"):
        batch.sample_indices(Event(np.empty((2, 0), dtype=np.bool_)))
    assert recording.calls == []


@pytest.mark.parametrize("size", [True, np.bool_(False), 1.5, "2"])
def test_invalid_size_types_fail_without_drawing(size: object) -> None:
    batch, recording = _batch(2)
    with pytest.raises(TypeError):
        batch.sample_indices(Event(np.ones((2, 3), dtype=np.bool_)), size=size)  # type: ignore[arg-type]
    assert recording.calls == []


@pytest.mark.parametrize("size", [0, -1])
def test_nonpositive_sizes_fail_without_drawing(size: int) -> None:
    batch, recording = _batch(2)
    with pytest.raises(ValueError, match="positive"):
        batch.sample_indices(Event(np.ones((2, 3), dtype=np.bool_)), size=size)
    assert recording.calls == []


def test_python_and_numpy_integer_sizes_are_accepted() -> None:
    batch, _ = _batch(2)
    eligible = Event(np.ones((2, 3), dtype=np.bool_))
    assert batch.sample_indices(eligible, size=2).values.shape == (2, 2)
    assert batch.sample_indices(eligible, size=np.int16(3)).values.shape == (2, 3)


@pytest.mark.parametrize("axis", [True, np.bool_(False), 1.5, "1"])
def test_invalid_axis_types_fail_without_drawing(axis: object) -> None:
    batch, recording = _batch(2)
    with pytest.raises(TypeError):
        batch.sample_indices(Event(np.ones((2, 3), dtype=np.bool_)), axis=axis)  # type: ignore[arg-type]
    assert recording.calls == []


@pytest.mark.parametrize("axis", [0, -2, 2, -3])
def test_repetition_and_out_of_range_axes_fail_without_drawing(axis: int) -> None:
    batch, recording = _batch(2)
    with pytest.raises(ValueError):
        batch.sample_indices(Event(np.ones((2, 3), dtype=np.bool_)), axis=axis)
    assert recording.calls == []


def test_wrong_kind_count_and_scalar_event_fail_without_drawing() -> None:
    batch, recording = _batch(2)
    with pytest.raises(TypeError, match="Event"):
        batch.sample_indices(Roll(np.ones((2, 3))))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2 repetitions"):
        batch.sample_indices(Event(np.ones((3, 3), dtype=np.bool_)))
    with pytest.raises(ValueError, match="repetitions axis"):
        batch.sample_indices(Event(np.ones(2, dtype=np.bool_)))
    assert recording.calls == []


def test_instrumented_call_records_draw_shape_counts_and_shared_stream() -> None:
    batch, recording = _batch(2, seed=99)
    eligible = Event(
        np.array(
            [
                [[True, False, True, False], [False, True, False, False]],
                [[True, True, True, True], [False, False, True, True]],
            ]
        )
    )
    before_positions = batch.positions.copy()
    before_eligible = eligible.values.copy()

    first = batch.sample_indices(eligible, size=3)
    second = batch.d(6)

    call = recording.calls[0]
    assert call.low == 0
    assert call.size == (2, 2, 3)
    assert call.dtype == np.dtype(np.intp)
    assert isinstance(call.high, np.ndarray)
    np.testing.assert_array_equal(call.high, [[[2], [1]], [[4], [2]]])
    assert recording.draw_count == 14
    assert first.values.shape == (2, 2, 3)
    assert second.values.shape == (2,)
    np.testing.assert_array_equal(batch.positions, before_positions)
    np.testing.assert_array_equal(eligible.values, before_eligible)
    assert not batch.positions.flags.writeable


def test_singleton_eligibility_is_deterministic_without_rng_draws() -> None:
    batch, recording = _batch(3, seed=77)
    eligible = Event(
        np.array(
            [
                [False, True, False, False],
                [False, False, False, True],
                [True, False, False, False],
            ]
        )
    )

    sampled = batch.sample_indices(eligible, size=5)

    np.testing.assert_array_equal(
        sampled.values,
        np.array([[1] * 5, [3] * 5, [0] * 5], dtype=np.intp),
    )
    assert recording.calls == []


def test_singleton_heavy_masks_draw_only_for_variable_slices() -> None:
    batch, recording = _batch(2, seed=91)
    eligible = Event(
        np.array(
            [
                [
                    [True, False, False, False],
                    [False, True, False, False],
                ],
                [
                    [False, False, True, False],
                    [True, False, False, True],
                ],
            ]
        )
    )

    sampled = batch.sample_indices(eligible, size=4)

    assert len(recording.calls) == 1
    call = recording.calls[0]
    assert call.size == (1, 4)
    np.testing.assert_array_equal(call.high, [[2]])
    assert recording.draw_count == 4
    assert sampled.values.shape == (2, 2, 4)
    np.testing.assert_array_equal(
        sampled.values.reshape(-1, 4)[:3],
        [[0] * 4, [1] * 4, [2] * 4],
    )
    assert np.all(eligible.lookup(sampled).values)


def test_seeded_stream_advancement_matches_direct_scalar_bound_draw() -> None:
    seed = 123
    batch, recording = _batch(2, seed=seed)
    eligible = Event(np.array([[True, False, True], [False, True, True]]))
    sampled = batch.sample_indices(eligible, size=3)
    following = batch.d(20)

    direct = np.random.default_rng(seed)
    ordinals = direct.integers(
        0,
        2,
        size=(2, 3),
        dtype=np.intp,
    )
    prepared = prepare_sampling(eligible.values, repetitions=2, size=3, axis=-1)
    expected = sample_vectorized(prepared, ordinals)
    expected_following = direct.integers(
        1,
        21,
        size=(2,),
        dtype=np.min_scalar_type(20),
    )

    np.testing.assert_array_equal(sampled.values, expected)
    np.testing.assert_array_equal(following.values, expected_following)
    assert recording.calls[0].high == 2
