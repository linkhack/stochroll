from __future__ import annotations

import numpy as np
import pytest

from stochroll import Event, Pool, Roll
from stochroll._prototypes.stateful_simulation._shared import (
    RecordingRNG,
    Roller,
    validate_max_steps,
)
from stochroll._prototypes.stateful_simulation.wp009_01_active_batch.reference import (
    HUNTERS,
    ROOMS,
    ROUNDS,
    dragon_hunt_active,
    dragon_hunt_dense,
    dragon_hunt_numpy,
    lantern_run_active,
    lantern_run_dense,
)


def test_activity_validation_positions_and_empty_batch() -> None:
    roller = Roller(repetitions=5, seed=7)
    active = Event(np.array([False, True, False, True, True]))
    batch = roller.active_batch(active)
    assert batch is not None
    assert batch.repetitions == 3
    np.testing.assert_array_equal(batch.positions, [1, 3, 4])
    assert batch.positions.dtype == np.intp
    assert not batch.positions.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        batch.positions[0] = 0

    with pytest.raises(ValueError, match="exact shape"):
        roller.active_batch(Event(np.ones((5, 1), dtype=np.bool_)))
    with pytest.raises(TypeError, match="Event"):
        roller.active_batch(Roll(np.ones(5, dtype=np.int8)))  # type: ignore[arg-type]
    assert roller.active_batch(Event(np.zeros(5, dtype=np.bool_))) is None

    compact_roller = Roller(repetitions=5, seed=99)
    dense_roller = Roller(repetitions=5, seed=99)
    all_active = compact_roller.active_batch(Event(np.ones(5, dtype=np.bool_)))
    assert all_active is not None
    np.testing.assert_array_equal(
        all_active.d(20, shape=2).values,
        dense_roller.d(20, shape=2).values,
    )


def test_take_and_merge_preserve_wrappers_shapes_and_inactive_state() -> None:
    roller = Roller(repetitions=4, seed=8)
    batch = roller.active_batch(Event(np.array([False, True, False, True])))
    assert batch is not None
    base_roll = Roll(np.arange(24, dtype=np.int16).reshape(4, 2, 3))
    base_event = Event(np.array([False, True, False, True]))

    compact_roll = batch.take(base_roll)
    compact_event = batch.take(base_event)
    assert isinstance(compact_roll, Roll)
    assert isinstance(compact_event, Event)
    assert compact_roll.values.shape == (2, 2, 3)
    update = Roll(np.full((2, 2, 3), 1.5, dtype=np.float32))
    merged = batch.merge(base_roll, update)
    merged_event = batch.merge(
        base_event,
        Event(np.array([False, False], dtype=np.bool_)),
    )

    assert merged.values.dtype == np.result_type(np.int16, np.float32)
    np.testing.assert_array_equal(merged.values[[0, 2]], base_roll.values[[0, 2]])
    np.testing.assert_array_equal(merged.values[[1, 3]], 1.5)
    np.testing.assert_array_equal(merged_event.values, [False, False, False, False])
    np.testing.assert_array_equal(base_roll.values, np.arange(24).reshape(4, 2, 3))


def test_compact_draws_pool_operations_rerolls_and_merge_use_shared_rng() -> None:
    roller = Roller(repetitions=5, seed=9)
    recording = RecordingRNG(9)
    roller.rng = recording  # type: ignore[assignment]
    batch = roller.active_batch(Event(np.array([True, False, True, False, False])))
    assert batch is not None

    roll = batch.d(12, shape=(2, 3))
    pool = batch.pool(4, d=6, shape=2)
    assert roll.values.shape == (2, 2, 3)
    assert pool.values.shape == (2, 2, 4)
    assert recording.calls[0].size == (2, 2, 3)
    assert recording.calls[1].size == (2, 2, 4)
    assert pool.sum().values.shape == (2, 2)
    assert pool.min().values.shape == (2, 2)
    assert pool.max().values.shape == (2, 2)
    assert pool.keep_highest(2).values.shape == (2, 2, 2)
    assert pool.drop_lowest(1).values.shape == (2, 2, 3)

    before = recording.draw_count
    rerolled = pool.reroll_once([1])
    assert recording.draw_count - before == int(np.count_nonzero(pool.values == 1))
    base = Pool(
        np.ones((5, 2, 4), dtype=pool.values.dtype),
        sides=6,
        roller=roller,
    )
    merged = batch.merge(base, rerolled)
    assert merged.roller is roller
    np.testing.assert_array_equal(merged.values[[1, 3, 4]], 1)


def test_validation_rejects_invalid_merge_metadata_without_mutation() -> None:
    roller = Roller(repetitions=3, seed=10)
    batch = roller.active_batch(Event(np.array([True, False, True])))
    assert batch is not None
    base = Roll(np.arange(6, dtype=np.int16).reshape(3, 2))
    original = base.values.copy()

    with pytest.raises(ValueError, match="trailing shapes"):
        batch.merge(base, Roll(np.ones((2, 1), dtype=np.int16)))
    with pytest.raises(TypeError, match="same wrapper"):
        batch.merge(base, Event(np.ones((2, 2), dtype=np.bool_)))  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="3 repetitions"):
        batch.take(Roll(np.ones((2, 2), dtype=np.int16)))

    parent_pool = roller.pool(2, d=6)
    compact_pool = batch.take(parent_pool)
    wrong_sides = Pool(compact_pool.values, sides=8, roller=compact_pool.roller)
    wrong_dtype = Pool(
        compact_pool.values.astype(np.int16),
        sides=6,
        roller=compact_pool.roller,
    )
    with pytest.raises(ValueError, match="sides"):
        batch.merge(parent_pool, wrong_sides)
    with pytest.raises(ValueError, match="dtypes"):
        batch.merge(parent_pool, wrong_dtype)
    with pytest.raises(ValueError, match="parent Roller"):
        batch.take(Pool(parent_pool.values, sides=6, roller=Roller(repetitions=3)))
    np.testing.assert_array_equal(base.values, original)


def test_limit_validation_and_immediate_termination() -> None:
    assert validate_max_steps(np.int64(2)) == 2
    with pytest.raises(TypeError, match="bool"):
        validate_max_steps(True)
    with pytest.raises(TypeError, match="integer"):
        validate_max_steps(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 1"):
        validate_max_steps(0)

    empty = Event(np.zeros(4, dtype=np.bool_))
    result = lantern_run_active(4, initially_active=empty)
    assert result.draws == 0
    assert result.transitions == 0
    limited = lantern_run_active(8, seed=11, max_steps=1)
    np.testing.assert_array_equal(limited.termination_step.values, 1)


def test_lantern_reference_has_successful_horizon_and_fewer_packed_draws() -> None:
    dense = lantern_run_dense(128, seed=12, max_steps=ROOMS)
    packed = lantern_run_active(128, seed=12, max_steps=ROOMS)

    assert dense.haul.values.shape == packed.haul.values.shape == (128,)
    assert np.all(
        (packed.termination_step.values >= 1)
        & (packed.termination_step.values <= ROOMS)
    )
    assert dense.draws == 128 * ROOMS
    assert packed.draws < dense.draws
    assert packed.draws == packed.transitions
    assert np.all(packed.haul.values[packed.busted.values] == 0)


def test_dragon_reference_is_bounded_dense_and_active_only() -> None:
    dense = dragon_hunt_dense(64, seed=13, max_steps=ROUNDS)
    packed = dragon_hunt_active(64, seed=13, max_steps=ROUNDS)
    numpy_compacted = dragon_hunt_numpy(64, seed=13, max_steps=ROUNDS)

    assert dense.dragon_hp.values.shape == packed.dragon_hp.values.shape == (64,)
    assert (
        dense.player_hp.values.shape == packed.player_hp.values.shape == (64, HUNTERS)
    )
    assert np.all(
        (packed.termination_step.values >= 1)
        & (packed.termination_step.values <= ROUNDS)
    )
    assert packed.transitions <= dense.transitions
    assert packed.draws <= dense.draws
    np.testing.assert_array_equal(
        numpy_compacted.dragon_hp.values,
        packed.dragon_hp.values,
    )
    np.testing.assert_array_equal(
        numpy_compacted.player_hp.values,
        packed.player_hp.values,
    )
    np.testing.assert_array_equal(
        numpy_compacted.termination_step.values,
        packed.termination_step.values,
    )
    assert numpy_compacted.transitions == packed.transitions
    assert numpy_compacted.draws == packed.draws
    no_players_alive = ~(packed.player_hp.values > 0).any(axis=1)
    assert np.all(
        (packed.dragon_hp.values <= 0)
        | no_players_alive
        | (packed.termination_step.values == ROUNDS)
    )
