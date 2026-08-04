from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from stochroll import Event, Pool, Roll
from stochroll._prototypes.stateful_simulation._shared import (
    ActiveBatch,
    RecordingRNG,
    Roller,
)
from stochroll._prototypes.stateful_simulation.wp009_02_callback_runner import (
    SimulationLimitExceeded,
    run_simulation,
)
from stochroll._prototypes.stateful_simulation.wp009_02_callback_runner import (
    dragon_hunt as dragon_scenario,
)
from stochroll._prototypes.stateful_simulation.wp009_02_callback_runner import (
    dragon_hunt_reporting as dragon_reporting_scenario,
)
from stochroll._prototypes.stateful_simulation.wp009_02_callback_runner import (
    lantern_run as lantern_scenario,
)


class RollAdapter:
    def take(self, state: Roll, batch: ActiveBatch, /) -> Roll:
        return batch.take(state)

    def merge(self, base: Roll, update: Roll, batch: ActiveBatch, /) -> Roll:
        return batch.merge(base, update)


ROLL_ADAPTER = RollAdapter()


def _recording_roller(
    repetitions: int,
    seed: int = 42,
) -> tuple[Roller, RecordingRNG]:
    roller = Roller(repetitions=repetitions, seed=seed)
    recording = RecordingRNG(seed)
    roller.rng = recording  # type: ignore[assignment]
    return roller, recording


def _assert_recorded_calls_equal(
    left: RecordingRNG,
    right: RecordingRNG,
) -> None:
    assert len(left.calls) == len(right.calls)
    for left_call, right_call in zip(left.calls, right.calls, strict=True):
        assert left_call.low == right_call.low
        assert left_call.size == right_call.size
        assert left_call.dtype == right_call.dtype
        if isinstance(left_call.high, np.ndarray):
            assert isinstance(right_call.high, np.ndarray)
            np.testing.assert_array_equal(left_call.high, right_call.high)
        else:
            assert left_call.high == right_call.high


def _assert_dragon_states_equal(
    left: dragon_scenario.DragonHuntState,
    right: dragon_scenario.DragonHuntState,
) -> None:
    for field in (
        "dragon_hp",
        "player_hp",
    ):
        np.testing.assert_array_equal(
            getattr(left, field).values,
            getattr(right, field).values,
        )


def test_runner_handles_mixed_termination_and_metadata() -> None:
    roller = Roller(repetitions=3, seed=1)
    initial = Roll(np.array([0, 1, 2], dtype=np.int16))

    result = run_simulation(
        roller,
        initial,
        adapter=ROLL_ADAPTER,
        is_active=lambda state, _: state > 0,
        step=lambda state, _batch, _step: state - 1,
        max_steps=2,
    )

    np.testing.assert_array_equal(result.state.values, [0, 0, 0])
    np.testing.assert_array_equal(result.termination_step.values, [0, 1, 2])
    assert result.termination_step.values.dtype == np.dtype(np.int64)
    assert result.steps == 2
    np.testing.assert_array_equal(initial.values, [0, 1, 2])


def test_immediate_termination_skips_adapter_transition_and_rng() -> None:
    roller, recording = _recording_roller(3)
    calls: list[str] = []

    class FailingAdapter(RollAdapter):
        def take(self, state: Roll, batch: ActiveBatch, /) -> Roll:
            calls.append("take")
            return super().take(state, batch)

    result = run_simulation(
        roller,
        Roll(np.zeros(3, dtype=np.int8)),
        adapter=FailingAdapter(),
        is_active=lambda _state, _step: Event(np.zeros(3, dtype=np.bool_)),
        step=lambda state, _batch, _step: state,
        max_steps=1,
    )

    assert result.steps == 0
    np.testing.assert_array_equal(result.termination_step.values, 0)
    assert calls == []
    assert recording.calls == []


@pytest.mark.parametrize("max_steps", [1, np.int16(2)])
def test_positive_python_and_numpy_maxima_are_accepted(max_steps: int) -> None:
    result = run_simulation(
        Roller(repetitions=1, seed=1),
        Roll(np.zeros(1, dtype=np.int8)),
        adapter=ROLL_ADAPTER,
        is_active=lambda _state, _step: Event(np.zeros(1, dtype=np.bool_)),
        step=lambda state, _batch, _step: state,
        max_steps=max_steps,
    )
    assert result.steps == 0


@pytest.mark.parametrize(
    ("max_steps", "error"),
    [
        (True, TypeError),
        (1.5, TypeError),
        ("1", TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_invalid_maxima_are_rejected(max_steps: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        run_simulation(
            Roller(repetitions=1, seed=1),
            Roll(np.zeros(1, dtype=np.int8)),
            adapter=ROLL_ADAPTER,
            is_active=lambda _state, _step: Event(np.zeros(1, dtype=np.bool_)),
            step=lambda state, _batch, _step: state,
            max_steps=max_steps,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "activity",
    [
        Roll(np.ones(2, dtype=np.int8)),
        Event(np.ones((2, 1), dtype=np.bool_)),
        Event(np.ones(3, dtype=np.bool_)),
    ],
)
def test_activity_requires_exact_scalar_event(activity: object) -> None:
    expected = TypeError if isinstance(activity, Roll) else ValueError
    with pytest.raises(expected):
        run_simulation(
            Roller(repetitions=2, seed=1),
            Roll(np.ones(2, dtype=np.int8)),
            adapter=ROLL_ADAPTER,
            is_active=lambda _state, _step: activity,  # type: ignore[arg-type,return-value]
            step=lambda state, _batch, _step: state,
            max_steps=1,
        )


def test_reactivation_is_rejected_before_another_transition() -> None:
    transitions = 0

    def activity(_state: Roll, completed_steps: int) -> Event:
        if completed_steps == 0:
            return Event(np.array([True, False]))
        return Event(np.array([True, True]))

    def transition(state: Roll, _batch: ActiveBatch, _step: int) -> Roll:
        nonlocal transitions
        transitions += 1
        return state

    with pytest.raises(ValueError, match="cannot reactivate"):
        run_simulation(
            Roller(repetitions=2, seed=1),
            Roll(np.ones(2, dtype=np.int8)),
            adapter=ROLL_ADAPTER,
            is_active=activity,
            step=transition,
            max_steps=2,
        )
    assert transitions == 1


def test_limit_failure_exposes_dense_partial_state_and_activity() -> None:
    with pytest.raises(SimulationLimitExceeded) as caught:
        run_simulation(
            Roller(repetitions=3, seed=1),
            Roll(np.array([1, 2, 3], dtype=np.int8)),
            adapter=ROLL_ADAPTER,
            is_active=lambda state, _: state > 0,
            step=lambda state, _batch, _step: state - 1,
            max_steps=2,
        )

    failure = caught.value
    np.testing.assert_array_equal(failure.result.state.values, [0, 0, 1])
    np.testing.assert_array_equal(failure.result.termination_step.values, [1, 2, -1])
    np.testing.assert_array_equal(failure.active.values, [False, False, True])
    assert failure.result.steps == 2


@dataclass(frozen=True, slots=True)
class StructuredState:
    values: Roll
    flags: Event


class StructuredAdapter:
    def take(self, state: StructuredState, batch: ActiveBatch, /) -> StructuredState:
        return StructuredState(
            values=batch.take(state.values),
            flags=batch.take(state.flags),
        )

    def merge(
        self,
        base: StructuredState,
        update: StructuredState,
        batch: ActiveBatch,
        /,
    ) -> StructuredState:
        return StructuredState(
            values=batch.merge(base.values, update.values),
            flags=batch.merge(base.flags, update.flags),
        )


def test_multidimensional_roll_and_event_fields_preserve_inactive_rows() -> None:
    initial = StructuredState(
        values=Roll(np.arange(18, dtype=np.int16).reshape(3, 2, 3)),
        flags=Event(np.arange(18).reshape(3, 2, 3) % 2 == 0),
    )
    original_values = initial.values.values.copy()
    original_flags = initial.flags.values.copy()

    result = run_simulation(
        Roller(repetitions=3, seed=1),
        initial,
        adapter=StructuredAdapter(),
        is_active=lambda _state, step: (
            Event(np.array([True, False, True]))
            if step == 0
            else Event(np.zeros(3, dtype=np.bool_))
        ),
        step=lambda state, _batch, _step: StructuredState(
            values=state.values + 1,
            flags=~state.flags,
        ),
        max_steps=1,
    )

    np.testing.assert_array_equal(result.state.values.values[1], original_values[1])
    np.testing.assert_array_equal(result.state.flags.values[1], original_flags[1])
    np.testing.assert_array_equal(
        result.state.values.values[[0, 2]], original_values[[0, 2]] + 1
    )
    np.testing.assert_array_equal(
        result.state.flags.values[[0, 2]], ~original_flags[[0, 2]]
    )
    np.testing.assert_array_equal(initial.values.values, original_values)
    np.testing.assert_array_equal(initial.flags.values, original_flags)


def test_adapter_order_batch_reuse_and_callback_exception_propagation() -> None:
    calls: list[tuple[str, ActiveBatch]] = []

    class RecordingAdapter(RollAdapter):
        def take(self, state: Roll, batch: ActiveBatch, /) -> Roll:
            calls.append(("take", batch))
            return super().take(state, batch)

        def merge(self, base: Roll, update: Roll, batch: ActiveBatch, /) -> Roll:
            calls.append(("merge", batch))
            return super().merge(base, update, batch)

    expected = RuntimeError("domain failure")

    def fail(_state: Roll, batch: ActiveBatch, _step: int) -> Roll:
        calls.append(("step", batch))
        raise expected

    with pytest.raises(RuntimeError) as caught:
        run_simulation(
            Roller(repetitions=2, seed=1),
            Roll(np.ones(2, dtype=np.int8)),
            adapter=RecordingAdapter(),
            is_active=lambda state, _: state > 0,
            step=fail,
            max_steps=1,
        )
    assert caught.value is expected
    assert [name for name, _ in calls] == ["take", "step"]
    assert calls[0][1] is calls[1][1]


@dataclass(frozen=True, slots=True)
class PoolState:
    dice: Pool
    total: Roll
    enabled: Event


class PoolStateAdapter:
    def take(self, state: PoolState, batch: ActiveBatch, /) -> PoolState:
        return PoolState(
            batch.take(state.dice),
            batch.take(state.total),
            batch.take(state.enabled),
        )

    def merge(
        self,
        base: PoolState,
        update: PoolState,
        batch: ActiveBatch,
        /,
    ) -> PoolState:
        return PoolState(
            batch.merge(base.dice, update.dice),
            batch.merge(base.total, update.total),
            batch.merge(base.enabled, update.enabled),
        )


def test_pool_state_draw_reduction_reroll_and_merge_preserve_inactive_rows() -> None:
    roller, recording = _recording_roller(3, seed=9)
    values = np.array([[1, 2, 3], [6, 6, 6], [1, 1, 5]], dtype=np.int8)
    initial = PoolState(
        dice=Pool(values.copy(), sides=6, roller=roller),
        total=Roll(np.zeros(3, dtype=np.int16)),
        enabled=Event(np.array([True, False, True])),
    )

    def transition(state: PoolState, batch: ActiveBatch, _step: int) -> PoolState:
        drawn = batch.pool(2, d=6)
        rerolled = state.dice.reroll_once(1)
        return PoolState(
            dice=rerolled,
            total=rerolled.sum() + drawn.sum(),
            enabled=state.enabled,
        )

    result = run_simulation(
        roller,
        initial,
        adapter=PoolStateAdapter(),
        is_active=lambda state, step: (
            state.enabled if step == 0 else Event(np.zeros(3, dtype=np.bool_))
        ),
        step=transition,
        max_steps=1,
    )

    assert result.state.dice.roller is roller
    np.testing.assert_array_equal(result.state.dice.values[1], values[1])
    np.testing.assert_array_equal(result.state.total.values[1], 0)
    np.testing.assert_array_equal(initial.dice.values, values)
    assert recording.draw_count == 4 + int(np.count_nonzero(values[[0, 2]] == 1))


def test_adapter_rejects_invalid_compact_pool_update() -> None:
    roller = Roller(repetitions=2, seed=1)
    initial = PoolState(
        dice=roller.pool(2, d=6),
        total=Roll(np.zeros(2, dtype=np.int16)),
        enabled=Event(np.ones(2, dtype=np.bool_)),
    )

    def bad_update(state: PoolState, _batch: ActiveBatch, _step: int) -> PoolState:
        wrong_pool = Pool(
            state.dice.values,
            sides=8,
            roller=state.dice.roller,
        )
        return replace(state, dice=wrong_pool)

    with pytest.raises(ValueError, match="sides"):
        run_simulation(
            roller,
            initial,
            adapter=PoolStateAdapter(),
            is_active=lambda state, _: state.enabled,
            step=bad_update,
            max_steps=1,
        )


def test_lantern_callbacks_match_manual_values_rng_and_input_preservation() -> None:
    callback_roller, callback_rng = _recording_roller(256, seed=12)
    manual_roller, manual_rng = _recording_roller(256, seed=12)
    callback_initial = lantern_scenario.initial_lantern_state(256)
    manual_initial = lantern_scenario.initial_lantern_state(256)

    callback = lantern_scenario.run_lantern_callbacks(
        callback_roller,
        callback_initial,
    )
    manual = lantern_scenario.run_lantern_manual(manual_roller, manual_initial)

    np.testing.assert_array_equal(callback.state.haul.values, manual.state.haul.values)
    np.testing.assert_array_equal(
        callback.state.busted.values,
        manual.state.busted.values,
    )
    np.testing.assert_array_equal(
        callback.termination_step.values,
        manual.termination_step.values,
    )
    assert callback.steps == manual.steps == lantern_scenario.ROOMS
    _assert_recorded_calls_equal(callback_rng, manual_rng)
    np.testing.assert_array_equal(callback_initial.haul.values, 0)
    np.testing.assert_array_equal(callback_initial.busted.values, False)


def test_dragon_callbacks_match_manual_values_rng_shapes_and_positions() -> None:
    callback_roller, callback_rng = _recording_roller(256, seed=17)
    manual_roller, manual_rng = _recording_roller(256, seed=17)
    callback_initial = dragon_scenario.initial_dragon_hunt_state(256)
    manual_initial = dragon_scenario.initial_dragon_hunt_state(256)

    callback = dragon_scenario.run_dragon_hunt_callbacks(
        callback_roller,
        callback_initial,
    )
    manual = dragon_scenario.run_dragon_hunt_manual(manual_roller, manual_initial)

    _assert_dragon_states_equal(callback.state, manual.state)
    np.testing.assert_array_equal(
        callback.termination_step.values,
        manual.termination_step.values,
    )
    assert callback.steps == manual.steps
    assert callback.state.player_hp.values.shape == (256, dragon_scenario.HUNTERS)
    assert callback.termination_step.values.shape == (256,)
    for completed_steps in range(callback.steps):
        callback_positions = np.flatnonzero(
            callback.termination_step.values > completed_steps
        )
        manual_positions = np.flatnonzero(
            manual.termination_step.values > completed_steps
        )
        np.testing.assert_array_equal(callback_positions, manual_positions)
    _assert_recorded_calls_equal(callback_rng, manual_rng)


def test_dragon_reporting_callbacks_match_manual_values_and_rng() -> None:
    callback_roller, callback_rng = _recording_roller(128, seed=19)
    manual_roller, manual_rng = _recording_roller(128, seed=19)
    callback_initial = dragon_reporting_scenario.initial_dragon_hunt_reporting_state(
        128
    )
    manual_initial = dragon_reporting_scenario.initial_dragon_hunt_reporting_state(128)

    callback = dragon_reporting_scenario.run_dragon_hunt_reporting_callbacks(
        callback_roller, callback_initial
    )
    manual = dragon_reporting_scenario.run_dragon_hunt_reporting_manual(
        manual_roller, manual_initial
    )

    for field in (
        "dragon_hp",
        "player_hp",
        "player_damage_dealt",
        "player_hits",
        "player_damage_taken",
    ):
        np.testing.assert_array_equal(
            getattr(callback.state, field).values,
            getattr(manual.state, field).values,
        )
    np.testing.assert_array_equal(
        callback.termination_step.values, manual.termination_step.values
    )
    assert callback.steps == manual.steps
    _assert_recorded_calls_equal(callback_rng, manual_rng)


def test_dragon_transition_samples_only_phase_start_living_players(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roller = Roller(repetitions=4, seed=21)
    batch = roller.active_batch(Event(np.ones(4, dtype=np.bool_)))
    assert batch is not None
    state = dragon_scenario.initial_dragon_hunt_state(4)
    hp = np.zeros((4, dragon_scenario.HUNTERS), dtype=np.int32)
    for row, alive_count in enumerate(range(1, 5)):
        hp[row, :alive_count] = dragon_scenario.PLAYER_HP
    state = replace(state, player_hp=Roll(hp))
    sampled: list[tuple[Event, Roll]] = []
    original = ActiveBatch.sample_indices

    def record_sample(
        self: ActiveBatch,
        eligible: Event,
        *,
        size: int = 1,
        axis: int = -1,
    ) -> Roll:
        result = original(self, eligible, size=size, axis=axis)
        sampled.append((eligible, result))
        return result

    monkeypatch.setattr(ActiveBatch, "sample_indices", record_sample)
    dragon_scenario.dragon_hunt_transition(state, batch, 0)

    assert len(sampled) == 1
    eligible, targets = sampled[0]
    assert targets.values.shape == (4, dragon_scenario.DRAGON_ATTACKS)
    assert np.all(eligible.lookup(targets).values)


def test_zero_player_battles_are_excluded_before_transition() -> None:
    roller = Roller(repetitions=2, seed=5)
    state = dragon_scenario.initial_dragon_hunt_state(2)
    state = replace(
        state,
        player_hp=Roll(
            np.array(
                [[0, 0, 0, 0], [30, 0, 0, 0]],
                dtype=np.int32,
            )
        ),
    )

    with pytest.raises(SimulationLimitExceeded) as caught:
        dragon_scenario.run_dragon_hunt_callbacks(
            roller,
            state,
            max_steps=1,
        )

    np.testing.assert_array_equal(caught.value.result.termination_step.values, [0, -1])
    np.testing.assert_array_equal(caught.value.active.values, [False, True])


def test_activity_callback_and_runner_bookkeeping_draw_nothing() -> None:
    roller, recording = _recording_roller(2)
    initial = Roll(np.array([0, 1], dtype=np.int8))
    activity_calls: list[int] = []

    def activity(state: Roll, completed_steps: int) -> Event:
        activity_calls.append(completed_steps)
        return state > 0

    result = run_simulation(
        roller,
        initial,
        adapter=ROLL_ADAPTER,
        is_active=activity,
        step=lambda state, _batch, _step: state - 1,
        max_steps=1,
    )

    assert result.steps == 1
    assert activity_calls == [0, 1]
    assert recording.calls == []
