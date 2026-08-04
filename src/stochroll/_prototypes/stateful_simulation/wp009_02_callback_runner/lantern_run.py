"""Caller-owned Lantern Run state, callbacks, adapter, and manual reference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stochroll import Event, Roll, where

from .._shared import ActiveBatch, Roller, validate_max_steps
from .runner import SimulationLimitExceeded, SimulationResult, run_simulation

ROOMS = 6
DEFAULT_SEED = 20260729


@dataclass(frozen=True, slots=True)
class LanternState:
    haul: Roll
    busted: Event


class LanternAdapter:
    def take(self, state: LanternState, batch: ActiveBatch, /) -> LanternState:
        return LanternState(
            haul=batch.take(state.haul),
            busted=batch.take(state.busted),
        )

    def merge(
        self,
        base: LanternState,
        update: LanternState,
        batch: ActiveBatch,
        /,
    ) -> LanternState:
        return LanternState(
            haul=batch.merge(base.haul, update.haul),
            busted=batch.merge(base.busted, update.busted),
        )


LANTERN_ADAPTER = LanternAdapter()


def initial_lantern_state(repetitions: int) -> LanternState:
    return LanternState(
        haul=Roll(np.zeros(repetitions, dtype=np.int32)),
        busted=Event(np.zeros(repetitions, dtype=np.bool_)),
    )


def lantern_activity(state: LanternState, completed_steps: int) -> Event:
    if completed_steps >= ROOMS:
        return Event(np.zeros(state.haul.values.shape[0], dtype=np.bool_))
    return ~state.busted


def room_gems(room: Roll) -> Roll:
    return where(room == 6, 7, where(room >= 5, 4, where(room >= 3, 2, 0)))


def lantern_transition(
    state: LanternState,
    batch: ActiveBatch,
    step: int,
) -> LanternState:
    del step
    room = batch.d(6)
    trap = room <= 2
    return LanternState(
        haul=where(trap, 0, state.haul + room_gems(room)),
        busted=state.busted | trap,
    )


def run_lantern_callbacks(
    roller: Roller,
    initial_state: LanternState,
    *,
    max_steps: int = ROOMS,
) -> SimulationResult[LanternState]:
    return run_simulation(
        roller,
        initial_state,
        adapter=LANTERN_ADAPTER,
        is_active=lantern_activity,
        step=lantern_transition,
        max_steps=max_steps,
    )


def run_lantern_manual(
    roller: Roller,
    initial_state: LanternState,
    *,
    max_steps: int = ROOMS,
) -> SimulationResult[LanternState]:
    """Equivalent caller-managed ActiveBatch loop for comparison."""
    limit = validate_max_steps(max_steps)
    state = initial_state
    active = lantern_activity(state, 0)
    termination = np.full(roller.repetitions, -1, dtype=np.int64)
    termination[~active.values] = 0

    for transition_index in range(limit):
        batch = roller.active_batch(active)
        if batch is None:
            return SimulationResult(state, transition_index, Roll(termination))
        compact = LANTERN_ADAPTER.take(state, batch)
        update = lantern_transition(compact, batch, transition_index)
        state = LANTERN_ADAPTER.merge(state, update, batch)
        next_active = lantern_activity(state, transition_index + 1)
        termination[active.values & ~next_active.values] = transition_index + 1
        active = next_active

    result = SimulationResult(state, limit, Roll(termination))
    if np.any(active.values):
        raise SimulationLimitExceeded(result, active)
    return result


def simulate_lantern_callbacks(
    repetitions: int,
    *,
    seed: int = DEFAULT_SEED,
) -> SimulationResult[LanternState]:
    roller = Roller(repetitions=repetitions, seed=seed)
    return run_lantern_callbacks(roller, initial_lantern_state(repetitions))


def simulate_lantern_manual(
    repetitions: int,
    *,
    seed: int = DEFAULT_SEED,
) -> SimulationResult[LanternState]:
    roller = Roller(repetitions=repetitions, seed=seed)
    return run_lantern_manual(roller, initial_lantern_state(repetitions))
