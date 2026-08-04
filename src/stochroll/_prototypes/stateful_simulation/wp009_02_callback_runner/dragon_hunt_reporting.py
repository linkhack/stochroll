"""Five-field Dragon Hunt variant for adapter-width and reporting evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stochroll import Event, Roll

from .._shared import ActiveBatch, Roller, validate_max_steps
from .dragon_hunt import (
    DEFAULT_SEED,
    DRAGON_HP,
    HUNTERS,
    PLAYER_HP,
    ROUNDS,
    resolve_dragon_round,
)
from .runner import SimulationLimitExceeded, SimulationResult, run_simulation


@dataclass(frozen=True, slots=True)
class DragonHuntReportingState:
    dragon_hp: Roll
    player_hp: Roll
    player_damage_dealt: Roll
    player_hits: Roll
    player_damage_taken: Roll


class DragonHuntReportingAdapter:
    def take(
        self,
        state: DragonHuntReportingState,
        batch: ActiveBatch,
        /,
    ) -> DragonHuntReportingState:
        return DragonHuntReportingState(
            dragon_hp=batch.take(state.dragon_hp),
            player_hp=batch.take(state.player_hp),
            player_damage_dealt=batch.take(state.player_damage_dealt),
            player_hits=batch.take(state.player_hits),
            player_damage_taken=batch.take(state.player_damage_taken),
        )

    def merge(
        self,
        base: DragonHuntReportingState,
        update: DragonHuntReportingState,
        batch: ActiveBatch,
        /,
    ) -> DragonHuntReportingState:
        return DragonHuntReportingState(
            dragon_hp=batch.merge(base.dragon_hp, update.dragon_hp),
            player_hp=batch.merge(base.player_hp, update.player_hp),
            player_damage_dealt=batch.merge(
                base.player_damage_dealt, update.player_damage_dealt
            ),
            player_hits=batch.merge(base.player_hits, update.player_hits),
            player_damage_taken=batch.merge(
                base.player_damage_taken, update.player_damage_taken
            ),
        )


DRAGON_HUNT_REPORTING_ADAPTER = DragonHuntReportingAdapter()


def initial_dragon_hunt_reporting_state(
    repetitions: int,
) -> DragonHuntReportingState:
    structural_shape = (repetitions, HUNTERS)

    def zeros() -> Roll:
        return Roll(np.zeros(structural_shape, dtype=np.int32))

    return DragonHuntReportingState(
        dragon_hp=Roll(np.full(repetitions, DRAGON_HP, dtype=np.int32)),
        player_hp=Roll(np.full(structural_shape, PLAYER_HP, dtype=np.int32)),
        player_damage_dealt=zeros(),
        player_hits=zeros(),
        player_damage_taken=zeros(),
    )


def dragon_hunt_reporting_activity(
    state: DragonHuntReportingState,
    completed_steps: int,
) -> Event:
    if completed_steps >= ROUNDS:
        return Event(np.zeros(state.dragon_hp.values.shape[0], dtype=np.bool_))
    return (state.dragon_hp > 0) & ((state.player_hp > 0).count() > 0)


def dragon_hunt_reporting_transition(
    state: DragonHuntReportingState,
    batch: ActiveBatch,
    step: int,
) -> DragonHuntReportingState:
    del step
    dragon_hp, player_hp, reporting = resolve_dragon_round(
        state.dragon_hp,
        state.player_hp,
        batch,
        collect_reporting=True,
    )
    assert reporting is not None
    return DragonHuntReportingState(
        dragon_hp=dragon_hp,
        player_hp=player_hp,
        player_damage_dealt=state.player_damage_dealt + reporting.player_damage_dealt,
        player_hits=state.player_hits + reporting.player_hits,
        player_damage_taken=state.player_damage_taken + reporting.player_damage_taken,
    )


def run_dragon_hunt_reporting_callbacks(
    roller: Roller,
    initial_state: DragonHuntReportingState,
    *,
    max_steps: int = ROUNDS,
) -> SimulationResult[DragonHuntReportingState]:
    return run_simulation(
        roller,
        initial_state,
        adapter=DRAGON_HUNT_REPORTING_ADAPTER,
        is_active=dragon_hunt_reporting_activity,
        step=dragon_hunt_reporting_transition,
        max_steps=max_steps,
    )


def run_dragon_hunt_reporting_manual(
    roller: Roller,
    initial_state: DragonHuntReportingState,
    *,
    max_steps: int = ROUNDS,
) -> SimulationResult[DragonHuntReportingState]:
    limit = validate_max_steps(max_steps)
    state = initial_state
    active = dragon_hunt_reporting_activity(state, 0)
    termination = np.full(roller.repetitions, -1, dtype=np.int64)
    termination[~active.values] = 0

    for transition_index in range(limit):
        batch = roller.active_batch(active)
        if batch is None:
            return SimulationResult(state, transition_index, Roll(termination))
        compact = DRAGON_HUNT_REPORTING_ADAPTER.take(state, batch)
        update = dragon_hunt_reporting_transition(compact, batch, transition_index)
        state = DRAGON_HUNT_REPORTING_ADAPTER.merge(state, update, batch)
        next_active = dragon_hunt_reporting_activity(state, transition_index + 1)
        termination[active.values & ~next_active.values] = transition_index + 1
        active = next_active

    result = SimulationResult(state, limit, Roll(termination))
    if np.any(active.values):
        raise SimulationLimitExceeded(result, active)
    return result


def simulate_dragon_hunt_reporting_callbacks(
    repetitions: int,
    *,
    seed: int = DEFAULT_SEED,
) -> SimulationResult[DragonHuntReportingState]:
    roller = Roller(repetitions=repetitions, seed=seed)
    return run_dragon_hunt_reporting_callbacks(
        roller, initial_dragon_hunt_reporting_state(repetitions)
    )


def simulate_dragon_hunt_reporting_manual(
    repetitions: int,
    *,
    seed: int = DEFAULT_SEED,
) -> SimulationResult[DragonHuntReportingState]:
    roller = Roller(repetitions=repetitions, seed=seed)
    return run_dragon_hunt_reporting_manual(
        roller, initial_dragon_hunt_reporting_state(repetitions)
    )
