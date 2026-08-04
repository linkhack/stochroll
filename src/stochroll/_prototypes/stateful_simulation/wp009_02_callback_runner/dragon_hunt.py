"""Caller-owned Dragon Hunt state, callbacks, adapter, and manual reference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stochroll import Event, Roll, concatenate, where

from .._shared import ActiveBatch, Roller, validate_max_steps
from .runner import SimulationLimitExceeded, SimulationResult, run_simulation

HUNTERS = 4
ROUNDS = 15
DRAGON_HP = 80
PLAYER_HP = 30
DRAGON_ATTACKS = 3
DEFAULT_SEED = 20260730


@dataclass(frozen=True, slots=True)
class DragonHuntState:
    dragon_hp: Roll
    player_hp: Roll


@dataclass(frozen=True, slots=True)
class DragonReportingDelta:
    player_damage_dealt: Roll
    player_hits: Roll
    player_damage_taken: Roll


class DragonHuntAdapter:
    def take(
        self,
        state: DragonHuntState,
        batch: ActiveBatch,
        /,
    ) -> DragonHuntState:
        return DragonHuntState(
            dragon_hp=batch.take(state.dragon_hp),
            player_hp=batch.take(state.player_hp),
        )

    def merge(
        self,
        base: DragonHuntState,
        update: DragonHuntState,
        batch: ActiveBatch,
        /,
    ) -> DragonHuntState:
        return DragonHuntState(
            dragon_hp=batch.merge(base.dragon_hp, update.dragon_hp),
            player_hp=batch.merge(base.player_hp, update.player_hp),
        )


DRAGON_HUNT_ADAPTER = DragonHuntAdapter()


def initial_dragon_hunt_state(repetitions: int) -> DragonHuntState:
    structural_shape = (repetitions, HUNTERS)
    return DragonHuntState(
        dragon_hp=Roll(np.full(repetitions, DRAGON_HP, dtype=np.int32)),
        player_hp=Roll(np.full(structural_shape, PLAYER_HP, dtype=np.int32)),
    )


def dragon_hunt_activity(state: DragonHuntState, completed_steps: int) -> Event:
    if completed_steps >= ROUNDS:
        return Event(np.zeros(state.dragon_hp.values.shape[0], dtype=np.bool_))
    return (state.dragon_hp > 0) & ((state.player_hp > 0).count() > 0)


def resolve_dragon_round(
    dragon_hp: Roll,
    player_hp: Roll,
    batch: ActiveBatch,
    *,
    collect_reporting: bool = False,
) -> tuple[Roll, Roll, DragonReportingDelta | None]:
    """Resolve one compact round, optionally collecting reporting deltas."""
    player_alive = player_hp > 0
    dragon_alive = dragon_hp > 0

    attacks = batch.d(20, shape=HUNTERS)
    critical = attacks == 20
    hits = critical | ((attacks + 6 >= 15) & ~(attacks == 1))
    normal_damage = batch.d(8, shape=HUNTERS) + 3
    critical_damage = normal_damage + batch.d(8, shape=HUNTERS)
    can_attack = player_alive & dragon_alive.broadcast_to(HUNTERS)
    damage = where(
        can_attack,
        where(critical, critical_damage, where(hits, normal_damage, 0)),
        0,
    )
    next_dragon_hp = where(dragon_alive, dragon_hp - damage.sum(), 0)
    next_dragon_alive = next_dragon_hp > 0

    targets = batch.sample_indices(player_alive, size=DRAGON_ATTACKS)
    claw_rolls = batch.d(20, shape=2)
    claw_hits = (claw_rolls + 6 >= 14) & ~(claw_rolls == 1)
    claw_damage = batch.d(6, shape=2) + 4
    bite_roll = batch.d(20)
    bite_hit = (bite_roll + 8 >= 14) & ~(bite_roll == 1)
    bite_damage = batch.pool(2, d=8).sum() + 6
    attack_hits = concatenate([claw_hits, bite_hit.add_axis()])
    attack_damage = concatenate([claw_damage, bite_damage.add_axis()])
    dealt = where(
        next_dragon_alive.broadcast_to(DRAGON_ATTACKS) & attack_hits,
        attack_damage,
        0,
    )
    incoming = dealt.route_sum(targets, size=HUNTERS)
    next_player_hp = where(player_alive, player_hp - incoming, 0)

    reporting = None
    if collect_reporting:
        reporting = DragonReportingDelta(
            player_damage_dealt=damage,
            player_hits=where(hits & can_attack, 1, 0),
            player_damage_taken=incoming,
        )
    return next_dragon_hp, next_player_hp, reporting


def dragon_hunt_transition(
    state: DragonHuntState,
    batch: ActiveBatch,
    step: int,
) -> DragonHuntState:
    del step
    dragon_hp, player_hp, _ = resolve_dragon_round(
        state.dragon_hp, state.player_hp, batch
    )
    return DragonHuntState(
        dragon_hp=dragon_hp,
        player_hp=player_hp,
    )


def run_dragon_hunt_callbacks(
    roller: Roller,
    initial_state: DragonHuntState,
    *,
    max_steps: int = ROUNDS,
) -> SimulationResult[DragonHuntState]:
    return run_simulation(
        roller,
        initial_state,
        adapter=DRAGON_HUNT_ADAPTER,
        is_active=dragon_hunt_activity,
        step=dragon_hunt_transition,
        max_steps=max_steps,
    )


def run_dragon_hunt_manual(
    roller: Roller,
    initial_state: DragonHuntState,
    *,
    max_steps: int = ROUNDS,
) -> SimulationResult[DragonHuntState]:
    """Equivalent caller-managed ActiveBatch loop for comparison."""
    limit = validate_max_steps(max_steps)
    state = initial_state
    active = dragon_hunt_activity(state, 0)
    termination = np.full(roller.repetitions, -1, dtype=np.int64)
    termination[~active.values] = 0

    for transition_index in range(limit):
        batch = roller.active_batch(active)
        if batch is None:
            return SimulationResult(state, transition_index, Roll(termination))
        compact = DRAGON_HUNT_ADAPTER.take(state, batch)
        update = dragon_hunt_transition(compact, batch, transition_index)
        state = DRAGON_HUNT_ADAPTER.merge(state, update, batch)
        next_active = dragon_hunt_activity(state, transition_index + 1)
        termination[active.values & ~next_active.values] = transition_index + 1
        active = next_active

    result = SimulationResult(state, limit, Roll(termination))
    if np.any(active.values):
        raise SimulationLimitExceeded(result, active)
    return result


def simulate_dragon_hunt_callbacks(
    repetitions: int,
    *,
    seed: int = DEFAULT_SEED,
) -> SimulationResult[DragonHuntState]:
    roller = Roller(repetitions=repetitions, seed=seed)
    return run_dragon_hunt_callbacks(roller, initial_dragon_hunt_state(repetitions))


def simulate_dragon_hunt_manual(
    repetitions: int,
    *,
    seed: int = DEFAULT_SEED,
) -> SimulationResult[DragonHuntState]:
    roller = Roller(repetitions=repetitions, seed=seed)
    return run_dragon_hunt_manual(roller, initial_dragon_hunt_state(repetitions))
