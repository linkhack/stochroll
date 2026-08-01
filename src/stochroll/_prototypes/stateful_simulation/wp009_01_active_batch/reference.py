"""Dense and ActiveBatch forms of the two motivating simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from stochroll import Event, Pool, Roll, concatenate, where
from stochroll._typing import ShapeLike

from .._shared import (
    RecordingRNG,
    Roller,
    SimulationLimitExceeded,
    validate_max_steps,
)

ROOMS = 6
HUNTERS = 4
ROUNDS = 15
DRAGON_HP = 80
PLAYER_HP = 30
DRAGON_ATTACKS = 3


@dataclass(frozen=True, slots=True)
class LanternState:
    haul: Roll
    busted: Event
    termination_step: Roll
    transitions: int
    draws: int


@dataclass(frozen=True, slots=True)
class DragonState:
    dragon_hp: Roll
    player_hp: Roll
    termination_step: Roll
    transitions: int
    draws: int


class _Drawer(Protocol):
    def d(self, sides: int, *, shape: ShapeLike | None = None) -> Roll: ...

    def pool(
        self,
        dice: int,
        *,
        d: int,
        shape: ShapeLike | None = None,
    ) -> Pool: ...


def _recording_roller(
    repetitions: int,
    seed: int,
) -> tuple[Roller, RecordingRNG]:
    roller = Roller(repetitions=repetitions, seed=seed)
    recording = RecordingRNG(seed)
    roller.rng = recording  # type: ignore[assignment]
    return roller, recording


def _zeros(repetitions: int, *, shape: tuple[int, ...] = ()) -> Roll:
    return Roll(np.zeros((repetitions, *shape), dtype=np.int32))


def _all_active(repetitions: int) -> Event:
    return Event(np.ones(repetitions, dtype=np.bool_))


def _room_gems(roll: Roll) -> Roll:
    return where(roll == 6, 7, where(roll >= 5, 4, where(roll >= 3, 2, 0)))


def lantern_run_active(
    repetitions: int,
    *,
    seed: int = 20260729,
    max_steps: int = ROOMS,
    initially_active: Event | None = None,
) -> LanternState:
    limit = validate_max_steps(max_steps)
    roller, recording = _recording_roller(repetitions, seed)
    haul = _zeros(repetitions)
    busted = Event(np.zeros(repetitions, dtype=np.bool_))
    termination = _zeros(repetitions)
    active = _all_active(repetitions) if initially_active is None else initially_active
    transitions = 0

    for step in range(limit):
        batch = roller.active_batch(active)
        if batch is None:
            break
        transitions += batch.repetitions

        compact_haul = batch.take(haul)
        compact_busted = batch.take(busted)
        compact_termination = batch.take(termination)
        room_roll = batch.d(6)
        trap = room_roll <= 2
        haul_update = where(trap, 0, compact_haul + _room_gems(room_roll))
        busted_update = compact_busted | trap
        continuing = (~busted_update) & Event(
            np.full(batch.repetitions, step + 1 < ROOMS, dtype=np.bool_)
        )
        termination_update = where(
            ~continuing,
            step + 1,
            compact_termination,
        )

        haul = batch.merge(haul, haul_update)
        busted = batch.merge(busted, busted_update)
        termination = batch.merge(termination, termination_update)
        active = Event(active.values.copy())
        active.values[batch.positions] = continuing.values
    else:
        remaining = int(np.count_nonzero(active.values))
        if remaining:
            raise SimulationLimitExceeded(limit, remaining)

    return LanternState(haul, busted, termination, transitions, recording.draw_count)


def lantern_run_dense(
    repetitions: int,
    *,
    seed: int = 20260729,
    max_steps: int = ROOMS,
) -> LanternState:
    limit = validate_max_steps(max_steps)
    roller, recording = _recording_roller(repetitions, seed)
    haul = _zeros(repetitions)
    busted = Event(np.zeros(repetitions, dtype=np.bool_))
    termination = _zeros(repetitions)
    active = _all_active(repetitions)
    transitions = 0

    for step in range(limit):
        room_roll = roller.d(6)
        trap = room_roll <= 2
        next_haul = where(trap, 0, haul + _room_gems(room_roll))
        next_busted = busted | trap
        continuing = (~next_busted) & Event(
            np.full(repetitions, step + 1 < ROOMS, dtype=np.bool_)
        )
        ended = active & ~continuing
        haul = where(active, next_haul, haul)
        busted = Event(np.where(active.values, next_busted.values, busted.values))
        termination = where(ended, step + 1, termination)
        transitions += repetitions
        active = active & continuing
        if not np.any(active.values):
            break
    else:
        remaining = int(np.count_nonzero(active.values))
        if remaining:
            raise SimulationLimitExceeded(limit, remaining)

    return LanternState(haul, busted, termination, transitions, recording.draw_count)


def _dragon_transition(
    draw: _Drawer,
    dragon_hp: Roll,
    player_hp: Roll,
) -> tuple[Roll, Roll]:
    player_alive = player_hp > 0
    dragon_alive = dragon_hp > 0

    attacks = draw.d(20, shape=HUNTERS)
    critical = attacks == 20
    hits = critical | ((attacks + 6 >= 15) & ~(attacks == 1))
    normal_damage = draw.d(8, shape=HUNTERS) + 3
    critical_damage = normal_damage + draw.d(8, shape=HUNTERS)
    can_attack = player_alive & dragon_alive.broadcast_to(HUNTERS)
    damage = where(
        can_attack,
        where(critical, critical_damage, where(hits, normal_damage, 0)),
        0,
    )
    next_dragon_hp = where(dragon_alive, dragon_hp - damage.sum(), 0)
    next_dragon_alive = next_dragon_hp > 0

    targets = draw.d(HUNTERS, shape=DRAGON_ATTACKS) - 1
    claw_rolls = draw.d(20, shape=2)
    claw_hits = (claw_rolls + 6 >= 14) & ~(claw_rolls == 1)
    claw_damage = draw.d(6, shape=2) + 4
    bite_roll = draw.d(20)
    bite_hit = (bite_roll + 8 >= 14) & ~(bite_roll == 1)
    bite_damage = draw.pool(2, d=8).sum() + 6
    attack_hits = concatenate([claw_hits, bite_hit.add_axis()])
    attack_damage = concatenate([claw_damage, bite_damage.add_axis()])
    target_alive = player_alive.lookup(targets)
    dealt = where(
        next_dragon_alive.broadcast_to(DRAGON_ATTACKS) & attack_hits & target_alive,
        attack_damage,
        0,
    )
    incoming = dealt.route_sum(targets, size=HUNTERS)
    next_player_hp = where(player_alive, player_hp - incoming, 0)
    return next_dragon_hp, next_player_hp


def dragon_hunt_active(
    repetitions: int,
    *,
    seed: int = 20260730,
    max_steps: int = ROUNDS,
) -> DragonState:
    limit = validate_max_steps(max_steps)
    roller, recording = _recording_roller(repetitions, seed)
    dragon_hp = Roll(np.full(repetitions, DRAGON_HP, dtype=np.int32))
    player_hp = Roll(np.full((repetitions, HUNTERS), PLAYER_HP, dtype=np.int32))
    termination = _zeros(repetitions)
    active = _all_active(repetitions)
    transitions = 0

    for step in range(limit):
        batch = roller.active_batch(active)
        if batch is None:
            break
        transitions += batch.repetitions
        compact_dragon = batch.take(dragon_hp)
        compact_players = batch.take(player_hp)
        next_dragon, next_players = _dragon_transition(
            batch,
            compact_dragon,
            compact_players,
        )
        continuing = (next_dragon > 0) & ((next_players > 0).count() > 0)
        continuing = continuing & Event(
            np.full(batch.repetitions, step + 1 < ROUNDS, dtype=np.bool_)
        )
        termination_update = where(
            ~continuing,
            step + 1,
            batch.take(termination),
        )
        dragon_hp = batch.merge(dragon_hp, next_dragon)
        player_hp = batch.merge(player_hp, next_players)
        termination = batch.merge(termination, termination_update)
        active = Event(active.values.copy())
        active.values[batch.positions] = continuing.values
    else:
        remaining = int(np.count_nonzero(active.values))
        if remaining:
            raise SimulationLimitExceeded(limit, remaining)

    return DragonState(
        dragon_hp,
        player_hp,
        termination,
        transitions,
        recording.draw_count,
    )


def dragon_hunt_dense(
    repetitions: int,
    *,
    seed: int = 20260730,
    max_steps: int = ROUNDS,
) -> DragonState:
    limit = validate_max_steps(max_steps)
    roller, recording = _recording_roller(repetitions, seed)
    dragon_hp = Roll(np.full(repetitions, DRAGON_HP, dtype=np.int32))
    player_hp = Roll(np.full((repetitions, HUNTERS), PLAYER_HP, dtype=np.int32))
    termination = _zeros(repetitions)
    active = _all_active(repetitions)
    transitions = 0

    for step in range(limit):
        next_dragon, next_players = _dragon_transition(roller, dragon_hp, player_hp)
        candidate = (next_dragon > 0) & ((next_players > 0).count() > 0)
        candidate = candidate & Event(
            np.full(repetitions, step + 1 < ROUNDS, dtype=np.bool_)
        )
        ended = active & ~candidate
        dragon_hp = where(active, next_dragon, dragon_hp)
        player_hp = where(active.broadcast_to(HUNTERS), next_players, player_hp)
        termination = where(ended, step + 1, termination)
        transitions += repetitions
        active = active & candidate
        if not np.any(active.values):
            break
    else:
        remaining = int(np.count_nonzero(active.values))
        if remaining:
            raise SimulationLimitExceeded(limit, remaining)

    return DragonState(
        dragon_hp,
        player_hp,
        termination,
        transitions,
        recording.draw_count,
    )


def dragon_hunt_numpy(
    repetitions: int,
    *,
    seed: int = 20260730,
    max_steps: int = ROUNDS,
) -> DragonState:
    limit = min(validate_max_steps(max_steps), ROUNDS)
    _, recording = _recording_roller(repetitions, seed)
    dragon_hp = Roll(np.full(repetitions, DRAGON_HP, dtype=np.int32))
    player_hp = Roll(np.full((repetitions, HUNTERS), PLAYER_HP, dtype=np.int32))

    active_positions = np.arange(repetitions, dtype=np.intp)
    final_dragon = np.empty_like(dragon_hp.values)
    final_players = np.empty_like(player_hp.values)
    termination = np.empty(repetitions, dtype=np.int64)

    active_dragon = dragon_hp
    active_players = player_hp
    transitions = 0

    for step in range(limit):
        transitions += active_positions.size
        draw = Roller(repetitions=active_positions.size, seed=seed)
        draw.rng = recording  # type: ignore[assignment]
        next_dragon, next_players = _dragon_transition(
            draw,
            active_dragon,
            active_players,
        )

        continuing = (
            (next_dragon > 0) & ((next_players > 0).count() > 0)
        ).values
        if step + 1 == limit:
            continuing[:] = False

        finished = ~continuing
        finished_positions = active_positions[finished]
        final_dragon[finished_positions] = next_dragon.values[finished]
        final_players[finished_positions] = next_players.values[finished]
        termination[finished_positions] = step + 1

        active_positions = active_positions[continuing]
        if active_positions.size == 0:
            break
        active_dragon = Roll(next_dragon.values[continuing])
        active_players = Roll(next_players.values[continuing])

    return DragonState(
        Roll(final_dragon),
        Roll(final_players),
        Roll(termination),
        transitions,
        recording.draw_count,
    )
