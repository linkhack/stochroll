"""Complete ActiveBatch Dragon Hunt scenario using WP-012 targeting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stochroll import Event, Roll, concatenate, where

from .._shared import RecordingRNG, Roller, validate_max_steps
from .._shared.active_batch import ActiveBatch
from .dragon_targeting import DRAGON_ATTACKS, dragon_targets

HUNTERS = 4
ROUNDS = 15
DRAGON_HP = 80
PLAYER_HP = 30


@dataclass(frozen=True, slots=True)
class DragonHuntResult:
    dragon_hp: Roll
    player_hp: Roll
    termination_step: Roll
    transitions: int
    draws: int


def _recording_roller(
    repetitions: int,
    seed: int,
) -> tuple[Roller, RecordingRNG]:
    roller = Roller(repetitions=repetitions, seed=seed)
    recording = RecordingRNG(seed)
    roller.rng = recording  # type: ignore[assignment]
    return roller, recording


def _dragon_transition(
    batch: ActiveBatch,
    dragon_hp: Roll,
    player_hp: Roll,
) -> tuple[Roll, Roll]:
    player_alive = player_hp > 0
    dragon_alive = dragon_hp > 0

    # Resolve every living hunter's attack before the dragon phase. The dragon
    # still makes its phase draws when killed, but the damage mask below makes
    # those attacks ineffective, matching the simultaneous-phase baseline.
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

    # Take all three targets from the same phase-start alive mask. Sampling is
    # with replacement, so several attacks may intentionally hit one hunter.
    targets = dragon_targets(batch, player_hp)
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
    # Routing restores the fixed player axis and sums collisions when multiple
    # dragon attacks selected the same target.
    incoming = dealt.route_sum(targets, size=HUNTERS)
    next_player_hp = where(player_alive, player_hp - incoming, 0)
    return next_dragon_hp, next_player_hp


def dragon_hunt_event_masked(
    repetitions: int,
    *,
    seed: int = 20260730,
    max_steps: int = ROUNDS,
) -> DragonHuntResult:
    """Run the complete bounded scenario with living-player target sampling."""
    limit = min(validate_max_steps(max_steps), ROUNDS)
    roller, recording = _recording_roller(repetitions, seed)
    dragon_hp = Roll(np.full(repetitions, DRAGON_HP, dtype=np.int32))
    player_hp = Roll(np.full((repetitions, HUNTERS), PLAYER_HP, dtype=np.int32))
    termination = Roll(np.zeros(repetitions, dtype=np.int32))
    active = Event(np.ones(repetitions, dtype=np.bool_))
    transitions = 0

    for step in range(limit):
        # Activity excludes battles with a dead dragon or no living players.
        # Consequently every compact player mask passed to sampling is nonempty.
        batch = roller.active_batch(active)
        if batch is None:
            break
        transitions += batch.repetitions
        next_dragon, next_players = _dragon_transition(
            batch,
            batch.take(dragon_hp),
            batch.take(player_hp),
        )
        continuing = (next_dragon > 0) & ((next_players > 0).count() > 0)
        termination_update = where(
            continuing,
            batch.take(termination),
            step + 1,
        )
        # Merge compact updates into dense identity order. Finished battles are
        # absent from later batches, so their terminal values stay unchanged.
        dragon_hp = batch.merge(dragon_hp, next_dragon)
        player_hp = batch.merge(player_hp, next_players)
        termination = batch.merge(termination, termination_update)
        active = batch.merge(active, continuing)

    # Repetitions still active at the explicit domain horizon terminate there;
    # earlier terminal repetitions already carry the step recorded on merge.
    termination = where(active, limit, termination)
    return DragonHuntResult(
        dragon_hp=dragon_hp,
        player_hp=player_hp,
        termination_step=termination,
        transitions=transitions,
        draws=recording.draw_count,
    )
