"""A small vectorized RPG combat: four hunters face a dragon across many rounds."""

from typing import cast

import numpy as np
from numpy.typing import NDArray

from stochroll import Roll, Roller, where

REPETITIONS = 250_000
DRAGON_HP = 80
PLAYER_HP = 30
HUNTERS = 4
ROUNDS = 15
PLAYER_ARMOR_CLASS = 14
CLAW_BONUS = 6
BITE_BONUS = 8


def main() -> None:
    roller = Roller(repetitions=REPETITIONS, seed=20260730)

    # The final structural axis is the player axis throughout the combat.
    player_numbers = Roll(
        np.broadcast_to(
            np.arange(1, HUNTERS + 1),
            (REPETITIONS, HUNTERS),
        )
    )
    dragon_hp = roller.d(1) + DRAGON_HP - 1
    player_hp = roller.d(1, shape=HUNTERS) + PLAYER_HP - 1

    player_damage_dealt = roller.d(1, shape=HUNTERS) - 1
    player_hits = roller.d(1, shape=HUNTERS) - 1
    player_damage_taken = roller.d(1, shape=HUNTERS) - 1
    dragon_alive = dragon_hp > 0

    for _ in range(ROUNDS):
        # All hunters attack in parallel. Dead hunters contribute zero damage.
        d20 = roller.d(20, shape=HUNTERS)
        critical = d20 == 20
        natural_one = d20 == 1
        hits = critical | ((d20 + 6 >= 15) & ~natural_one)
        normal_damage = roller.d(8, shape=HUNTERS) + 3
        critical_damage = normal_damage + roller.d(8, shape=HUNTERS)

        player_alive = player_hp > 0
        can_attack = player_alive & dragon_alive.broadcast_to(HUNTERS)
        damage = where(
            can_attack,
            where(critical, critical_damage, where(hits, normal_damage, 0)),
            0,
        )

        dragon_hp = where(dragon_alive, dragon_hp - damage.sum(), 0)
        player_damage_dealt = player_damage_dealt + damage
        player_hits = player_hits + where(hits & can_attack, 1, 0)
        dragon_alive = dragon_hp > 0

        # The surviving dragon makes two claws and one bite. Each attack picks
        # a random target, and the target matching uses the player axis above.
        claw_targets = [roller.d(HUNTERS) for _ in range(2)]
        claw_rolls = [roller.d(20) for _ in range(2)]
        claw_hits = [
            (roll + CLAW_BONUS >= PLAYER_ARMOR_CLASS) & ~(roll == 1)
            for roll in claw_rolls
        ]
        claw_damage = [roller.d(6) + 4 for _ in range(2)]

        bite_target = roller.d(HUNTERS)
        bite_roll = roller.d(20)
        bite_hit = (bite_roll + BITE_BONUS >= PLAYER_ARMOR_CLASS) & ~(bite_roll == 1)
        bite_damage = roller.pool(2, d=8).sum() + 6

        dragon_attacking = dragon_alive.broadcast_to(HUNTERS)
        incoming_damage = 0
        for target, hit, damage in zip(
            claw_targets,
            claw_hits,
            claw_damage,
            strict=True,
        ):
            incoming_damage = incoming_damage + where(
                dragon_attacking
                & player_alive
                & hit.broadcast_to(HUNTERS)
                & (target.broadcast_to(HUNTERS) == player_numbers),
                damage.broadcast_to(HUNTERS),
                0,
            )

        incoming_damage = incoming_damage + where(
            dragon_attacking
            & player_alive
            & bite_hit.broadcast_to(HUNTERS)
            & (bite_target.broadcast_to(HUNTERS) == player_numbers),
            bite_damage.broadcast_to(HUNTERS),
            0,
        )
        player_hp = where(player_alive, player_hp - incoming_damage, 0)
        player_damage_taken = player_damage_taken + incoming_damage

    dragon_defeated = ~dragon_alive
    player_alive = player_hp > 0
    surviving_players = player_alive.count()

    survival_probability = cast(NDArray[np.float64], player_alive.probability())
    expected_hp = cast(NDArray[np.float64], player_hp.expected())
    expected_damage_dealt = cast(NDArray[np.float64], player_damage_dealt.expected())
    expected_hits = cast(NDArray[np.float64], player_hits.expected())
    expected_damage_taken = cast(NDArray[np.float64], player_damage_taken.expected())

    print("\n" + "=" * 72)
    print("DRAGON HUNT")
    print(
        f"Four 30-HP hunters attack for {ROUNDS} rounds; "
        "the dragon then uses 2 claws and 1 bite."
    )
    print("-" * 72)
    print(f"Simulated hunts:              {REPETITIONS:,}")
    print(f"Dragon defeated:              {dragon_defeated.probability():.1%}")
    print(f"Expected damage to dragon:    {(DRAGON_HP - dragon_hp).expected():.2f}")
    print(f"Expected dragon HP remaining: {dragon_hp.expected():.2f}")
    print(f"Expected players surviving:   {surviving_players.expected():.2f}")
    print("\nPlayer statistics")
    print("Player | Survives | HP left | Damage dealt | Hits | Damage taken")
    print("-------+----------+---------+--------------+------+-------------")
    for player in range(HUNTERS):
        print(
            f"  {player + 1:>2}   |  {survival_probability[player]:>6.1%}  |"
            f"  {expected_hp[player]:>6.2f} |"
            f"     {expected_damage_dealt[player]:>7.2f} |"
            f" {expected_hits[player]:>4.2f} |"
            f"       {expected_damage_taken[player]:>7.2f}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
