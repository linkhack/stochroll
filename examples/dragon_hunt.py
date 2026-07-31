"""A small vectorized RPG combat: four hunters face a dragon across many rounds."""

from typing import Any

from stochroll import Roller, concatenate, where

REPETITIONS = 250_000
DRAGON_HP = 80
PLAYER_HP = 30
HUNTERS = 4
ROUNDS = 15
PLAYER_ARMOR_CLASS = 14
CLAW_BONUS = 6
BITE_BONUS = 8
CLAWS = 2
DRAGON_ATTACKS = CLAWS + 1


def calc_old(
    repetitions: int = REPETITIONS,
    *,
    seed: int = 20260730,
) -> dict[str, Any]:
    """Run the reference version that routes each dragon attack separately."""
    roller = Roller(repetitions=repetitions, seed=seed)

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
        # a zero-based random target; routed collisions are summed per player.
        claw_targets = [roller.d(HUNTERS) - 1 for _ in range(CLAWS)]
        claw_rolls = [roller.d(20) for _ in range(CLAWS)]
        claw_hits = [
            (roll + CLAW_BONUS >= PLAYER_ARMOR_CLASS) & ~(roll == 1)
            for roll in claw_rolls
        ]
        claw_damage = [roller.d(6) + 4 for _ in range(CLAWS)]

        bite_target = roller.d(HUNTERS) - 1
        bite_roll = roller.d(20)
        bite_hit = (bite_roll + BITE_BONUS >= PLAYER_ARMOR_CLASS) & ~(bite_roll == 1)
        bite_damage = roller.pool(2, d=8).sum() + 6

        incoming_damage: Any = 0
        for target, hit, damage in zip(
            claw_targets,
            claw_hits,
            claw_damage,
            strict=True,
        ):
            target_alive = player_alive.lookup(target).select(0)
            dealt = where(dragon_alive & hit & target_alive, damage, 0)
            incoming_damage = incoming_damage + dealt.route_sum(
                target,
                size=HUNTERS,
            )

        bite_target_alive = player_alive.lookup(bite_target).select(0)
        bite_dealt = where(
            dragon_alive & bite_hit & bite_target_alive,
            bite_damage,
            0,
        )
        incoming_damage = incoming_damage + bite_dealt.route_sum(
            bite_target,
            size=HUNTERS,
        )
        player_hp = where(player_alive, player_hp - incoming_damage, 0)
        player_damage_taken = player_damage_taken + incoming_damage

    dragon_defeated = ~dragon_alive
    player_alive = player_hp > 0
    surviving_players = player_alive.count()

    survival_probability = player_alive.probability()
    expected_hp = player_hp.expected()
    expected_damage_dealt = player_damage_dealt.expected()
    expected_hits = player_hits.expected()
    expected_damage_taken = player_damage_taken.expected()

    return {
        "dragon_defeated": dragon_defeated.probability(),
        "damage_to_dragon": (DRAGON_HP - dragon_hp).expected(),
        "dragon_hp": dragon_hp.expected(),
        "surviving_players": surviving_players.expected(),
        "survival_probability": survival_probability,
        "expected_hp": expected_hp,
        "expected_damage_dealt": expected_damage_dealt,
        "expected_hits": expected_hits,
        "expected_damage_taken": expected_damage_taken,
    }


def calc(
    repetitions: int = REPETITIONS,
    *,
    seed: int = 20260730,
) -> dict[str, Any]:
    """Run the combat with all dragon attacks routed as one vectorized batch."""
    roller = Roller(repetitions=repetitions, seed=seed)

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

        # Resolve both claws and the bite along one attack axis. Lookup reads
        # whether each target is alive; route_sum sends all damage back to the
        # player axis and sums attacks that selected the same hunter.
        targets = roller.d(HUNTERS, shape=DRAGON_ATTACKS) - 1

        claw_rolls = roller.d(20, shape=CLAWS)
        claw_hits = (claw_rolls + CLAW_BONUS >= PLAYER_ARMOR_CLASS) & ~(claw_rolls == 1)
        claw_damage = roller.d(6, shape=CLAWS) + 4

        bite_roll = roller.d(20)
        bite_hit = (bite_roll + BITE_BONUS >= PLAYER_ARMOR_CLASS) & ~(bite_roll == 1)
        bite_damage = roller.pool(2, d=8).sum() + 6

        attack_hits = concatenate([claw_hits, bite_hit.add_axis()])
        attack_damage = concatenate([claw_damage, bite_damage.add_axis()])
        target_alive = player_alive.lookup(targets)
        dealt = where(
            dragon_alive.broadcast_to(DRAGON_ATTACKS) & attack_hits & target_alive,
            attack_damage,
            0,
        )
        incoming_damage = dealt.route_sum(targets, size=HUNTERS)

        player_hp = where(player_alive, player_hp - incoming_damage, 0)
        player_damage_taken = player_damage_taken + incoming_damage

    dragon_defeated = ~dragon_alive
    player_alive = player_hp > 0
    surviving_players = player_alive.count()

    survival_probability = player_alive.probability()
    expected_hp = player_hp.expected()
    expected_damage_dealt = player_damage_dealt.expected()
    expected_hits = player_hits.expected()
    expected_damage_taken = player_damage_taken.expected()

    return {
        "dragon_defeated": dragon_defeated.probability(),
        "damage_to_dragon": (DRAGON_HP - dragon_hp).expected(),
        "dragon_hp": dragon_hp.expected(),
        "surviving_players": surviving_players.expected(),
        "survival_probability": survival_probability,
        "expected_hp": expected_hp,
        "expected_damage_dealt": expected_damage_dealt,
        "expected_hits": expected_hits,
        "expected_damage_taken": expected_damage_taken,
    }


def print_results(results: dict[str, Any], repetitions: int = REPETITIONS) -> None:
    print("\n" + "=" * 72)
    print("DRAGON HUNT")
    print(
        f"Four 30-HP hunters attack for {ROUNDS} rounds; "
        "the dragon then uses 2 claws and 1 bite."
    )
    print("-" * 72)
    print(f"Simulated hunts:              {repetitions:,}")
    print(f"Dragon defeated:              {results['dragon_defeated']:.1%}")
    print(f"Expected damage to dragon:    {results['damage_to_dragon']:.2f}")
    print(f"Expected dragon HP remaining: {results['dragon_hp']:.2f}")
    print(f"Expected players surviving:   {results['surviving_players']:.2f}")
    print("\nPlayer statistics")
    print("Player | Survives | HP left | Damage dealt | Hits | Damage taken")
    print("-------+----------+---------+--------------+------+-------------")
    for player in range(HUNTERS):
        print(
            f"  {player + 1:>2}   |  {results['survival_probability'][player]:>6.1%}  |"
            f"  {results['expected_hp'][player]:>6.2f} |"
            f"     {results['expected_damage_dealt'][player]:>7.2f} |"
            f" {results['expected_hits'][player]:>4.2f} |"
            f"       {results['expected_damage_taken'][player]:>7.2f}"
        )
    print("=" * 72)


def main() -> None:
    print_results(calc())


if __name__ == "__main__":
    main()
