"""A fixed-policy push-your-luck game played through six haunted rooms.

The explorer always enters the next room. A 1 or 2 triggers a trap and wipes
out the haul; 3-4 find two gems, 5 finds four, and 6 finds seven.
"""

from typing import Any

from stochroll import Roll, Roller, where

REPETITIONS = 250_000
ROOMS = 6


def room_gems(roll: Roll) -> Roll:
    return where(roll == 6, 7, where(roll >= 5, 4, where(roll >= 3, 2, 0)))


def calc(
    repetitions: int = REPETITIONS,
    *,
    seed: int = 20260729,
) -> dict[str, Any]:
    roller = Roller(repetitions=repetitions, seed=seed)

    first_roll = roller.d(6)
    first_trap = first_roll <= 2
    haul = where(first_trap, 0, room_gems(first_roll))
    busted = first_trap

    for _ in range(ROOMS - 1):
        room_roll = roller.d(6)
        trap = room_roll <= 2
        gems = room_gems(room_roll)

        # Once the explorer is trapped, later rooms cannot add to the haul.
        haul = haul + where(~busted, where(trap, 0, gems), 0)
        busted = busted | trap

    banked_haul = where(~busted, haul, 0)
    survived = ~busted

    return {
        "survived": survived.probability(),
        "busted": busted.probability(),
        "banked_haul": banked_haul.expected(),
        "banked_10_plus": banked_haul.probability_at_least(10),
    }


def print_results(results: dict[str, Any], repetitions: int = REPETITIONS) -> None:
    print("\n" + "=" * 58)
    print("LANTERN RUN")
    print("Keep entering rooms: a trap loses every gem collected so far.")
    print("-" * 58)
    print(f"Simulated runs:               {repetitions:,}")
    print(f"Rooms entered per run:        {ROOMS}")
    print(f"Survive every room:            {results['survived']:.1%}")
    print(f"Lose the haul to a trap:       {results['busted']:.1%}")
    print(f"Expected banked gems:          {results['banked_haul']:.2f}")
    print(f"Chance to bank 10+ gems:       {results['banked_10_plus']:.1%}")
    print("=" * 58)


def main() -> None:
    print_results(calc())


if __name__ == "__main__":
    main()
