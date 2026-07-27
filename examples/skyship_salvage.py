"""A skyship crew races a storm while repairing three salvage systems.

Each system check rolls 5d6, rerolls all 1s once, drops the lowest die, and
needs 14+. A storm damages one repaired system. Two surviving repairs recover
a relic; otherwise the crew salvages ordinary cargo.
"""

from stochroll import Roller, where

REPETITIONS = 250_000
SYSTEMS = 3


def main() -> None:
    roller = Roller(repetitions=REPETITIONS, seed=20260731)

    system_checks = (
        roller.pool(5, d=6, shape=SYSTEMS).reroll_once([1]).drop_lowest().sum()
    )
    repaired = (system_checks >= 14).count()

    storm = roller.d(12) <= 3
    systems_after_storm = where(storm, repaired - 1, repaired)
    mission_success = systems_after_storm >= 2

    relic_value = roller.d(20) + 20
    cargo_value = roller.d(6) * 5
    haul = where(mission_success, relic_value, cargo_value)

    expected_checks = system_checks.expected()

    print("\n" + "=" * 58)
    print("SKYSHIP SALVAGE")
    print("Repair three systems before a storm ruins one of them.")
    print("-" * 58)
    print(f"Simulated salvage runs:       {REPETITIONS:,}")
    print(
        "Expected system checks:       "
        + ", ".join(f"{value:.2f}" for value in expected_checks)
    )
    print(f"Expected systems repaired:    {repaired.expected():.2f}")
    print(f"Storm probability:             {storm.probability():.1%}")
    print(f"Expected systems after storm:  {systems_after_storm.expected():.2f}")
    print(f"Relic mission success:         {mission_success.probability():.1%}")
    print(f"Expected haul value:            {haul.expected():.2f}")
    print(f"Chance of a 35+ haul:          {haul.probability_at_least(35):.1%}")
    print("=" * 58)


if __name__ == "__main__":
    main()
