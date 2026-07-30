"""A skyship crew races a storm while repairing three salvage systems.

Each system check rolls 5d6, rerolls all 1s once, drops the lowest die, and
needs 14+. A storm damages one repaired system. Two surviving repairs recover
a relic; otherwise the crew salvages ordinary cargo.
"""

from typing import Any

from stochroll import Roller, where

REPETITIONS = 250_000
SYSTEMS = 3


def calc(
    repetitions: int = REPETITIONS,
    *,
    seed: int = 20260731,
) -> dict[str, Any]:
    roller = Roller(repetitions=repetitions, seed=seed)

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

    return {
        "expected_checks": expected_checks,
        "repaired": repaired.expected(),
        "storm": storm.probability(),
        "systems_after_storm": systems_after_storm.expected(),
        "mission_success": mission_success.probability(),
        "haul": haul.expected(),
        "haul_35_plus": haul.probability_at_least(35),
    }


def print_results(results: dict[str, Any], repetitions: int = REPETITIONS) -> None:
    print("\n" + "=" * 58)
    print("SKYSHIP SALVAGE")
    print("Repair three systems before a storm ruins one of them.")
    print("-" * 58)
    print(f"Simulated salvage runs:       {repetitions:,}")
    print(
        "Expected system checks:       "
        + ", ".join(f"{value:.2f}" for value in results["expected_checks"])
    )
    print(f"Expected systems repaired:    {results['repaired']:.2f}")
    print(f"Storm probability:             {results['storm']:.1%}")
    print(f"Expected systems after storm:  {results['systems_after_storm']:.2f}")
    print(f"Relic mission success:         {results['mission_success']:.1%}")
    print(f"Expected haul value:            {results['haul']:.2f}")
    print(f"Chance of a 35+ haul:          {results['haul_35_plus']:.1%}")
    print("=" * 58)


def main() -> None:
    print_results(calc())


if __name__ == "__main__":
    main()
