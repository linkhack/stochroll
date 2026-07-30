"""A simple high-roller game: two players compete on a 3d6 score."""

from typing import Any

from stochroll import Roller

REPETITIONS = 250_000


def calc(
    repetitions: int = REPETITIONS,
    *,
    seed: int = 20260728,
) -> dict[str, Any]:
    roller = Roller(repetitions=repetitions, seed=seed)

    red_score = roller.pool(3, d=6).sum()
    blue_score = roller.pool(3, d=6).sum()

    red_wins = red_score > blue_score
    blue_wins = blue_score > red_score
    ties = red_score == blue_score

    return {
        "red_expected": red_score.expected(),
        "blue_expected": blue_score.expected(),
        "red_wins": red_wins.probability(),
        "blue_wins": blue_wins.probability(),
        "ties": ties.probability(),
        "red_15_plus": red_score.probability_at_least(15),
    }


def print_results(results: dict[str, Any], repetitions: int = REPETITIONS) -> None:
    print("\n" + "=" * 58)
    print("THREE-DICE DUEL")
    print("Each player rolls 3d6; the higher total wins.")
    print("-" * 58)
    print(f"Simulated games:             {repetitions:,}")
    print(f"Red expected score:          {results['red_expected']:.2f}")
    print(f"Blue expected score:         {results['blue_expected']:.2f}")
    print(f"Red win probability:          {results['red_wins']:.1%}")
    print(f"Blue win probability:         {results['blue_wins']:.1%}")
    print(f"Tie probability:              {results['ties']:.1%}")
    print(f"Chance of a score of 15+:    {results['red_15_plus']:.1%}")
    print("=" * 58)


def main() -> None:
    print_results(calc())


if __name__ == "__main__":
    main()
