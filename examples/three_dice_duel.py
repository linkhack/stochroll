"""A simple high-roller game: two players compete on a 3d6 score."""

from stochroll import Roller

REPETITIONS = 250_000


def main() -> None:
    roller = Roller(repetitions=REPETITIONS, seed=20260728)

    red_score = roller.pool(3, d=6).sum()
    blue_score = roller.pool(3, d=6).sum()

    red_wins = red_score > blue_score
    blue_wins = blue_score > red_score
    ties = red_score == blue_score

    print("\n" + "=" * 58)
    print("THREE-DICE DUEL")
    print("Each player rolls 3d6; the higher total wins.")
    print("-" * 58)
    print(f"Simulated games:             {REPETITIONS:,}")
    print(f"Red expected score:          {red_score.expected():.2f}")
    print(f"Blue expected score:         {blue_score.expected():.2f}")
    print(f"Red win probability:          {red_wins.probability():.1%}")
    print(f"Blue win probability:         {blue_wins.probability():.1%}")
    print(f"Tie probability:              {ties.probability():.1%}")
    print(f"Chance of a score of 15+:    {red_score.probability_at_least(15):.1%}")
    print("=" * 58)


if __name__ == "__main__":
    main()
