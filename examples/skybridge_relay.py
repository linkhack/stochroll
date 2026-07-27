"""A three-team relay that uses structural axes for teams and legs.

Every leg is a 4d6 check with the lowest die dropped. A team that clears all
legs earns a finish bonus, and the highest final total wins the relay.
"""

from typing import cast

import numpy as np
from numpy.typing import NDArray

from stochroll import Roller, where

REPETITIONS = 250_000
TEAMS = 3
LEGS = 4
FINISH_BONUS = 5


def main() -> None:
    roller = Roller(repetitions=REPETITIONS, seed=20260732)

    # One sample has shape (teams, legs), while the leading axis remains the
    # independent Monte Carlo repetitions axis.
    leg_scores = roller.pool(4, d=6, shape=(TEAMS, LEGS)).drop_lowest().sum()
    cleared_legs = (leg_scores >= 14).count(axis=2)
    team_totals = leg_scores.sum(axis=2) + where(
        cleared_legs == LEGS,
        FINISH_BONUS,
        0,
    )

    winning_score = team_totals.max(axis=1)
    team_wins = team_totals == winning_score.broadcast_to(TEAMS)
    winner_count = team_wins.count(axis=1)
    sole_winner = winner_count == 1
    sole_team_wins = team_wins & sole_winner.broadcast_to(TEAMS)

    expected_leg_scores = cast(NDArray[np.float64], leg_scores.expected())
    expected_cleared_legs = cast(NDArray[np.float64], cleared_legs.expected())
    expected_team_totals = cast(NDArray[np.float64], team_totals.expected())
    sole_win_probabilities = cast(NDArray[np.float64], sole_team_wins.probability())

    print("\n" + "=" * 72)
    print("SKYBRIDGE RELAY")
    print(
        f"{TEAMS} teams race across {LEGS} legs; all legs cleared earns a finish bonus."
    )
    print("-" * 72)
    print(f"Simulated relays:             {REPETITIONS:,}")
    print(f"Expected winning score:       {winning_score.expected():.2f}")
    print(f"Relay has a sole winner:       {sole_winner.probability():.1%}")
    print(f"Relay ends in a tie:           {(~sole_winner).probability():.1%}")
    print("\nTeam statistics")
    print("Team | Leg averages       | Legs cleared | Final total | Sole wins")
    print("-----+---------------------+--------------+-------------+-----------")
    for team in range(TEAMS):
        leg_averages = ", ".join(f"{score:.2f}" for score in expected_leg_scores[team])
        print(
            f"  {team + 1:>2} | {leg_averages:<19} |"
            f"     {expected_cleared_legs[team]:>5.2f}    |"
            f"    {expected_team_totals[team]:>7.2f}  |"
            f"   {sole_win_probabilities[team]:>6.1%}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
