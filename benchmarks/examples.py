"""End-to-end benchmarks for the runnable examples.

The examples expose their simulation work through ``calc`` functions so ASV
can measure the workload without capturing or timing console output.
"""

import sys
from pathlib import Path
from typing import ClassVar

# ASV discovers benchmark modules with ``benchmarks/`` as the import root,
# while the runnable examples are sibling modules in the source checkout.
sys.path.insert(0, str(Path(__file__).parents[1]))
from examples.advanced_guidance import calc as calc_advanced_guidance
from examples.dragon_hunt import calc as calc_dragon_hunt
from examples.lantern_run import calc as calc_lantern_run
from examples.skybridge_relay import calc as calc_skybridge_relay
from examples.skyship_salvage import calc as calc_skyship_salvage
from examples.three_dice_duel import calc as calc_three_dice_duel

EXAMPLE_REPETITIONS = (1_000, 250_000)


class ExampleCalculations:
    """Measure complete example simulations at representative sizes."""

    params: ClassVar = [EXAMPLE_REPETITIONS]
    param_names: ClassVar = ["repetitions"]

    def time_advanced_guidance(self, repetitions: int) -> None:
        calc_advanced_guidance(repetitions)

    def time_dragon_hunt(self, repetitions: int) -> None:
        calc_dragon_hunt(repetitions)

    def time_lantern_run(self, repetitions: int) -> None:
        calc_lantern_run(repetitions)

    def time_skybridge_relay(self, repetitions: int) -> None:
        calc_skybridge_relay(repetitions)

    def time_skyship_salvage(self, repetitions: int) -> None:
        calc_skyship_salvage(repetitions)

    def time_three_dice_duel(self, repetitions: int) -> None:
        calc_three_dice_duel(repetitions)
