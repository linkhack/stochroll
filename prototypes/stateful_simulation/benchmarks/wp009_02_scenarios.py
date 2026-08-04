from __future__ import annotations

from typing import ClassVar

try:
    from stochroll._prototypes.stateful_simulation.wp009_02_callback_runner import (
        dragon_hunt as dragon_scenario,
    )
    from stochroll._prototypes.stateful_simulation.wp009_02_callback_runner import (
        dragon_hunt_reporting as dragon_reporting_scenario,
    )
    from stochroll._prototypes.stateful_simulation.wp009_02_callback_runner import (
        lantern_run as lantern_scenario,
    )
except ModuleNotFoundError as error:
    if error.name != (
        "stochroll._prototypes.stateful_simulation.wp009_02_callback_runner"
    ):
        raise
    dragon_scenario = None  # type: ignore[assignment]
    dragon_reporting_scenario = None  # type: ignore[assignment]
    lantern_scenario = None  # type: ignore[assignment]


class CallbackRunnerScenarios:
    params: ClassVar[list[int]] = [2_000, 10_000, 100_000]
    param_names: ClassVar[list[str]] = ["repetitions"]

    def time_lantern_callback_runner(self, repetitions: int) -> None:
        assert lantern_scenario is not None
        lantern_scenario.simulate_lantern_callbacks(repetitions)

    def time_lantern_hand_written(self, repetitions: int) -> None:
        assert lantern_scenario is not None
        lantern_scenario.simulate_lantern_manual(repetitions)

    def time_dragon_callback_runner(self, repetitions: int) -> None:
        assert dragon_scenario is not None
        dragon_scenario.simulate_dragon_hunt_callbacks(repetitions)

    def time_dragon_hand_written(self, repetitions: int) -> None:
        assert dragon_scenario is not None
        dragon_scenario.simulate_dragon_hunt_manual(repetitions)

    def time_dragon_reporting_callback_runner(self, repetitions: int) -> None:
        assert dragon_reporting_scenario is not None
        dragon_reporting_scenario.simulate_dragon_hunt_reporting_callbacks(repetitions)

    def time_dragon_reporting_hand_written(self, repetitions: int) -> None:
        assert dragon_reporting_scenario is not None
        dragon_reporting_scenario.simulate_dragon_hunt_reporting_manual(repetitions)
