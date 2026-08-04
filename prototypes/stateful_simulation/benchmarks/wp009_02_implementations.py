from __future__ import annotations

from typing import ClassVar

import numpy as np

from stochroll import Event
from stochroll._prototypes.stateful_simulation._shared import Roller


class LanternOrchestrationImplementations:
    """Compare generic and hand-written orchestration at fixed active fractions."""

    params: ClassVar[list[float]] = [1.0, 0.75, 0.25, 0.01, 0.0]
    param_names: ClassVar[list[str]] = ["active_fraction"]

    def setup(self, active_fraction: float) -> None:
        from stochroll._prototypes.stateful_simulation import (
            wp009_02_callback_runner as callback_runner_package,
        )

        lantern_scenario = callback_runner_package.lantern_run
        repetitions = 10_000
        active_count = round(repetitions * active_fraction)
        initial = lantern_scenario.initial_lantern_state(repetitions)
        busted = np.ones(repetitions, dtype=np.bool_)
        busted[:active_count] = False
        self.initial_state = lantern_scenario.LanternState(
            haul=initial.haul,
            busted=Event(busted),
        )
        self.callback_roller = Roller(repetitions=repetitions, seed=20260804)
        self.manual_roller = Roller(repetitions=repetitions, seed=20260804)
        self.lantern_scenario = lantern_scenario

    def time_callback_runner(self, active_fraction: float) -> None:
        from stochroll._prototypes.stateful_simulation import (
            wp009_02_callback_runner as callback_runner_package,
        )

        callback_runner_package.runner.run_simulation(
            self.callback_roller,
            self.initial_state,
            adapter=self.lantern_scenario.LANTERN_ADAPTER,
            is_active=self.lantern_scenario.lantern_activity,
            step=self.lantern_scenario.lantern_transition,
            max_steps=self.lantern_scenario.ROOMS,
        )

    def time_hand_written_active_batch(self, active_fraction: float) -> None:
        self.lantern_scenario.run_lantern_manual(
            self.manual_roller,
            self.initial_state,
        )
