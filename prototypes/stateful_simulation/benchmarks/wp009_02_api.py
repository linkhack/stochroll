from __future__ import annotations

import numpy as np

from stochroll import Event
from stochroll._prototypes.stateful_simulation._shared import Roller


class CallbackRunnerApi:
    def setup(self) -> None:
        from stochroll._prototypes.stateful_simulation import (
            wp009_02_callback_runner as callback_runner_package,
        )

        lantern_scenario = callback_runner_package.lantern_run
        repetitions = 10_000
        self.roller = Roller(repetitions=repetitions, seed=20260804)
        self.initial_state = lantern_scenario.initial_lantern_state(repetitions)
        active = Event(np.arange(repetitions) % 2 == 0)
        batch = self.roller.active_batch(active)
        assert batch is not None
        self.batch = batch
        self.compact_state = lantern_scenario.LANTERN_ADAPTER.take(
            self.initial_state,
            batch,
        )
        self.terminal_state = lantern_scenario.LanternState(
            haul=self.initial_state.haul,
            busted=Event(np.ones(repetitions, dtype=np.bool_)),
        )
        self.lantern_scenario = lantern_scenario

    def time_adapter_take(self) -> None:
        self.lantern_scenario.LANTERN_ADAPTER.take(self.initial_state, self.batch)

    def time_adapter_merge(self) -> None:
        self.lantern_scenario.LANTERN_ADAPTER.merge(
            self.initial_state,
            self.compact_state,
            self.batch,
        )

    def time_immediate_termination(self) -> None:
        from stochroll._prototypes.stateful_simulation import (
            wp009_02_callback_runner as callback_runner_package,
        )

        callback_runner_package.runner.run_simulation(
            self.roller,
            self.terminal_state,
            adapter=self.lantern_scenario.LANTERN_ADAPTER,
            is_active=self.lantern_scenario.lantern_activity,
            step=self.lantern_scenario.lantern_transition,
            max_steps=self.lantern_scenario.ROOMS,
        )
