from typing import Any, assert_type

import numpy as np

from stochroll import Event, Roll
from stochroll._prototypes.stateful_simulation._shared import ActiveBatch, Roller
from stochroll._prototypes.stateful_simulation.wp009_02_callback_runner import (
    ActivityCallback,
    SimulationLimitExceeded,
    SimulationResult,
    StateAdapter,
    StepCallback,
    run_simulation,
)
from stochroll._prototypes.stateful_simulation.wp009_02_callback_runner import (
    lantern_run as lantern_scenario,
)

roller = Roller(repetitions=2, seed=42)
state = lantern_scenario.initial_lantern_state(2)

lantern_adapter: StateAdapter[lantern_scenario.LanternState] = (
    lantern_scenario.LANTERN_ADAPTER
)
activity_callback: ActivityCallback[lantern_scenario.LanternState] = (
    lantern_scenario.lantern_activity
)
step_callback: StepCallback[lantern_scenario.LanternState] = (
    lantern_scenario.lantern_transition
)
assert_type(lantern_adapter, StateAdapter[lantern_scenario.LanternState])
assert_type(
    activity_callback,
    ActivityCallback[lantern_scenario.LanternState],
)
assert_type(step_callback, StepCallback[lantern_scenario.LanternState])

result = lantern_scenario.run_lantern_callbacks(roller, state)
assert_type(result, SimulationResult[lantern_scenario.LanternState])
assert_type(result.state, lantern_scenario.LanternState)
assert_type(result.state.haul, Roll)
assert_type(result.state.busted, Event)
assert_type(result.termination_step, Roll)


class RollAdapter:
    def take(self, state: Roll, batch: ActiveBatch, /) -> Roll:
        return batch.take(state)

    def merge(self, base: Roll, update: Roll, batch: ActiveBatch, /) -> Roll:
        return batch.merge(base, update)


roll_result = run_simulation(
    Roller(repetitions=2, seed=1),
    Roll(np.ones(2, dtype=np.int8)),
    adapter=RollAdapter(),
    is_active=lambda value, _: value > 0,
    step=lambda value, _batch, _step: value - 1,
    max_steps=1,
)
assert_type(roll_result, SimulationResult[Roll])

try:
    lantern_scenario.run_lantern_callbacks(roller, state, max_steps=1)
except SimulationLimitExceeded as error:
    # Python exception matching cannot carry the caught StateT specialization.
    assert_type(error, SimulationLimitExceeded[Any])
    assert_type(error.result, SimulationResult[Any])
    assert_type(error.result.state, Any)
    assert_type(error.active, Event)
