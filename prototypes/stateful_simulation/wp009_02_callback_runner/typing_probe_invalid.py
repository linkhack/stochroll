import numpy as np

from stochroll import Event, Roll
from stochroll._prototypes.stateful_simulation._shared import ActiveBatch, Roller
from stochroll._prototypes.stateful_simulation.wp009_02_callback_runner import (
    lantern_run as lantern_scenario,
)
from stochroll._prototypes.stateful_simulation.wp009_02_callback_runner import (
    run_simulation,
)


class RollAdapter:
    def take(self, state: Roll, batch: ActiveBatch, /) -> Roll:
        return batch.take(state)

    def merge(self, base: Roll, update: Roll, batch: ActiveBatch, /) -> Roll:
        return batch.merge(base, update)


roller = Roller(repetitions=2, seed=42)
lantern = lantern_scenario.initial_lantern_state(2)

run_simulation(
    roller,
    lantern,
    adapter=RollAdapter(),
    is_active=lambda state, _: state > 0,
    step=lambda state, _batch, _step: state - 1,
    max_steps=1,
)

run_simulation(
    roller,
    lantern,
    adapter=lantern_scenario.LANTERN_ADAPTER,
    is_active=lambda _state, _: Roll(np.ones(2, dtype=np.int8)),
    step=lambda state, _batch, _step: state,
    max_steps=1,
)

run_simulation(
    roller,
    lantern,
    adapter=lantern_scenario.LANTERN_ADAPTER,
    is_active=lambda _state, _: Event(np.ones(2, dtype=np.bool_)),
    step=lambda _state, _batch, _step: Roll(np.ones(2, dtype=np.int8)),
    max_steps=1,
)
