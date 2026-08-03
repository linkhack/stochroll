from typing import assert_type

import numpy as np

from stochroll import Event, Roll
from stochroll._prototypes.stateful_simulation._shared import ActiveBatch, Roller
from stochroll._prototypes.stateful_simulation.wp012_event_sampling.dragon_hunt import (
    DragonHuntResult,
    dragon_hunt_event_masked,
)

roller = Roller(repetitions=2, seed=42)
maybe_batch = roller.active_batch(Event(np.ones(2, dtype=np.bool_)))
if maybe_batch is not None:
    batch = maybe_batch
    assert_type(batch, ActiveBatch)
    assert_type(
        batch.sample_indices(Event(np.ones((2, 4), dtype=np.bool_))),
        Roll,
    )
    assert_type(
        batch.sample_indices(
            Event(np.ones((2, 3, 4), dtype=np.bool_)),
            size=np.int16(3),
            axis=np.int8(-1),
        ),
        Roll,
    )

assert_type(dragon_hunt_event_masked(10), DragonHuntResult)
