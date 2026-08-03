from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from stochroll import Event, Pool, Roll
from stochroll._prototypes.stateful_simulation._shared import (
    ActiveBatch,
    Roller,
)

roller = Roller(repetitions=2, seed=42)
maybe_batch = roller.active_batch(Event(np.array([True, False])))
if maybe_batch is not None:
    batch = maybe_batch
    assert_type(batch, ActiveBatch)
    assert_type(batch.positions, NDArray[np.intp])
    assert_type(batch.take(Roll(np.ones(2, dtype=np.int16))), Roll)
    assert_type(batch.take(Event(np.ones(2, dtype=np.bool_))), Event)
    dense_pool = roller.pool(2, d=6)
    assert_type(batch.take(dense_pool), Pool)
    assert_type(batch.merge(Roll(np.ones(2)), batch.d(6)), Roll)
    assert_type(
        batch.merge(
            Event(np.ones(2, dtype=np.bool_)),
            Event(np.ones(1, dtype=np.bool_)),
        ),
        Event,
    )
    assert_type(batch.merge(dense_pool, batch.take(dense_pool)), Pool)
