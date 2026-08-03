import numpy as np

from stochroll import Event, Roll
from stochroll._prototypes.stateful_simulation._shared import Roller

roller = Roller(repetitions=2, seed=42)
maybe_batch = roller.active_batch(Event(np.ones(2, dtype=np.bool_)))
if maybe_batch is not None:
    maybe_batch.sample_indices(Roll(np.ones((2, 4))))
    maybe_batch.sample_indices(Event(np.ones((2, 4), dtype=np.bool_)), size=1.5)
    maybe_batch.sample_indices(Event(np.ones((2, 4), dtype=np.bool_)), axis="1")
