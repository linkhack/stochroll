from typing import assert_type

import numpy as np

from stochroll import Event, Pool, Roll, Roller


def check_structural_select_lookup_types() -> None:
    roller = Roller(repetitions=2, seed=42)
    roll = Roll(np.arange(6, dtype=np.int64).reshape(2, 3))
    event = Event(roll.values > 2)
    pool = roller.pool(2, d=6, shape=3)
    indices = np.array([0, 1], dtype=np.int64)

    assert_type(roll.select(0), Roll)
    assert_type(event.select(slice(None)), Event)
    assert_type(pool.select(indices), Pool)
    assert_type(roll.lookup(indices), Roll)
    assert_type(event.lookup(Roll(indices)), Event)
    assert_type(pool.lookup(indices), Pool)
    assert_type(roll.add_axis(), Roll)
    assert_type(event.add_axis(axis=1), Event)
