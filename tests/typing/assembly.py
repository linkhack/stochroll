from typing import assert_type

import numpy as np

from stochroll import Event, Pool, Roll, Roller, concatenate, stack


def check_assembly_types() -> None:
    roller = Roller(repetitions=2, seed=42)
    roll = Roll(np.ones((2, 1), dtype=np.int64))
    event = Event(np.ones((2, 1), dtype=np.bool_))
    pool = Pool(np.ones((2, 1, 2), dtype=np.int8), sides=6, roller=roller)

    assert_type(stack([roll, roll]), Roll)
    assert_type(stack([event, event]), Event)
    assert_type(stack([pool, pool]), Pool)
    assert_type(concatenate([roll, roll]), Roll)
    assert_type(concatenate([event, event]), Event)
    assert_type(concatenate([pool, pool]), Pool)
