from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from stochroll import Event, Roll


def check_typed_summaries() -> None:
    roll = Roll(np.array([1, 2], dtype=np.int64))
    event = Event(np.array([True, False], dtype=np.bool_))

    assert_type(roll.expected(), NDArray[np.float64])
    assert_type(roll.probability_at_least(2), NDArray[np.float64])
    assert_type(event.probability(), NDArray[np.float64])
