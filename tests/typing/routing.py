from typing import assert_type

import numpy as np

from stochroll import Event, Roll


def check_routing_types() -> None:
    roll = Roll(np.ones((2, 3), dtype=np.int64))
    event = Event(np.ones((2, 3), dtype=np.bool_))
    destinations = np.zeros((2, 3), dtype=np.int64)

    assert_type(roll.route_sum(destinations, size=2), Roll)
    assert_type(event.route_any(destinations, size=2), Event)
