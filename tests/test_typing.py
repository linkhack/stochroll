from typing import assert_type

import numpy as np

from stochroll import Event, Roll


def test_event_indicator_static_return_type() -> None:
    event = Event(np.array([True], dtype=np.bool_))

    assert_type(event.indicator(), Roll)
