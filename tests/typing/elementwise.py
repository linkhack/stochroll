from typing import assert_type

import numpy as np

from stochroll import Event, Roll, where
from stochroll._typing import EventArray, NumericScalar, RollArray
from stochroll.core import _validated_operand_values


def check_elementwise_types() -> None:
    roll = Roll(np.array([1, 2], dtype=np.int64))
    event = Event(np.array([True, False], dtype=np.bool_))

    assert_type(
        _validated_operand_values(roll, roll),
        RollArray | NumericScalar,
    )
    assert_type(_validated_operand_values(event, event), EventArray)
    assert_type(
        _validated_operand_values(event, roll),
        RollArray | NumericScalar,
    )

    assert_type(roll + roll, Roll)
    assert_type(1 + roll, Roll)
    assert_type(roll - roll, Roll)
    assert_type(1 - roll, Roll)
    assert_type(roll * roll, Roll)
    assert_type(1 * roll, Roll)
    assert_type(roll == roll, Event)
    assert_type(roll != roll, Event)
    assert_type(roll < roll, Event)
    assert_type(roll <= roll, Event)
    assert_type(roll > roll, Event)
    assert_type(roll >= roll, Event)
    assert_type(event | event, Event)
    assert_type(event & event, Event)
    assert_type(where(event, roll, 0), Roll)
