"""Prototype-local bounded-loop validation and failure type."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np


def validate_max_steps(max_steps: int | np.integer[Any]) -> int:
    if isinstance(max_steps, (bool, np.bool_)):
        raise TypeError("max_steps must be an integer, not bool")
    try:
        normalized = operator.index(max_steps)
    except TypeError:
        raise TypeError("max_steps must be an integer") from None
    if normalized < 1:
        raise ValueError("max_steps must be >= 1")
    return normalized
