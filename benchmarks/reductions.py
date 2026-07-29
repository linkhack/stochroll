"""Focused comparisons for StochRoll's custom optimized reductions.

These are implementation benchmarks, not public API regression benchmarks.
Each NumPy and StochRoll pair receives the same ``uint8`` input, final axis,
and output or accumulator dtype.
"""
from functools import lru_cache
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from stochroll import Pool, Roller
from stochroll._reductions import (
    _reduce_max_last_axis,
    _reduce_sum_last_axis,
)

FOCUSED_REPETITIONS = 250_000
SUM_LAST_AXIS_SIZES = (1, 4, 12, 40)
MINMAX_LAST_AXIS_SIZES = (1, 4, 12, 40)
FUSED_LAST_AXIS_SIZES = (2, 4, 12, 40)
POOL_SUM_BOUNDARY_CASES = (
    (21, 12),  # maximum sum 252: uint8
    (22, 12),  # maximum sum 264: uint16
)

@lru_cache(maxsize=None, typed=True)
def _pool_values(last_axis_size: int) -> NDArray[Any]:
    roller = Roller(repetitions=FOCUSED_REPETITIONS, seed=0)
    return roller.pool(last_axis_size, d=6).values


class SumLastAxis:
    """Compare custom sum paths with the equivalent NumPy reduction."""

    params: ClassVar = [SUM_LAST_AXIS_SIZES]
    param_names: ClassVar = ["last_axis_size"]

    def setup(self, last_axis_size: int) -> None:
        self.values = _pool_values(last_axis_size)
        self.dtype: np.dtype[Any] = np.dtype(np.min_scalar_type(6 * last_axis_size))

    def time_numpy(self, last_axis_size: int) -> None:
        np.sum(self.values, axis=-1, dtype=self.dtype)

    def time_stochroll(self, last_axis_size: int) -> None:
        _reduce_sum_last_axis(self.values, dtype=self.dtype)


class MaxLastAxis:
    """Compare custom maximum paths with the equivalent NumPy reduction."""

    params: ClassVar = [MINMAX_LAST_AXIS_SIZES]
    param_names: ClassVar = ["last_axis_size"]

    def setup(self, last_axis_size: int) -> None:
        self.values = _pool_values(last_axis_size)

    def time_numpy(self, last_axis_size: int) -> None:
        np.max(self.values, axis=-1)

    def time_stochroll(self, last_axis_size: int) -> None:
        _reduce_max_last_axis(self.values)


class DropLowestSum:
    """Compare the fused public operation with direct NumPy composition."""

    params: ClassVar = [FUSED_LAST_AXIS_SIZES]
    param_names: ClassVar = ["last_axis_size"]

    def setup(self, last_axis_size: int) -> None:
        roller = Roller(repetitions=FOCUSED_REPETITIONS, seed=0)
        self.pool: Pool = roller.pool(last_axis_size, d=6)
        self.values = self.pool.values
        self.dtype: np.dtype[Any] = np.dtype(
            np.min_scalar_type(6 * (last_axis_size - 1))
        )

    def time_numpy(self, last_axis_size: int) -> None:
        total = np.sum(self.values, axis=-1, dtype=self.dtype)
        minimum = np.min(self.values, axis=-1)
        np.subtract(
            total,
            minimum.astype(self.dtype, copy=False),
            out=total,
            casting="unsafe",
        )

    def time_stochroll(self, last_axis_size: int) -> None:
        self.pool.drop_lowest_sum()


class PoolSumDtypeBoundary:
    """Measure Pool.sum immediately around an output-dtype boundary."""

    params: ClassVar = [POOL_SUM_BOUNDARY_CASES]
    param_names: ClassVar = ["sides_and_dice"]

    def setup(self, sides_and_dice: tuple[int, int]) -> None:
        sides, dice = sides_and_dice
        roller = Roller(repetitions=FOCUSED_REPETITIONS, seed=0)
        self.pool = roller.pool(dice, d=sides, shape=6)

    def time_sum(self, sides_and_dice: tuple[int, int]) -> None:
        self.pool.sum()
