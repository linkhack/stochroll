"""Deterministic RNG call instrumentation for prototype evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class IntegerCall:
    low: int
    high: int | NDArray[np.integer[Any]] | None
    size: int | tuple[int, ...] | None
    dtype: np.dtype[np.generic]
    draws: int


class RecordingRNG:
    """Small Generator proxy that records integer draw shapes and counts."""

    def __init__(self, seed: int | None = None) -> None:
        self._generator = np.random.default_rng(seed)
        self.calls: list[IntegerCall] = []

    @property
    def draw_count(self) -> int:
        return sum(call.draws for call in self.calls)

    def integers(
        self,
        low: int,
        high: int | NDArray[np.integer[Any]] | None = None,
        size: int | tuple[int, ...] | None = None,
        dtype: type[np.integer[Any]] | np.dtype[np.integer[Any]] = np.int64,
        endpoint: bool = False,
    ) -> NDArray[np.integer[Any]]:
        draws = 1 if size is None else int(np.prod(size, dtype=np.intp))
        normalized_dtype = np.dtype(dtype)
        # Array-valued bounds are mutable, so retain a read-only snapshot of
        # the exact bounds observed by this call rather than the caller's view.
        recorded_high = high.copy() if isinstance(high, np.ndarray) else high
        if isinstance(recorded_high, np.ndarray):
            recorded_high.flags.writeable = False
        self.calls.append(
            IntegerCall(low, recorded_high, size, normalized_dtype, draws)
        )
        return self._generator.integers(
            low,
            high,
            size=size,
            dtype=dtype,
            endpoint=endpoint,
        )
