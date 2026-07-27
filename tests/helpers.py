import numpy as np


class FixedRNG:
    def integers(
        self,
        low: int,
        high: int,
        *,
        size: int,
        dtype: np.dtype,
    ) -> np.ndarray:
        assert (low, high) == (1, 7)
        return np.full(size, 6, dtype=dtype)
