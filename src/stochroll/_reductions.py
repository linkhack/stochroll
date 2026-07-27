from typing import Any, Protocol, overload

import numpy as np
from numpy.typing import NDArray

from ._typing import NumericDTypeScalar

# ============================================================
# Constants
# ============================================================

_MINMAX_SMALL_AXIS_THRESHOLDS = {
    1: 16,  # int8 / uint8
    2: 12,  # int16 / uint16
    4: 10,  # int32 / uint32 / float32
    8: 7,  # int64 / uint64 / float64
}

_SUM_SMALL_AXIS_THRESHOLDS = {
    1: 8,
    2: 6,
    4: 4,
    8: 3,
}


def _signed_dtype_for_unsigned(dtype: np.dtype) -> np.dtype:
    dtype = np.dtype(dtype)

    if dtype.itemsize <= 1:
        return np.dtype(np.int16)
    if dtype.itemsize <= 2:
        return np.dtype(np.int32)
    if dtype.itemsize <= 4:
        return np.dtype(np.int64)

    raise TypeError("uint64 Roll values cannot be safely converted to signed dtype")


@overload
def _default_sum_dtype(dtype: np.dtype[np.integer[Any]]) -> np.dtype[np.int64]: ...


@overload
def _default_sum_dtype(dtype: np.dtype[np.floating[Any]]) -> np.dtype[np.float64]: ...


@overload
def _default_sum_dtype(
    dtype: np.dtype[NumericDTypeScalar],
) -> np.dtype[np.int64] | np.dtype[np.float64]: ...


def _default_sum_dtype(
    dtype: np.dtype[Any],
) -> np.dtype[Any]:
    if np.issubdtype(dtype, np.integer):
        return np.dtype(np.int64)

    if np.issubdtype(dtype, np.floating):
        return np.dtype(np.float64)

    return dtype


def _reduce_sum_last_axis[T: NumericDTypeScalar](
    values: NDArray[T],
    *,
    dtype: np.dtype[T],
) -> NDArray[T]:
    k = values.shape[-1]

    if k == 0:
        return np.zeros(values.shape[:-1], dtype=dtype)
    if k == 1:
        return values[..., 0].astype(dtype, copy=True)

    out = np.empty(
        values.shape[:-1],
        dtype=dtype,
    )
    if k <= _SUM_SMALL_AXIS_THRESHOLDS.get(values.dtype.itemsize, 0):
        np.add(values[..., 0], values[..., 1], out=out, dtype=dtype, casting="unsafe")
        for i in range(2, k):
            np.add(out, values[..., i], out=out, dtype=dtype, casting="unsafe")
        return out
    np.einsum("...i->...", values, out=out, dtype=dtype, casting="unsafe")
    return out


class _Reducer[T: NumericDTypeScalar](Protocol):
    def __call__(
        self,
        a: NDArray[T],
        /,
        *,
        axis: int,
    ) -> NDArray[T]: ...


def _reduce_minmax_last_axis[T: NumericDTypeScalar](
    values: NDArray[T],
    op: np.ufunc,
    fallback: _Reducer[T],
) -> NDArray[T]:
    k = values.shape[-1]

    if k == 0:
        return fallback(values, axis=-1)
    if k == 1:
        return values[..., 0].copy()

    if k <= _MINMAX_SMALL_AXIS_THRESHOLDS.get(values.dtype.itemsize, 0):
        out = np.empty(
            values.shape[:-1],
            dtype=values.dtype,
        )
        op(values[..., 0], values[..., 1], out=out)

        for i in range(2, k):
            op(out, values[..., i], out=out)

        return out

    out = fallback(values, axis=-1)

    return out


def _reduce_min_last_axis[T: NumericDTypeScalar](values: NDArray[T]) -> NDArray[T]:
    return _reduce_minmax_last_axis(values, np.minimum, np.min)


def _reduce_max_last_axis[T: NumericDTypeScalar](values: NDArray[T]) -> NDArray[T]:
    return _reduce_minmax_last_axis(values, np.maximum, np.max)
