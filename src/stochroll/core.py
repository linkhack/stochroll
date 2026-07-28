from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._reductions import (
    _default_sum_dtype,
    _reduce_max_last_axis,
    _reduce_min_last_axis,
    _reduce_sum_last_axis,
    _signed_dtype_for_unsigned,
)
from ._typing import (
    AxisLike,
    EventArray,
    NumericScalar,
    PoolArray,
    RollArray,
    ShapeLike,
)

type NumericLike = Roll | NumericScalar
type InternalValue = Roll | Event | NumericScalar


# ============================================================
# Helpers
# ============================================================
def _vals(x: InternalValue) -> RollArray | NumericScalar:
    if isinstance(x, Event):
        raise TypeError(
            "Event cannot be combined arithmetically with Roll; use .count() first"
        )
    return x.values if isinstance(x, Roll) else x


def _normalize_shape(shape: ShapeLike | None) -> tuple[int, ...]:
    if shape is None:
        return ()
    if isinstance(shape, int):
        return (shape,)
    return shape


def _normalize_axis(axis: AxisLike, ndim: int) -> tuple[int, ...] | None:
    if axis is None:
        return None

    axes = (axis,) if isinstance(axis, int) else axis
    normalized = tuple(i + ndim if i < 0 else i for i in axes)

    if not all(0 <= i < ndim for i in normalized):
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")

    return normalized


def _normalize_reduction_axis(axis: AxisLike, ndim: int) -> tuple[int, ...]:
    normalized = _normalize_axis(axis, ndim)
    if normalized is None or 0 in normalized:
        raise ValueError("cannot reduce the repetitions axis")
    return normalized


# ============================================================
# Core simulation engine
# ============================================================


class Roller:
    def __init__(self, repetitions: int = 1_000_000, seed: int | None = None) -> None:
        if repetitions < 1:
            raise ValueError("repetitions must be >= 1")
        self.repetitions = repetitions
        self.rng = np.random.default_rng(seed=seed)

    def d(self, sides: int, *, shape: ShapeLike | None = None) -> Roll:
        """
        Roll one die per simulation sample.

        ``shape`` creates trailing structural axes of independent resolved
        rolls, not dice pools. Entries must be non-negative; zero-sized axes
        represent empty collections and follow NumPy reduction semantics.
        The leading repetitions axis is always nonempty.

        Examples
        --------
        r.d(20)
            One d20 per sample. Shape: (samples,)

        r.d(20, shape=3)
            Three independent d20 rolls per sample. Shape: (samples, 3)

        Use r.pool(4, d=6) for 4d6.
        """
        shape = _normalize_shape(shape)
        if sides < 1:
            raise ValueError(f"sides must be >= 1, got {sides}")
        if any(s < 0 for s in shape):
            raise ValueError(f"shape entries must be non-negative, got {shape}")
        dtype = np.min_scalar_type(sides)

        values = self.rng.integers(
            1,
            sides + 1,
            size=(self.repetitions, *shape),
            dtype=dtype,
        )

        return Roll(values)

    def pool(
        self,
        dice: int,
        *,
        d: int,
        shape: ShapeLike | None = None,
    ) -> Pool:
        """
        Roll an unresolved dice pool.

        The returned shape is ``(repetitions, *shape, dice)``. ``shape``
        entries must be non-negative and may be zero, representing empty
        collections of pools. The leading repetitions axis and final dice
        axis are always nonempty.

        Examples
        --------
        r.pool(4, d=6)
            One 4d6 pool per sample. Shape: (samples, 4)

        r.pool(4, d=6, shape=6)
            Six independent 4d6 pools per sample. Shape: (samples, 6, 4)
        """
        shape = _normalize_shape(shape)
        if dice < 1:
            raise ValueError(f"dice must be >= 1, got {dice}")
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        if any(s < 0 for s in shape):
            raise ValueError(f"shape entries must be non-negative, got {shape}")
        dtype = np.min_scalar_type(d)

        values = self.rng.integers(
            1,
            d + 1,
            size=(self.repetitions, *shape, dice),
            dtype=dtype,
        )

        return Pool(values, sides=d, roller=self)


# ============================================================
# Pool: unresolved dice
# ============================================================


@dataclass(frozen=True, slots=True)
class Pool:
    values: PoolArray
    sides: int  # >=1
    roller: Roller

    def __post_init__(self) -> None:
        if self.values.ndim < 2:
            raise ValueError("values must be at least 2-D")
        if self.values.shape[0] == 0:
            raise ValueError("repetitions must be >= 1")
        if self.values.shape[-1] == 0:
            raise ValueError("Pool must contain at least one die")
        if self.sides < 1:
            raise ValueError("sides must be >= 1")

    def first(self) -> Roll:
        return Roll(self.values[..., 0])

    def last(self) -> Roll:
        return Roll(self.values[..., -1])

    def single(self) -> Roll:
        if self.values.shape[-1] != 1:
            raise ValueError("Pool contains more than one die. Reduce it first.")
        return Roll(self.values[..., 0])

    def sum(self) -> Roll:
        k = self.values.shape[-1]
        dtype = np.min_scalar_type(self.sides * k)
        return Roll(_reduce_sum_last_axis(self.values, dtype=dtype))

    def min(self) -> Roll:
        return Roll(_reduce_min_last_axis(self.values))

    def max(self) -> Roll:
        return Roll(_reduce_max_last_axis(self.values))

    def keep_highest(self, k: int) -> Pool:
        n = self.values.shape[-1]
        if not (0 <= k <= n):
            raise ValueError(f"k must be between 0 and {n}, got {k}")
        if k == 0:
            raise ValueError("k=0 is not supported")
        if k == n:
            return self
        if k > n // 2:
            return self.drop_lowest(n - k)
        values = np.partition(self.values, -k, axis=-1)[..., -k:]
        return Pool(values, sides=self.sides, roller=self.roller)

    def keep_lowest(self, k: int) -> Pool:
        n = self.values.shape[-1]
        if not (0 <= k <= n):
            raise ValueError(f"k must be between 0 and {n}, got {k}")
        if k == 0:
            raise ValueError("k=0 is not supported")
        if k == n:
            return self
        if k > n // 2:
            return self.drop_highest(n - k)
        values = np.partition(self.values, k - 1, axis=-1)[..., :k]
        return Pool(values, sides=self.sides, roller=self.roller)

    # drop is symtetric to keep: keep k is the same as drop n-k
    def drop_lowest(self, k: int = 1) -> Pool:
        n = self.values.shape[-1]
        if not (0 <= k <= n):
            raise ValueError(f"k must be between 0 and {n}, got {k}")
        if k == 0:
            return self
        if k == n:
            raise ValueError("k=n is not supported")
        if k > n // 2:
            return self.keep_highest(n - k)
        values = np.partition(self.values, k - 1, axis=-1)[..., k:]
        return Pool(values, sides=self.sides, roller=self.roller)

    def drop_highest(self, k: int = 1) -> Pool:
        n = self.values.shape[-1]
        if not (0 <= k <= n):
            raise ValueError(f"k must be between 0 and {n}, got {k}")
        if k == 0:
            return self
        if k == n:
            raise ValueError("k=n is not supported")
        if k > n // 2:
            return self.keep_lowest(n - k)
        values = np.partition(self.values, -k, axis=-1)[..., :-k]
        return Pool(values, sides=self.sides, roller=self.roller)

    def drop_lowest_sum(self) -> Roll:
        values = self.values
        k = values.shape[-1]

        dtype = np.min_scalar_type(self.sides * max(k - 1, 0))

        total = _reduce_sum_last_axis(values, dtype=dtype)
        lowest = _reduce_min_last_axis(values)

        np.subtract(
            total, lowest.astype(dtype, copy=False), out=total, casting="unsafe"
        )

        return Roll(total)

    def reroll_once(self, values: ArrayLike) -> Pool:
        targets = np.atleast_1d(values)
        if np.issubdtype(targets.dtype, np.integer):
            pass
        elif np.issubdtype(targets.dtype, np.floating):
            if not (np.isfinite(targets)).all():
                raise ValueError(f"reroll values must be finite, got {values!r}")
            if not (targets == np.trunc(targets)).all():
                raise ValueError(f"reroll values must be integers, got {values!r}")
        else:
            raise TypeError(f"reroll values must be integers, got {values!r}")

        if ((targets < 1) | (targets > self.sides)).any():
            raise ValueError(
                f"reroll values must be between 1 and {self.sides}, got {values!r}"
            )

        mask = np.isin(self.values, values)
        if not mask.any():
            return self

        out = self.values.copy()
        out[mask] = self.roller.rng.integers(
            1,
            self.sides + 1,
            size=int(mask.sum()),
            dtype=self.values.dtype,
        )

        return Pool(out, sides=self.sides, roller=self.roller)

    def count_at_least(self, target: NumericScalar) -> Roll:
        return Roll((self.values >= target).sum(axis=-1, dtype=np.int32))


# ============================================================
# Roll: resolved numeric value
# ============================================================


@dataclass(frozen=True, slots=True, eq=False)
class Roll:
    values: RollArray

    def __post_init__(self) -> None:
        values = self.values
        if values.ndim < 1:
            raise ValueError("values must be at least 1-D")
        if values.shape[0] == 0:
            raise ValueError("repetitions must be >= 1")
        if np.issubdtype(values.dtype, np.unsignedinteger):
            dtype = _signed_dtype_for_unsigned(values.dtype)
            object.__setattr__(self, "values", values.astype(dtype, copy=False))

    # ------------------------------------------------------------
    # DSL
    # ------------------------------------------------------------
    def __add__(self, other: Roll | NumericScalar) -> Roll:
        return Roll(self.values + _vals(other))

    def __radd__(self, other: NumericScalar) -> Roll:
        return Roll(_vals(other) + self.values)

    def __sub__(self, other: Roll | NumericScalar) -> Roll:
        return Roll(self.values - _vals(other))

    def __rsub__(self, other: NumericScalar) -> Roll:
        return Roll(_vals(other) - self.values)

    def __mul__(self, other: Roll | NumericScalar) -> Roll:
        return Roll(self.values * _vals(other))

    def __rmul__(self, other: NumericScalar) -> Roll:
        return Roll(_vals(other) * self.values)

    # Comparisons intentionally return Event for the vectorized dice DSL.
    def __eq__(self, other: Roll | NumericScalar) -> Event:  # type: ignore[override]
        return Event(self.values == _vals(other))

    def __ne__(self, other: NumericLike) -> Event:  # type: ignore[override]
        return Event(self.values != _vals(other))

    def __le__(self, other: Roll | NumericScalar) -> Event:
        return Event(self.values <= _vals(other))

    def __lt__(self, other: Roll | NumericScalar) -> Event:
        return Event(self.values < _vals(other))

    def __ge__(self, other: Roll | NumericScalar) -> Event:
        return Event(self.values >= _vals(other))

    def __gt__(self, other: Roll | NumericScalar) -> Event:
        return Event(self.values > _vals(other))

    # ------------------------------------------------------------
    # Shape reductions
    # ------------------------------------------------------------
    def sum(self, axis: AxisLike = -1) -> Roll:
        dtype = _default_sum_dtype(self.values.dtype)
        axis = _normalize_reduction_axis(axis, self.values.ndim)

        if axis == (self.values.ndim - 1,):
            return Roll(_reduce_sum_last_axis(self.values, dtype=dtype))

        return Roll(np.sum(self.values, axis=axis, dtype=dtype))

    def mean(self, axis: AxisLike = -1) -> Roll:
        dtype = _default_sum_dtype(self.values.dtype)
        k = self.values.shape[-1]
        axis = _normalize_reduction_axis(axis, self.values.ndim)

        if axis == (self.values.ndim - 1,):
            if k == 0:
                return Roll(np.mean(self.values, axis=axis))
            return Roll(_reduce_sum_last_axis(self.values, dtype=dtype) / k)

        return Roll(np.mean(self.values, axis=axis))

    def min(self, axis: AxisLike = -1) -> Roll:
        axis = _normalize_reduction_axis(axis, self.values.ndim)

        if axis == (self.values.ndim - 1,):
            return Roll(_reduce_min_last_axis(self.values))

        return Roll(np.min(self.values, axis=axis))

    def max(self, axis: AxisLike = -1) -> Roll:
        axis = _normalize_reduction_axis(axis, self.values.ndim)

        if axis == (self.values.ndim - 1,):
            return Roll(_reduce_max_last_axis(self.values))

        return Roll(np.max(self.values, axis=axis))

    def broadcast_to(self, *shape: int) -> Roll:
        """
        Broadcast each simulation sample to a full target shape.

        The repetitions axis is preserved. Existing structural axes align
        with the target shape from the right, following NumPy broadcasting
        rules.

        Example:
            Roll shape (T,) -> broadcast_to(6) -> (T, 6)
            Roll shape (T, 1) -> broadcast_to(6) -> (T, 6)
            Roll shape (T, 4) -> broadcast_to(6, 4) -> (T, 6, 4)
            Roll shape (T, 1, 4) -> broadcast_to(6, 4) -> (T, 6, 4)
        """
        if len(shape) == 0:
            return self

        values = self.values
        structural_ndim = values.ndim - 1
        if structural_ndim < len(shape):
            values = values.reshape(
                (values.shape[0],)
                + (1,) * (len(shape) - structural_ndim)
                + values.shape[1:]
            )

        values = np.broadcast_to(values, (values.shape[0], *shape))

        return Roll(values)

    # ------------------------------------------------------------
    # Statistical summaries
    # ------------------------------------------------------------
    def expected(self) -> NDArray[np.float64]:
        """Return the mean over the repetitions axis.

        Structural axes are preserved and the result uses ``float64``. A
        scalar-valued Roll may produce a NumPy array scalar, which behaves as
        a zero-dimensional array.
        """
        dtype = _default_sum_dtype(self.values.dtype)
        k = len(self.values)
        result = np.divide(
            np.sum(self.values, axis=0, dtype=dtype),
            k,
            dtype=np.float64,
        )
        return cast(NDArray[np.float64], result)

    def probability_at_least(self, target: NumericScalar) -> NDArray[np.float64]:
        """Return the probability of reaching at least ``target``.

        The repetitions axis is reduced, structural axes are preserved, and
        the result uses ``float64``. A scalar-valued Roll may produce a NumPy
        array scalar, which behaves as a zero-dimensional array.
        """
        k = len(self.values)
        out = np.count_nonzero(self.values >= target, axis=0)
        result = np.divide(out, k, dtype=np.float64)
        return cast(NDArray[np.float64], result)


# ============================================================
# Event: boolean mask
# ============================================================


@dataclass(frozen=True, slots=True)
class Event:
    values: EventArray

    def __post_init__(self) -> None:
        values = self.values
        if values.ndim < 1:
            raise ValueError("values must be at least 1-D")
        if values.shape[0] == 0:
            raise ValueError("repetitions must be >= 1")

    def __or__(self, other: Event) -> Event:
        return Event(self.values | other.values)

    def __and__(self, other: Event) -> Event:
        return Event(self.values & other.values)

    def __invert__(self) -> Event:
        return Event(~self.values)

    def broadcast_to(self, *shape: int) -> Event:
        """
        Broadcast each simulation sample to a full target shape.

        The repetitions axis is preserved. Existing structural axes align
        with the target shape from the right, following NumPy broadcasting
        rules.

        Example:
            Event shape (T,) -> broadcast_to(6) -> (T, 6)
            Event shape (T, 1) -> broadcast_to(6) -> (T, 6)
            Event shape (T, 4) -> broadcast_to(6, 4) -> (T, 6, 4)
            Event shape (T, 1, 4) -> broadcast_to(6, 4) -> (T, 6, 4)
        """
        if len(shape) == 0:
            return self

        values = self.values
        structural_ndim = values.ndim - 1
        if structural_ndim < len(shape):
            values = values.reshape(
                (values.shape[0],)
                + (1,) * (len(shape) - structural_ndim)
                + values.shape[1:]
            )

        values = np.broadcast_to(values, (values.shape[0], *shape))

        return Event(values)

    def count(self, axis: AxisLike = -1) -> Roll:
        axis = _normalize_reduction_axis(axis, self.values.ndim)
        return Roll(np.count_nonzero(self.values, axis=axis))

    def indicator(self) -> Roll:
        """Convert each event outcome to a signed integer 0 or 1.

        The repetitions and structural axes are preserved exactly. Use
        ``count()`` instead when reducing an event's structural axis.
        """
        return Roll(self.values.astype(np.int8, copy=False))

    def probability(self) -> NDArray[np.float64]:
        """Return the probability that this event is true.

        The repetitions axis is reduced, structural axes are preserved, and
        the result uses ``float64``. A scalar-valued Event may produce a NumPy
        array scalar, which behaves as a zero-dimensional array.
        """
        k = len(self.values)
        out = np.count_nonzero(self.values, axis=0)
        result = np.divide(out, k, dtype=np.float64)
        return cast(NDArray[np.float64], result)


# ============================================================
# Helper functions
# ============================================================


def where(
    event: Event, yes: Roll | NumericScalar, no: Roll | NumericScalar = 0
) -> Roll:
    return Roll(np.where(event.values, _vals(yes), _vals(no)))
