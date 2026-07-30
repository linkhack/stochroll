from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._reductions import (
    _default_sum_dtype,
    _reduce_max_last_axis,
    _reduce_min_last_axis,
    _reduce_sum_last_axis,
    _signed_dtype_for_unsigned,
)
from ._routing import _route_all_indexed as _route_all_backend
from ._routing import _route_any_indexed as _route_any_backend
from ._routing import _route_multiply_indexed as _route_multiply_backend
from ._routing import _route_sum_indexed as _route_sum_backend
from ._typing import (
    AxisLike,
    EventArray,
    IntegerArray,
    NumericScalar,
    PoolArray,
    RollArray,
    ShapeLike,
)

type NumericLike = Roll | NumericScalar
type ElementwiseWrapper = Roll | Event
type ElementwiseOperand = ElementwiseWrapper | NumericScalar
type FixedIndices = int | slice | ArrayLike
type LookupIndices = Roll | ArrayLike
type AssemblyValue = Roll | Event | Pool


# ============================================================
# Helpers
# ============================================================
@overload
def _validated_operand_values(
    left: Roll,
    right: Roll | NumericScalar,
) -> RollArray | NumericScalar: ...


@overload
def _validated_operand_values(
    left: Event,
    right: Event,
) -> EventArray: ...


@overload
def _validated_operand_values(
    left: Event,
    right: Roll | NumericScalar,
) -> RollArray | NumericScalar: ...


def _validated_operand_values(
    left: ElementwiseWrapper,
    right: ElementwiseOperand,
) -> RollArray | EventArray | NumericScalar:
    if isinstance(left, Roll) and isinstance(right, Event):
        raise TypeError(
            "Event cannot be combined arithmetically with Roll; use .count() first"
        )

    if not isinstance(right, (Roll, Event)):
        return right

    if left.values.shape[0] != right.values.shape[0]:
        raise ValueError("elementwise operands must have matching repetitions")
    if left.values.ndim != right.values.ndim:
        raise ValueError(
            "Opperand ranks must match; use add_axis() to express "
            "structural broadcasting explicitly"
        )

    return right.values


def _normalize_shape(shape: ShapeLike | None) -> tuple[int, ...]:
    if shape is None:
        return ()
    if isinstance(shape, int):
        return (shape,)
    return shape


def _normalize_axis_index(axis: int, ndim: int) -> int:
    if isinstance(axis, bool):
        raise TypeError("axis must be an integer, not bool")

    normalized = axis + ndim if axis < 0 else axis
    if not 0 <= normalized < ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")

    return normalized


def _normalize_axis_tuple(axis: AxisLike, ndim: int) -> tuple[int, ...] | None:
    if axis is None:
        return None

    axes = (axis,) if isinstance(axis, int) else axis
    return tuple(_normalize_axis_index(index, ndim) for index in axes)


def _normalize_reduction_axis(axis: AxisLike, ndim: int) -> tuple[int, ...]:
    normalized = _normalize_axis_tuple(axis, ndim)
    if normalized is None or 0 in normalized:
        raise ValueError("cannot reduce the repetitions axis")
    return normalized


def _normalize_structural_axis(
    axis: int,
    ndim: int,
    *,
    pool: bool = False,
) -> int:
    selected = _normalize_axis_index(axis, ndim)

    if selected == 0:
        raise ValueError("cannot select the repetitions axis")
    if pool and selected == ndim - 1:
        raise ValueError("cannot select the Pool dice axis")

    return selected


def _integer_indices(indices: ArrayLike) -> IntegerArray:
    values = np.asarray(indices)
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("indices must have an integer dtype")
    return cast(IntegerArray, values)


def _validate_index_bounds(
    indices: IntegerArray,
    *,
    axis: int,
    size: int,
) -> None:
    if np.any(indices < 0):
        raise IndexError("indices must be non-negative")
    if np.any(indices >= size):
        raise IndexError(f"index out of bounds for axis {axis} with size {size}")


def _select_values[DType: np.generic](
    values: NDArray[DType],
    indices: FixedIndices,
    *,
    axis: int,
) -> NDArray[DType]:
    if isinstance(indices, slice):
        selection = (slice(None),) * axis + (indices,)
        return values[selection]

    integer_indices = _integer_indices(indices)
    _validate_index_bounds(integer_indices, axis=axis, size=values.shape[axis])
    return np.take(values, integer_indices, axis=axis)


def _normalize_lookup_indices[DType: np.generic](
    values: NDArray[DType],
    indices: LookupIndices,
    *,
    axis: int,
    structural_ndim: int,
) -> IntegerArray:
    raw_indices = indices.values if isinstance(indices, Roll) else indices
    normalized = _integer_indices(raw_indices)

    if normalized.ndim != values.ndim:
        if structural_ndim != 1 or normalized.ndim not in (1, 2):
            raise ValueError(
                "lookup indices must have the same rank as the source; "
                "shorthand is only supported for one structural axis"
            )

        repetitions = normalized.shape[0]
        if normalized.ndim == 1:
            shape = [1] * values.ndim
            shape[0] = repetitions
            normalized = normalized.reshape(shape)
        else:
            lookup_size = normalized.shape[1]
            shape = [1] * values.ndim  # [1,1,1]
            shape[0] = repetitions  # [R,1,1]
            shape[axis] = lookup_size  # [R,T,1] or [R,1,T]
            normalized = normalized.reshape(shape)

    if normalized.ndim != values.ndim:
        raise ValueError("lookup indices must have the same rank as the source")

    repetitions = normalized.shape[0]
    if repetitions not in (1, values.shape[0]):
        raise ValueError(
            "lookup indices repetitions dimension must be 1 or match the source"
        )

    for dimension, (index_size, source_size) in enumerate(
        zip(normalized.shape, values.shape, strict=True)
    ):
        if dimension in (0, axis):
            continue
        if index_size not in (1, source_size):
            raise ValueError(
                "lookup index dimensions must be 1 or match the source "
                f"(dimension {dimension}: {index_size} not in (1, {source_size}))"
            )

    _validate_index_bounds(normalized, axis=axis, size=values.shape[axis])
    return normalized


def _lookup_values[DType: np.generic](
    values: NDArray[DType],
    indices: LookupIndices,
    *,
    axis: int,
    structural_ndim: int,
) -> NDArray[DType]:
    normalized = _normalize_lookup_indices(
        values,
        indices,
        axis=axis,
        structural_ndim=structural_ndim,
    )
    return np.take_along_axis(values, normalized, axis=axis)


def _normalize_route_size(size: int) -> int:
    if isinstance(size, (bool, np.bool_)):
        raise TypeError("size must be an integer, not bool")

    try:
        normalized = operator.index(size)
    except TypeError:
        raise TypeError("size must be an integer") from None

    if normalized <= 0:
        raise ValueError(f"size must be positive, got {normalized}")

    return normalized


def _prepare_route_inputs[DType: np.generic](
    values: NDArray[DType],
    destinations: LookupIndices,
    *,
    size: int,
    axis: int,
) -> tuple[NDArray[DType], IntegerArray, int, int]:
    """Validate and canonicalize route inputs before backend dispatch."""
    normalized_size = _normalize_route_size(size)
    raw_destinations = (
        destinations.values if isinstance(destinations, Roll) else destinations
    )
    integer_destinations = _integer_indices(raw_destinations)

    if values.ndim == 1:
        normalized_axis = _normalize_axis_index(axis, values.ndim + 1)
        if normalized_axis == 0:
            raise ValueError("cannot route along the repetitions axis")
        if integer_destinations.shape != values.shape:
            raise ValueError(
                "scalar route destinations must have the same shape as the source"
            )

        _validate_index_bounds(
            integer_destinations,
            axis=normalized_axis,
            size=normalized_size,
        )
        canonical_values = np.expand_dims(values, axis=1)
        canonical_destinations = np.expand_dims(integer_destinations, axis=1)
        return (
            canonical_values,
            canonical_destinations,
            normalized_axis,
            normalized_size,
        )

    normalized_axis = _normalize_structural_axis(axis, values.ndim)
    if integer_destinations.ndim != values.ndim:
        raise ValueError(
            "route destinations must have the same rank as the source; "
            "lower-rank broadcasting is not supported"
        )

    repetitions = integer_destinations.shape[0]
    if repetitions not in (1, values.shape[0]):
        raise ValueError(
            "route destinations repetitions dimension must be 1 or match the source"
        )

    for dimension, (destination_size, source_size) in enumerate(
        zip(integer_destinations.shape, values.shape, strict=True)
    ):
        if dimension == 0:
            continue
        if source_size == 0:
            if destination_size != 0:
                raise ValueError(
                    "route destination dimensions must be zero when the "
                    "corresponding source dimension is zero"
                )
        elif destination_size not in (1, source_size):
            raise ValueError(
                "route destination dimensions must be 1 or match the source "
                f"(dimension {dimension}: {destination_size} not in "
                f"(1, {source_size}))"
            )

    _validate_index_bounds(
        integer_destinations,
        axis=normalized_axis,
        size=normalized_size,
    )
    broadcasted_destinations = np.broadcast_to(integer_destinations, values.shape)
    return (
        values,
        broadcasted_destinations,
        normalized_axis,
        normalized_size,
    )


def _add_structural_axis[DType: np.generic](
    values: NDArray[DType],
    axis: int,
) -> NDArray[DType]:
    output_ndim = values.ndim + 1
    normalized = _normalize_axis_index(axis, output_ndim)
    if normalized == 0:
        raise ValueError("cannot insert an axis before the repetitions axis")
    return np.expand_dims(values, axis=normalized)


def _normalize_assembly_axis(
    axis: int,
    ndim: int,
    *,
    operation: str,
) -> int:
    assembly_axis = _normalize_axis_index(axis, ndim)
    if assembly_axis == 0:
        raise ValueError(f"cannot {operation} along the repetitions axis")
    return assembly_axis


def _validate_assembly_values(
    values: Sequence[AssemblyValue],
) -> tuple[AssemblyValue, ...]:
    items = tuple(values)
    if not items:
        raise ValueError("values must contain at least one wrapper")

    first_type = type(items[0])
    if first_type not in (Roll, Event, Pool):
        raise TypeError("values must contain only Roll, Event, or Pool wrappers")
    if any(type(item) is not first_type for item in items[1:]):
        raise TypeError("values must contain homogeneous wrapper types")

    repetitions = items[0].values.shape[0]
    if any(item.values.shape[0] != repetitions for item in items[1:]):
        raise ValueError("all values must have matching repetitions")

    ndim = items[0].values.ndim
    if any(item.values.ndim != ndim for item in items[1:]):
        raise ValueError("all values must have matching ranks")

    if first_type is Pool:
        first_pool = cast(Pool, items[0])
        for item in items[1:]:
            pool = cast(Pool, item)
            if pool.values.shape[-1] != first_pool.values.shape[-1]:
                raise ValueError("all Pools must have matching dice extents")
            if pool.sides != first_pool.sides:
                raise ValueError("all Pools must have matching sides")
            if pool.roller is not first_pool.roller:
                raise ValueError("all Pools must reference the same Roller")

    return items


def _wrap_assembled(
    values: NDArray[np.generic],
    *,
    template: AssemblyValue,
) -> AssemblyValue:
    if type(template) is Roll:
        return Roll(cast(RollArray, values))
    if type(template) is Event:
        return Event(cast(EventArray, values))

    pool = cast(Pool, template)
    return Pool(
        cast(PoolArray, values),
        sides=pool.sides,
        roller=pool.roller,
    )


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

    def select(self, indices: FixedIndices, *, axis: int = -2) -> Pool:
        """Select fixed entries from a structural axis.

        ``indices`` may be a non-negative integer, a slice, or an integer
        array-like. A scalar integer removes the selected axis; a slice or
        array replaces it with the index shape, following ``numpy.take``
        semantics. Negative explicit indices and out-of-bounds values raise
        ``IndexError``; slices retain ordinary NumPy normalization.

        Axes use absolute NumPy numbering, including negative axes. The
        default ``-2`` is the last structural axis. Axis 0 (repetitions) and
        the final Pool dice axis are prohibited. The result remains a Pool
        with the same ``sides`` and ``roller``.
        """
        selected_axis = _normalize_structural_axis(
            axis,
            self.values.ndim,
            pool=True,
        )
        values = _select_values(self.values, indices, axis=selected_axis)
        return Pool(values, sides=self.sides, roller=self.roller)

    def lookup(self, indices: LookupIndices, *, axis: int = -2) -> Pool:
        """Look up structural entries using resolved per-repetition indices.

        ``indices`` may be a Roll or a raw integer array-like. Integer dtypes
        are required; values must be non-negative and in bounds. Canonical
        indices have the same rank as the Pool, with dimension 0 equal to the
        repetition count or 1, the lookup axis holding the result extent, and
        every other dimension equal to the source extent or 1.

        For a Pool with exactly one structural axis, ``(R,)`` and ``(R, K)``
        (including a leading singleton repetition dimension) are accepted as
        shorthand. Pools with multiple structural axes require full-rank
        indices with explicit singleton dimensions. Axis 0 and the final dice
        axis are prohibited; the default ``-2`` selects the last structural
        axis. The output is a Pool whose broadcast lookup shape retains the
        final dice axis and the original ``sides`` and ``roller``. Indices are
        not stored as provenance.
        """
        selected_axis = _normalize_structural_axis(
            axis,
            self.values.ndim,
            pool=True,
        )
        values = _lookup_values(
            self.values,
            indices,
            axis=selected_axis,
            structural_ndim=self.values.ndim - 2,
        )
        return Pool(values, sides=self.sides, roller=self.roller)

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
        return Roll(self.values + _validated_operand_values(self, other))

    def __radd__(self, other: NumericScalar) -> Roll:
        return self + other

    def __sub__(self, other: Roll | NumericScalar) -> Roll:
        return Roll(self.values - _validated_operand_values(self, other))

    def __rsub__(self, other: NumericScalar) -> Roll:
        return Roll(_validated_operand_values(self, other) - self.values)

    def __mul__(self, other: Roll | NumericScalar) -> Roll:
        return Roll(self.values * _validated_operand_values(self, other))

    def __rmul__(self, other: NumericScalar) -> Roll:
        return self * other

    # Comparisons intentionally return Event for the vectorized dice DSL.
    def __eq__(self, other: Roll | NumericScalar) -> Event:  # type: ignore[override]
        return Event(self.values == _validated_operand_values(self, other))

    def __ne__(self, other: NumericLike) -> Event:  # type: ignore[override]
        return Event(self.values != _validated_operand_values(self, other))

    def __le__(self, other: Roll | NumericScalar) -> Event:
        return Event(self.values <= _validated_operand_values(self, other))

    def __lt__(self, other: Roll | NumericScalar) -> Event:
        return Event(self.values < _validated_operand_values(self, other))

    def __ge__(self, other: Roll | NumericScalar) -> Event:
        return Event(self.values >= _validated_operand_values(self, other))

    def __gt__(self, other: Roll | NumericScalar) -> Event:
        return Event(self.values > _validated_operand_values(self, other))

    # ------------------------------------------------------------
    # Structural indexing
    # ------------------------------------------------------------
    def select(self, indices: FixedIndices, *, axis: int = -1) -> Roll:
        """Select fixed entries from a structural axis.

        ``indices`` may be a non-negative integer, a slice, or an integer
        array-like. A scalar integer removes the selected axis; a slice or
        array replaces it with the index shape, following ``numpy.take``
        semantics. Negative explicit indices and out-of-bounds values raise
        ``IndexError``; slices retain ordinary NumPy normalization.

        Axes use absolute NumPy numbering, including negative axes. Axis 0 is
        the protected repetitions axis and cannot be selected. The default is
        the final structural axis, and the result is a Roll.
        """
        selected_axis = _normalize_structural_axis(axis, self.values.ndim)
        return Roll(_select_values(self.values, indices, axis=selected_axis))

    def lookup(self, indices: LookupIndices, *, axis: int = -1) -> Roll:
        """Look up structural entries using resolved per-repetition indices.

        ``indices`` may be a Roll or a raw integer array-like. Integer dtypes
        are required; values must be non-negative and in bounds. Canonical
        indices have the same rank as this Roll, with dimension 0 equal to the
        repetition count or 1, the lookup axis holding the result extent, and
        every other dimension equal to the source extent or 1.

        For a Roll with exactly one structural axis, ``(R,)`` and ``(R, K)``
        (including a leading singleton repetition dimension) are accepted as
        shorthand. Rolls with multiple structural axes require full-rank
        indices with explicit singleton dimensions. Axis 0 is prohibited.
        The output is a Roll with the broadcast normalized index shape;
        indices are not stored as provenance.
        """
        selected_axis = _normalize_structural_axis(axis, self.values.ndim)
        return Roll(
            _lookup_values(
                self.values,
                indices,
                axis=selected_axis,
                structural_ndim=self.values.ndim - 1,
            )
        )

    def add_axis(self, axis: int = -1) -> Roll:
        """Insert a singleton structural axis without creating new values.

        ``axis`` uses NumPy insertion-axis coordinates, including negative
        axes. Position 0, before the protected repetitions axis, is rejected.
        The default ``-1`` appends a trailing structural axis. The result is a
        Roll with one additional length-one dimension.
        """
        return Roll(_add_structural_axis(self.values, axis))

    def route_sum(
        self,
        destinations: LookupIndices,
        *,
        size: int,
        axis: int = -1,
    ) -> Roll:
        """Route numeric values to destinations and sum collisions.

        ``destinations`` is a resolved ``Roll`` or an integer array-like of
        zero-based, non-negative destination indices. ``size`` is keyword-only
        and must be a strictly positive integer. Axes use absolute NumPy
        numbering and axis 0, the repetitions axis, is rejected. For shaped
        values, the selected structural axis is replaced by ``size``. For a
        source shaped ``(R,)``, destinations must also be ``(R,)`` and the
        destination axis is inserted after repetitions, producing ``(R, size)``;
        ``axis=1`` and ``axis=-1`` are equivalent in that case.

        Shaped destinations must have the same rank as the source. Their
        repetition dimension may be ``R`` or ``1`` and their structural
        dimensions may be ``1`` or the corresponding source extent. Singleton
        dimensions are explicit; lower-rank broadcasting is rejected, and
        zero-length source dimensions require matching zero-length
        destination dimensions. Invalid dtypes, bounds, shapes, axes, and
        sizes are rejected before accumulation.

        Duplicate destinations are summed, never overwritten. The result is a
        ``Roll`` using the same accumulation dtype convention as ``sum``:
        integer inputs accumulate to ``int64`` and floating inputs to
        ``float64``. Empty source write axes produce zero-filled output with
        the selected positive destination size. Inputs are not mutated.

        Pool routing and overwrite policies are not supported. The active
        routing backend is intentionally kept behind the validated public
        method boundary so implementations can be exchanged for local
        correctness and performance experiments without changing these
        semantics.
        """
        values, normalized_destinations, normalized_axis, normalized_size = (
            _prepare_route_inputs(
                self.values,
                destinations,
                size=size,
                axis=axis,
            )
        )
        dtype = _default_sum_dtype(self.values.dtype)
        return Roll(
            _route_sum_backend(
                values,
                normalized_destinations,
                size=normalized_size,
                axis=normalized_axis,
                dtype=dtype,
            )
        )

    def route_multiply(
        self,
        destinations: LookupIndices,
        *,
        size: int,
        axis: int = -1,
    ) -> Roll:
        """Route numeric values to destinations and multiply collisions.

        Routing uses the same destination, shape, axis, and validation rules
        as :meth:`route_sum`. Duplicate destinations are multiplied together;
        destination slots with no source values retain the multiplicative
        identity of one. Numeric outputs use the same accumulation dtype
        convention as :meth:`sum`.
        """
        values, normalized_destinations, normalized_axis, normalized_size = (
            _prepare_route_inputs(
                self.values,
                destinations,
                size=size,
                axis=axis,
            )
        )
        dtype = _default_sum_dtype(self.values.dtype)
        return Roll(
            _route_multiply_backend(
                values,
                normalized_destinations,
                size=normalized_size,
                axis=normalized_axis,
                dtype=dtype,
            )
        )

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


@dataclass(frozen=True, slots=True, eq=False)
class Event:
    values: EventArray

    def __post_init__(self) -> None:
        values = self.values
        if values.ndim < 1:
            raise ValueError("values must be at least 1-D")
        if values.shape[0] == 0:
            raise ValueError("repetitions must be >= 1")

    def __or__(self, other: Event) -> Event:
        if not isinstance(other, Event):
            raise TypeError("Event boolean operations require another Event")
        return Event(self.values | _validated_operand_values(self, other))

    def __and__(self, other: Event) -> Event:
        if not isinstance(other, Event):
            raise TypeError("Event boolean operations require another Event")
        return Event(self.values & _validated_operand_values(self, other))

    def __invert__(self) -> Event:
        return Event(~self.values)

    def select(self, indices: FixedIndices, *, axis: int = -1) -> Event:
        """Select fixed entries from a structural axis.

        ``indices`` may be a non-negative integer, a slice, or an integer
        array-like. A scalar integer removes the selected axis; a slice or
        array replaces it with the index shape, following ``numpy.take``
        semantics. Negative explicit indices and out-of-bounds values raise
        ``IndexError``; slices retain ordinary NumPy normalization.

        Axes use absolute NumPy numbering, including negative axes. Axis 0 is
        the protected repetitions axis and cannot be selected. The default is
        the final structural axis, and the result is an Event.
        """
        selected_axis = _normalize_structural_axis(axis, self.values.ndim)
        return Event(_select_values(self.values, indices, axis=selected_axis))

    def lookup(self, indices: LookupIndices, *, axis: int = -1) -> Event:
        """Look up structural entries using resolved per-repetition indices.

        ``indices`` may be a Roll or a raw integer array-like. Integer dtypes
        are required; values must be non-negative and in bounds. Canonical
        indices have the same rank as this Event, with dimension 0 equal to
        the repetition count or 1, the lookup axis holding the result extent,
        and every other dimension equal to the source extent or 1.

        For an Event with exactly one structural axis, ``(R,)`` and ``(R, K)``
        (including a leading singleton repetition dimension) are accepted as
        shorthand. Events with multiple structural axes require full-rank
        indices with explicit singleton dimensions. Axis 0 is prohibited.
        The output is an Event with the broadcast normalized index shape;
        indices are not stored as provenance.
        """
        selected_axis = _normalize_structural_axis(axis, self.values.ndim)
        return Event(
            _lookup_values(
                self.values,
                indices,
                axis=selected_axis,
                structural_ndim=self.values.ndim - 1,
            )
        )

    def add_axis(self, axis: int = -1) -> Event:
        """Insert a singleton structural axis without creating new values.

        ``axis`` uses NumPy insertion-axis coordinates, including negative
        axes. Position 0, before the protected repetitions axis, is rejected.
        The default ``-1`` appends a trailing structural axis. The result is
        an Event with one additional length-one dimension.
        """
        return Event(_add_structural_axis(self.values, axis))

    def route_any(
        self,
        destinations: LookupIndices,
        *,
        size: int,
        axis: int = -1,
    ) -> Event:
        """Route Boolean values to destinations and combine collisions with OR.

        ``destinations`` is a resolved ``Roll`` or an integer array-like of
        zero-based, non-negative destination indices. ``size`` is keyword-only
        and must be a strictly positive integer. Axes use absolute NumPy
        numbering and axis 0, the repetitions axis, is rejected. For shaped
        values, the selected structural axis is replaced by ``size``. For a
        source shaped ``(R,)``, destinations must also be ``(R,)`` and the
        destination axis is inserted after repetitions, producing ``(R, size)``;
        ``axis=1`` and ``axis=-1`` are equivalent in that case.

        Shaped destinations must have the same rank as the source. Their
        repetition dimension may be ``R`` or ``1`` and their structural
        dimensions may be ``1`` or the corresponding source extent. Singleton
        dimensions are explicit; lower-rank broadcasting is rejected, and
        zero-length source dimensions require matching zero-length
        destination dimensions. Invalid dtypes, bounds, shapes, axes, and
        sizes are rejected before accumulation.

        Duplicate destinations are combined with logical OR, never
        overwritten. The result is an ``Event`` with Boolean data. Empty
        source write axes produce ``False``-filled output with the selected
        positive destination size. Inputs are not mutated.

        Pool routing and overwrite policies are not supported. The active
        routing backend is intentionally kept behind the validated public
        method boundary so implementations can be exchanged for local
        correctness and performance experiments without changing these
        semantics.
        """
        values, normalized_destinations, normalized_axis, normalized_size = (
            _prepare_route_inputs(
                self.values,
                destinations,
                size=size,
                axis=axis,
            )
        )
        return Event(
            _route_any_backend(
                values,
                normalized_destinations,
                size=normalized_size,
                axis=normalized_axis,
            )
        )

    def route_all(
        self,
        destinations: LookupIndices,
        *,
        size: int,
        axis: int = -1,
    ) -> Event:
        """Route Boolean values to destinations and combine collisions with AND.

        Routing uses the same destination, shape, axis, and validation rules
        as :meth:`route_any`. Duplicate destinations are combined with
        logical AND; destination slots with no source values retain the
        Boolean identity ``True``.
        """
        values, normalized_destinations, normalized_axis, normalized_size = (
            _prepare_route_inputs(
                self.values,
                destinations,
                size=size,
                axis=axis,
            )
        )
        return Event(
            _route_all_backend(
                values,
                normalized_destinations,
                size=normalized_size,
                axis=normalized_axis,
            )
        )

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
    return Roll(
        np.where(
            event.values,
            _validated_operand_values(event, yes),
            _validated_operand_values(event, no),
        )
    )


@overload
def stack(values: Sequence[Roll], *, axis: int = 1) -> Roll: ...


@overload
def stack(values: Sequence[Event], *, axis: int = 1) -> Event: ...


@overload
def stack(values: Sequence[Pool], *, axis: int = 1) -> Pool: ...


def stack(
    values: Sequence[AssemblyValue],
    *,
    axis: int = 1,
) -> AssemblyValue:
    """Stack homogeneous wrappers along a new structural axis.

    ``values`` must be a nonempty sequence containing only ``Roll``,
    ``Event``, or ``Pool`` objects of one wrapper type. Every input must have
    the same rank, shape, and repetition count; inputs are not broadcast.
    The result preserves the wrapper type and follows NumPy dtype promotion.

    ``axis`` uses ``numpy.stack`` output-array coordinates, including negative
    axes. The default ``axis=1`` inserts the new axis immediately after
    repetitions. Axis 0 is prohibited because repetitions are independent
    simulation samples. If each input has shape ``(R, *S)``, the result inserts
    ``len(values)`` at the normalized output axis.

    Pool inputs must additionally have matching ``sides`` and dice extents and
    reference the same ``Roller`` object. The new axis must be inserted before
    the final dice axis, which remains unchanged; stacking after the dice axis
    is prohibited.
    """
    items = _validate_assembly_values(values)
    template = items[0]
    input_ndim = template.values.ndim
    stack_axis = _normalize_assembly_axis(
        axis,
        input_ndim + 1,
        operation="stack",
    )

    if type(template) is Pool and stack_axis == input_ndim:
        raise ValueError("cannot stack after the Pool dice axis")

    shape = template.values.shape
    if any(item.values.shape != shape for item in items[1:]):
        raise ValueError("all values must have matching shapes for stack")

    result = np.stack([item.values for item in items], axis=stack_axis)
    return _wrap_assembled(result, template=template)


@overload
def concatenate(values: Sequence[Roll], *, axis: int = 1) -> Roll: ...


@overload
def concatenate(values: Sequence[Event], *, axis: int = 1) -> Event: ...


@overload
def concatenate(values: Sequence[Pool], *, axis: int = 1) -> Pool: ...


def concatenate(
    values: Sequence[AssemblyValue],
    *,
    axis: int = 1,
) -> AssemblyValue:
    """Concatenate homogeneous wrappers along a structural axis.

    ``values`` must be a nonempty sequence containing only ``Roll``,
    ``Event``, or ``Pool`` objects of one wrapper type. Every input must have
    the same rank and repetition count. All dimensions except ``axis`` must
    match exactly; inputs are not broadcast. The result preserves the wrapper
    type, sums the selected-axis extents, and follows NumPy dtype promotion.

    ``axis`` uses ``numpy.concatenate`` input-array coordinates, including
    negative axes. The default is the first structural axis, ``axis=1``.
    Axis 0 is prohibited because repetitions are independent simulation
    samples.

    Pool inputs must additionally have matching ``sides`` and final dice
    extents and reference the same ``Roller`` object. Concatenation may target
    only an existing structural axis; the final dice axis is prohibited. A
    Pool shaped only ``(R, D)`` therefore has no valid concatenation axis.
    """
    items = _validate_assembly_values(values)
    template = items[0]
    concatenate_axis = _normalize_assembly_axis(
        axis,
        template.values.ndim,
        operation="concatenate",
    )

    if type(template) is Pool and concatenate_axis == template.values.ndim - 1:
        raise ValueError("cannot concatenate along the Pool dice axis")

    expected_shape = template.values.shape
    for item in items[1:]:
        for dimension, (actual, expected) in enumerate(
            zip(item.values.shape, expected_shape, strict=True)
        ):
            if dimension != concatenate_axis and actual != expected:
                raise ValueError(
                    "all non-concatenated dimensions must match "
                    f"(dimension {dimension}: {actual} != {expected})"
                )

    result = np.concatenate(
        [item.values for item in items],
        axis=concatenate_axis,
    )
    return _wrap_assembled(result, template=template)
