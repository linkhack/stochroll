"""Validated vectorized and reference candidates for WP-012."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import SupportsIndex, cast

import numpy as np
from numpy.typing import NDArray

from stochroll._typing import EventArray, IntegerArray, RollArray


@dataclass(frozen=True, slots=True)
class PreparedSampling:
    """Validated arrays and shapes shared by both mapping candidates."""

    moved_eligible: EventArray
    counts: NDArray[np.intp]
    offsets: NDArray[np.intp]
    eligible_positions: NDArray[np.intp]
    prefix_eligible: bool
    minimum_count: int | None
    uniform_count: int | None
    axis: int
    output_shape: tuple[int, ...]
    moved_output_shape: tuple[int, ...]

    @property
    def all_eligible(self) -> bool:
        return bool(self.minimum_count == self.moved_eligible.shape[-1])


def _positive_size(size: SupportsIndex) -> int:
    if isinstance(size, (bool, np.bool_)):
        raise TypeError("size must be an integer, not bool")
    try:
        normalized = operator.index(size)
    except TypeError:
        raise TypeError("size must be an integer") from None
    if normalized <= 0:
        raise ValueError(f"size must be positive, got {normalized}")
    return normalized


def _structural_axis(axis: SupportsIndex, ndim: int) -> int:
    if isinstance(axis, (bool, np.bool_)):
        raise TypeError("axis must be an integer, not bool")
    try:
        indexed = operator.index(axis)
    except TypeError:
        raise TypeError("axis must be an integer") from None

    normalized = indexed + ndim if indexed < 0 else indexed
    if not 0 <= normalized < ndim:
        raise ValueError(
            f"axis {indexed} is out of bounds for array of dimension {ndim}"
        )
    if normalized == 0:
        raise ValueError("cannot sample the repetitions axis")
    return normalized


def prepare_sampling(
    eligible: EventArray,
    *,
    repetitions: int,
    size: SupportsIndex,
    axis: SupportsIndex,
) -> PreparedSampling:
    """Validate a request completely before the caller accesses its RNG."""
    normalized_size = _positive_size(size)
    if eligible.shape[0] != repetitions:
        raise ValueError(f"eligible must have {repetitions} repetitions")
    normalized_axis = _structural_axis(axis, eligible.ndim)

    # Moving candidates last turns every repetition/preserved-coordinate
    # combination into one independent slice without flattening its identity.
    moved = np.moveaxis(eligible, normalized_axis, -1)
    counts = np.count_nonzero(moved, axis=-1)
    minimum_count: int | None = None
    uniform_count: int | None = None
    if counts.size:
        minimum_count = int(counts.min())
        if minimum_count == 0:
            raise ValueError("every eligibility slice must contain a True entry")

        # Uniform bounds use NumPy's faster scalar-bound generator path. A
        # completely eligible mask additionally needs no ordinal mapping.
        maximum_count = int(counts.max())
        if minimum_count == maximum_count:
            uniform_count = minimum_count

    # Uniform prefix masks map an ordinal directly to the candidate index. The
    # count already proves that no later candidates can also be eligible, so
    # checking only the prefix avoids materializing compressed positions.
    prefix_eligible = not counts.size
    if uniform_count is not None:
        if uniform_count == moved.shape[-1]:
            prefix_eligible = True
        else:
            prefix = moved[..., :uniform_count]
            # Counting wins for very narrow strided prefixes; the Boolean
            # reduction is faster once the prefix contains more columns.
            prefix_eligible = (
                np.count_nonzero(prefix) == counts.size * uniform_count
                if uniform_count <= 4
                else bool(np.all(prefix))
            )

    if prefix_eligible:
        offsets = np.empty(0, dtype=np.intp)
        eligible_positions = np.empty(0, dtype=np.intp)
    else:
        # Store arbitrary masks as compressed rows. An ordinal is mapped by
        # adding its slice offset and indexing this ascending position array,
        # avoiding the candidates-by-output broadcast of a cumulative mask.
        flat_counts = counts.reshape(-1)
        offsets = np.empty_like(flat_counts)
        offsets[0] = 0
        np.cumsum(flat_counts[:-1], dtype=np.intp, out=offsets[1:])
        offsets = offsets.reshape(counts.shape)
        eligible_positions = np.flatnonzero(moved)

    output_shape = list(eligible.shape)
    output_shape[normalized_axis] = normalized_size
    return PreparedSampling(
        moved_eligible=moved,
        counts=counts,
        offsets=offsets,
        eligible_positions=eligible_positions,
        prefix_eligible=prefix_eligible,
        minimum_count=minimum_count,
        uniform_count=uniform_count,
        axis=normalized_axis,
        output_shape=tuple(output_shape),
        moved_output_shape=(*moved.shape[:-1], normalized_size),
    )


def draw_ordinals(
    rng: np.random.Generator,
    prepared: PreparedSampling,
) -> NDArray[np.intp]:
    """Draw only the random ordinals needed by the prepared eligibility."""
    if prepared.counts.size == 0:
        return np.empty(prepared.moved_output_shape, dtype=np.intp)

    sample_size = prepared.moved_output_shape[-1]
    uniform_count = prepared.uniform_count

    if uniform_count == 1:
        # A singleton slice has only one possible result, so consuming random
        # values cannot change its distribution.
        return np.zeros(prepared.moved_output_shape, dtype=np.intp)
    if uniform_count is not None:
        return cast(
            NDArray[np.intp],
            rng.integers(
                0,
                uniform_count,
                size=prepared.moved_output_shape,
                dtype=np.intp,
            ),
        )

    counts = prepared.counts
    if prepared.minimum_count == 1:
        variable = counts > 1
        variable_count = int(np.count_nonzero(variable))
        # Filtering is worthwhile only when it removes at least three quarters
        # of the draws; otherwise Boolean gather/scatter costs as much as the
        # high=1 draws it avoids on interleaved masks.
        if variable_count * 4 <= counts.size:
            ordinals = np.zeros(prepared.moved_output_shape, dtype=np.intp)
            if variable_count:
                ordinals[variable] = rng.integers(
                    0,
                    counts[variable][..., None],
                    size=(variable_count, sample_size),
                    dtype=np.intp,
                )
            return ordinals

    return cast(
        NDArray[np.intp],
        rng.integers(
            0,
            counts[..., None],
            size=prepared.moved_output_shape,
            dtype=np.intp,
        ),
    )


def _validated_ordinals(
    prepared: PreparedSampling,
    ordinals: IntegerArray,
) -> IntegerArray:
    values = np.asarray(ordinals)
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("ordinals must have an integer dtype")
    if values.shape != prepared.moved_output_shape:
        raise ValueError("ordinal shape must match the prepared output shape")
    if np.any(values < 0) or np.any(values >= prepared.counts[..., None]):
        raise ValueError("ordinals must be within each eligibility count")
    return values


def sample_vectorized(
    prepared: PreparedSampling,
    ordinals: IntegerArray,
) -> RollArray:
    """Validate and map varying-range ordinals through compressed positions."""
    validated = _validated_ordinals(prepared, ordinals)
    return sample_vectorized_unchecked(
        prepared,
        validated.astype(np.intp, copy=False),
    )


def sample_vectorized_unchecked(
    prepared: PreparedSampling,
    ordinals: NDArray[np.intp],
) -> RollArray:
    """Map trusted ordinals without repeating generator-guaranteed checks."""
    if prepared.prefix_eligible:
        indices = ordinals
    else:
        positions = prepared.eligible_positions[prepared.offsets[..., None] + ordinals]
        np.remainder(
            positions,
            prepared.moved_eligible.shape[-1],
            out=positions,
        )
        indices = positions
    return np.moveaxis(indices, -1, prepared.axis)


def sample_reference(
    prepared: PreparedSampling,
    ordinals: IntegerArray,
) -> RollArray:
    """Clearly map the same prepared ordinals one eligibility slice at a time."""
    validated = _validated_ordinals(prepared, ordinals)
    result = np.empty(prepared.moved_output_shape, dtype=np.intp)
    # Keep this deliberately direct: it is the readable correctness oracle
    # for the compressed-position transformation above.
    for coordinate in np.ndindex(prepared.moved_eligible.shape[:-1]):
        candidates = np.flatnonzero(prepared.moved_eligible[coordinate])
        result[coordinate] = candidates[validated[coordinate]]
    return np.moveaxis(result, -1, prepared.axis)
