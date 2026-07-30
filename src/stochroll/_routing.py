"""Interchangeable, validated-array routing implementations.

The public wrapper methods in :mod:`stochroll.core` validate and canonicalize
all inputs before calling these functions. Keeping the candidates array-only
makes it possible to benchmark them on identical prepared inputs and to
replace the active backend without changing public validation semantics.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._typing import BooleanArray, IntegerArray, NumericScalar

type RouteInitial = NumericScalar | bool | np.bool_


def _route_ufunc_indexed[
    SourceDType: np.generic,
    OutputDType: np.generic,
](
    values: NDArray[SourceDType],
    destinations: IntegerArray,
    *,
    size: int,
    axis: int,
    dtype: np.dtype[OutputDType],
    operation: np.ufunc,
    initial: RouteInitial,
) -> NDArray[OutputDType]:
    """Accumulate routed values using an unbuffered binary ufunc."""
    moved_values = np.moveaxis(values, axis, -1)
    moved_destinations = np.moveaxis(destinations, axis, -1)

    output_shape = (*moved_values.shape[:-1], size)
    output = (
        np.zeros(output_shape, dtype=dtype)
        if initial == 0
        else np.full(output_shape, initial, dtype=dtype)
    )

    width = moved_values.shape[-1]
    if output.size == 0 or width == 0:
        return np.moveaxis(output, -1, axis)

    flat_output = output.reshape(-1, size)
    flat_values = moved_values.astype(dtype, copy=False).reshape(-1, width)
    flat_destinations = moved_destinations.reshape(-1, width)

    row_indices = np.arange(flat_output.shape[0], dtype=np.intp)[:, None]
    operation.at(
        flat_output,
        (row_indices, flat_destinations),
        flat_values,
    )

    return np.moveaxis(output, -1, axis)


def _route_sum_indexed(
    values: NDArray[Any],
    destinations: IntegerArray,
    *,
    size: int,
    axis: int,
    dtype: np.dtype[Any],
) -> NDArray[Any]:
    """Accumulate routed values with duplicate-safe indexed writes."""
    return _route_ufunc_indexed(
        values,
        destinations,
        size=size,
        axis=axis,
        dtype=dtype,
        operation=np.add,
        initial=0,
    )


def _route_multiply_indexed(
    values: NDArray[Any],
    destinations: IntegerArray,
    *,
    size: int,
    axis: int,
    dtype: np.dtype[Any],
) -> NDArray[Any]:
    """Accumulate routed values with duplicate-safe indexed products."""
    return _route_ufunc_indexed(
        values,
        destinations,
        size=size,
        axis=axis,
        dtype=dtype,
        operation=np.multiply,
        initial=1,
    )


def _route_any_indexed(
    values: BooleanArray,
    destinations: IntegerArray,
    *,
    size: int,
    axis: int,
) -> BooleanArray:
    """Combine routed events with duplicate-safe indexed logical OR writes."""
    return _route_ufunc_indexed(
        values,
        destinations,
        size=size,
        axis=axis,
        dtype=np.dtype(np.bool_),
        operation=np.logical_or,
        initial=False,
    )


def _route_all_indexed(
    values: BooleanArray,
    destinations: IntegerArray,
    *,
    size: int,
    axis: int,
) -> BooleanArray:
    """Combine routed events with duplicate-safe indexed logical AND writes."""
    return _route_ufunc_indexed(
        values,
        destinations,
        size=size,
        axis=axis,
        dtype=np.dtype(np.bool_),
        operation=np.logical_and,
        initial=True,
    )


def _route_reference_sum(
    values: NDArray[Any],
    destinations: IntegerArray,
    *,
    size: int,
    axis: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
    moved_values = np.moveaxis(values, axis, -1)
    moved_destinations = np.moveaxis(destinations, axis, -1)
    output = np.zeros((*moved_values.shape[:-1], size), dtype=dtype)
    for row in np.ndindex(output.shape[:-1]):
        for source_index in range(moved_values.shape[-1]):
            destination_index = moved_destinations[(*row, source_index)]
            output[(*row, destination_index)] += moved_values[(*row, source_index)]
    return np.moveaxis(output, -1, axis)


def _route_reference_multiply(
    values: NDArray[Any],
    destinations: IntegerArray,
    *,
    size: int,
    axis: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
    moved_values = np.moveaxis(values, axis, -1)
    moved_destinations = np.moveaxis(destinations, axis, -1)
    output = np.ones((*moved_values.shape[:-1], size), dtype=dtype)
    for row in np.ndindex(output.shape[:-1]):
        for source_index in range(moved_values.shape[-1]):
            destination_index = moved_destinations[(*row, source_index)]
            output[(*row, destination_index)] *= moved_values[(*row, source_index)]
    return np.moveaxis(output, -1, axis)


def _route_reference_any(
    values: NDArray[Any],
    destinations: IntegerArray,
    *,
    size: int,
    axis: int,
) -> np.ndarray:
    moved_values = np.moveaxis(values, axis, -1)
    moved_destinations = np.moveaxis(destinations, axis, -1)
    output = np.zeros((*moved_values.shape[:-1], size), dtype=np.bool_)
    for row in np.ndindex(output.shape[:-1]):
        for source_index in range(moved_values.shape[-1]):
            destination_index = moved_destinations[(*row, source_index)]
            output[(*row, destination_index)] |= moved_values[(*row, source_index)]
    return np.moveaxis(output, -1, axis)


def _route_reference_all(
    values: NDArray[Any],
    destinations: IntegerArray,
    *,
    size: int,
    axis: int,
) -> np.ndarray:
    moved_values = np.moveaxis(values, axis, -1)
    moved_destinations = np.moveaxis(destinations, axis, -1)
    output = np.ones((*moved_values.shape[:-1], size), dtype=np.bool_)
    for row in np.ndindex(output.shape[:-1]):
        for source_index in range(moved_values.shape[-1]):
            destination_index = moved_destinations[(*row, source_index)]
            output[(*row, destination_index)] &= moved_values[(*row, source_index)]
    return np.moveaxis(output, -1, axis)
