"""Focused comparisons for interchangeable routing implementations.

The candidate benchmarks receive the same validated arrays and metadata. They
do not perform correctness checks; equivalence belongs to ``tests/``.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from stochroll._reductions import _default_sum_dtype
from stochroll._routing import (
    _route_any_indexed,
    _route_reference_any,
    _route_reference_sum,
    _route_sum_indexed,
)
from stochroll.core import _prepare_route_inputs

ROUTE_IMPLEMENTATIONS: tuple[str, ...] = ("indexed",)
ROUTE_SEED = 47


@dataclass(frozen=True, slots=True)
class RouteWorkload:
    """A routing shape with repetitions scaled to bound mask memory use."""

    name: str
    repetitions: int
    structural_shape: tuple[int, ...]
    size: int
    axis: int

    def __repr__(self) -> str:
        source_shape = (self.repetitions, *self.structural_shape)
        return f"{self.name}[source={source_shape},size={self.size},axis={self.axis}]"


ROUTE_WORKLOADS = (
    RouteWorkload(
        name="scalar",
        repetitions=250_000,
        structural_shape=(),
        size=4,
        axis=-1,
    ),
    RouteWorkload(
        name="one_axis_narrow",
        repetitions=250_000,
        structural_shape=(6,),
        size=4,
        axis=-1,
    ),
    RouteWorkload(
        name="one_axis_wide",
        repetitions=25_000,
        structural_shape=(40,),
        size=12,
        axis=-1,
    ),
    RouteWorkload(
        name="two_axes_trailing",
        repetitions=50_000,
        structural_shape=(6, 8),
        size=6,
        axis=-1,
    ),
    RouteWorkload(
        name="two_axes_interior",
        repetitions=50_000,
        structural_shape=(8, 6),
        size=4,
        axis=1,
    ),
    RouteWorkload(
        name="three_axes_interior",
        repetitions=20_000,
        structural_shape=(3, 4, 5),
        size=8,
        axis=2,
    ),
)


class RouteCandidates:
    """Compare routing candidates on identical prepared arrays."""

    params: ClassVar = [ROUTE_WORKLOADS, ROUTE_IMPLEMENTATIONS]
    param_names: ClassVar = ["workload", "implementation"]

    def setup(self, workload: RouteWorkload, implementation: str) -> None:
        rng = np.random.default_rng(ROUTE_SEED)
        source_shape = (workload.repetitions, *workload.structural_shape)
        roll_values = rng.integers(1, 21, size=source_shape, dtype=np.int16)
        event_values = roll_values >= 11
        destinations = rng.integers(
            0,
            workload.size,
            size=source_shape,
            dtype=np.int16,
        )
        self.roll_values, self.destinations, self.axis, self.size = (
            _prepare_route_inputs(
                roll_values,
                destinations,
                size=workload.size,
                axis=workload.axis,
            )
        )
        self.event_values, _, _, _ = _prepare_route_inputs(
            event_values,
            destinations,
            size=workload.size,
            axis=workload.axis,
        )
        sum_implementations: dict[str, Any] = {
            "reference": _route_reference_sum,
            "indexed": _route_sum_indexed,
        }
        any_implementations: dict[str, Any] = {
            "reference": _route_reference_any,
            "indexed": _route_any_indexed,
        }
        self.sum_implementation = sum_implementations[implementation]
        self.any_implementation = any_implementations[implementation]
        self.dtype = _default_sum_dtype(roll_values.dtype)

    def time_route_sum(
        self,
        workload: RouteWorkload,
        implementation: str,
    ) -> None:
        self.sum_implementation(
            self.roll_values,
            self.destinations,
            size=self.size,
            axis=self.axis,
            dtype=self.dtype,
        )

    def time_route_any(
        self,
        workload: RouteWorkload,
        implementation: str,
    ) -> None:
        self.any_implementation(
            self.event_values,
            self.destinations,
            size=self.size,
            axis=self.axis,
        )
