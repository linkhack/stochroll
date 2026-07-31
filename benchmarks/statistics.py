"""Opt-in benchmarks for the public quantile method matrix."""

from typing import ClassVar

from benchmarks._support import STANDARD_WORKLOADS, WorkloadSpec, make_roller
from stochroll import Roll

QUANTILE_METHODS = (
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
)
SHAPES = ((), (6,))


class QuantileMethods:
    """Compare every supported quantile method on prepared Roll inputs."""

    params: ClassVar = [QUANTILE_METHODS, STANDARD_WORKLOADS, SHAPES]
    param_names: ClassVar = ["method", "workload", "shape"]

    def setup(
        self, method: str, workload: WorkloadSpec, shape: tuple[int, ...]
    ) -> None:
        self.roll: Roll = make_roller(workload, seed=42).d(20, shape=shape or None)
        self.method = method

    def time_quantile(
        self, method: str, workload: WorkloadSpec, shape: tuple[int, ...]
    ) -> None:
        self.roll.quantile(0.5, method=method)
