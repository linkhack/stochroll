"""Deterministic, non-timing validation for benchmark setup and metadata."""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from benchmarks._support import (
    KEEP_DROP_BRANCHES,
    LOOKUP_SIZE,
    POOL_DICE,
    REPETITIONS,
    REROLL_DENSITIES,
    STANDARD_WORKLOADS,
    STRUCTURAL_SHAPE,
    build_event,
    build_pool,
    build_pool_lookup_indices,
    build_roll,
    build_roll_lookup_indices,
    build_route_destinations,
)
from benchmarks.reductions import (
    FUSED_LAST_AXIS_SIZES,
    MINMAX_LAST_AXIS_SIZES,
    POOL_SUM_BOUNDARY_CASES,
    SUM_LAST_AXIS_SIZES,
)


def _load_config() -> dict[str, Any]:
    config_path = Path(__file__).parents[1] / "asv.conf.json"
    with config_path.open(encoding="utf-8") as stream:
        return cast(dict[str, Any], json.load(stream))


def check_suite() -> None:
    """Check matrices, native metadata configuration, and seeded inputs."""
    assert REPETITIONS == {"small": 1_000, "large": 250_000}
    assert tuple(workload.name for workload in STANDARD_WORKLOADS) == (
        "small",
        "large",
    )
    assert repr(STANDARD_WORKLOADS[0]) == ("small[R=1000,roll=(R,6),pool=(R,6,12)]")
    assert STRUCTURAL_SHAPE == (6,)
    assert POOL_DICE == 12
    assert LOOKUP_SIZE == 3
    assert KEEP_DROP_BRANCHES == ("direct", "identity", "delegated")
    assert REROLL_DENSITIES == ("none", "sparse", "dense")
    assert SUM_LAST_AXIS_SIZES == (1, 4, 12, 40)
    assert MINMAX_LAST_AXIS_SIZES == (1, 4, 12, 40)
    assert FUSED_LAST_AXIS_SIZES == (2, 4, 12, 40)
    assert POOL_SUM_BOUNDARY_CASES == ((21, 12), (22, 12))

    workload = STANDARD_WORKLOADS[0]
    first_roll = build_roll(workload)
    second_roll = build_roll(workload)
    first_event = build_event(workload)
    second_event = build_event(workload)
    first_pool = build_pool(workload)
    second_pool = build_pool(workload)
    first_roll_indices = build_roll_lookup_indices(workload)
    second_roll_indices = build_roll_lookup_indices(workload)
    first_pool_indices = build_pool_lookup_indices(workload)
    second_pool_indices = build_pool_lookup_indices(workload)
    first_route_destinations = build_route_destinations(workload)
    second_route_destinations = build_route_destinations(workload)

    assert first_roll.values.shape == (1_000, 6)
    assert first_event.values.shape == (1_000, 6)
    assert first_pool.values.shape == (1_000, 6, 12)
    assert first_roll_indices.values.shape == (1_000, 3)
    assert first_pool_indices.shape == (1_000, 3, 12)
    assert first_route_destinations.shape == (1_000, 6)
    np.testing.assert_array_equal(first_roll.values, second_roll.values)
    np.testing.assert_array_equal(first_event.values, second_event.values)
    np.testing.assert_array_equal(first_pool.values, second_pool.values)
    np.testing.assert_array_equal(
        first_roll_indices.values,
        second_roll_indices.values,
    )
    np.testing.assert_array_equal(first_pool_indices, second_pool_indices)
    np.testing.assert_array_equal(
        first_route_destinations,
        second_route_destinations,
    )

    config = _load_config()
    assert config["benchmark_dir"] == "benchmarks"
    assert config["results_dir"] == ".asv/results"
    assert config["env_dir"] == ".asv/env"
    assert config["environment_type"] == "uv"
    assert config["build_command"] == [
        "python -m pip wheel --no-deps -w {build_cache_dir} {build_dir}"
    ]
    assert config["pythons"] == ["3.12", "3.14"]
    assert config["matrix"]["req"]["numpy"] == ["1.26.4", "2.5.1"]
    assert config["matrix"]["env_nobuild"]["STOCHROLL_BENCH_CONFIG"] == ["standard"]
    assert config["exclude"] == [
        {
            "python": "3.14",
            "req": {"numpy": "1.26.4"},
        }
    ]


if __name__ == "__main__":
    check_suite()
