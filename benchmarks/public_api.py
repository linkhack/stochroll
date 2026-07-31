"""Realistic end-to-end benchmarks for the public StochRoll API.

Every class uses the standard ``small`` and ``large`` workloads. Roll and
Event inputs have shape ``(R, 6)``; Pool inputs have shape ``(R, 6, 12)``.
Setup prepares deterministic inputs and each timed method contains one public
operation.
"""

from copy import deepcopy
from typing import ClassVar

from benchmarks._support import (
    KEEP_DROP_BRANCHES,
    REROLL_DENSITIES,
    STANDARD_WORKLOADS,
    WorkloadSpec,
    build_event,
    build_pool,
    build_pool_lookup_indices,
    build_roll,
    build_roll_lookup_indices,
    build_route_destinations,
    make_roller,
)
from stochroll import Event, Pool, Roll, Roller, concatenate, stack, where


class RollerMethods:
    """Measure public simulation construction and draw methods."""

    params: ClassVar = [STANDARD_WORKLOADS]
    param_names: ClassVar = ["workload"]

    def setup(self, workload: WorkloadSpec) -> None:
        self.repetitions = workload.repetitions
        self.roller = make_roller(workload)

    def time_construct(self, workload: WorkloadSpec) -> None:
        Roller(repetitions=self.repetitions, seed=42)

    def time_d(self, workload: WorkloadSpec) -> None:
        self.roller.d(20, shape=6)

    def time_pool(self, workload: WorkloadSpec) -> None:
        self.roller.pool(12, d=6, shape=6)


class RollOperators:
    """Measure arithmetic and comparison operations on shaped Rolls."""

    params: ClassVar = [STANDARD_WORKLOADS]
    param_names: ClassVar = ["workload"]

    def setup(self, workload: WorkloadSpec) -> None:
        self.left = build_roll(workload)
        self.right = build_roll(workload, sides=12)

    def time_add(self, workload: WorkloadSpec) -> None:
        self.left + self.right

    def time_add_scalar(self, workload: WorkloadSpec) -> None:
        self.left + self.right

    def time_radd(self, workload: WorkloadSpec) -> None:
        3 + self.left

    def time_reflected_subtract(self, workload: WorkloadSpec) -> None:
        30 - self.left

    def time_equal(self, workload: WorkloadSpec) -> Event:
        return self.left == self.right

    def time_greater_than(self, workload: WorkloadSpec) -> Event:
        return self.left > self.right


class RollMethods:
    """Measure public structural and reduction Roll methods."""

    params: ClassVar = [STANDARD_WORKLOADS]
    param_names: ClassVar = ["workload"]

    def setup(self, workload: WorkloadSpec) -> None:
        self.roll = build_roll(workload)
        self.scalar_roll = make_roller(workload, seed=43).d(20)
        self.lookup_indices = build_roll_lookup_indices(workload)
        self.select_indices = (0, 2, 4)

    def time_select(self, workload: WorkloadSpec) -> None:
        self.roll.select(self.select_indices)

    def time_lookup(self, workload: WorkloadSpec) -> None:
        self.roll.lookup(self.lookup_indices)

    def time_add_axis(self, workload: WorkloadSpec) -> None:
        self.roll.add_axis()

    def time_sum(self, workload: WorkloadSpec) -> None:
        self.roll.sum()

    def time_mean(self, workload: WorkloadSpec) -> None:
        self.roll.mean()

    def time_min(self, workload: WorkloadSpec) -> None:
        self.roll.min()

    def time_max(self, workload: WorkloadSpec) -> None:
        self.roll.max()

    def time_broadcast_to(self, workload: WorkloadSpec) -> None:
        self.scalar_roll.broadcast_to(6)


class EventMethods:
    """Measure Boolean, structural, and reduction Event methods."""

    params: ClassVar = [STANDARD_WORKLOADS]
    param_names: ClassVar = ["workload"]

    def setup(self, workload: WorkloadSpec) -> None:
        self.left = build_event(workload)
        self.right = build_roll(workload, sides=12) >= 6
        self.scalar_event = make_roller(workload, seed=43).d(20) >= 11
        self.lookup_indices = build_roll_lookup_indices(workload)
        self.select_indices = (0, 2, 4)

    def time_or(self, workload: WorkloadSpec) -> None:
        self.left | self.right

    def time_and(self, workload: WorkloadSpec) -> None:
        self.left & self.right

    def time_invert(self, workload: WorkloadSpec) -> Event:
        return ~self.left

    def time_select(self, workload: WorkloadSpec) -> None:
        self.left.select(self.select_indices)

    def time_lookup(self, workload: WorkloadSpec) -> None:
        self.left.lookup(self.lookup_indices)

    def time_add_axis(self, workload: WorkloadSpec) -> None:
        self.left.add_axis()

    def time_broadcast_to(self, workload: WorkloadSpec) -> None:
        self.scalar_event.broadcast_to(6)

    def time_count(self, workload: WorkloadSpec) -> None:
        self.left.count()

    def time_indicator(self, workload: WorkloadSpec) -> None:
        self.left.indicator()


class StatisticsMethods:
    """Measure public statistical summaries on prepared Roll and Event values."""

    params: ClassVar = [STANDARD_WORKLOADS]
    param_names: ClassVar = ["workload"]

    def setup(self, workload: WorkloadSpec) -> None:
        self.roll = build_roll(workload)
        self.event = build_event(workload)

    def time_expected(self, workload: WorkloadSpec) -> None:
        self.roll.expected()

    def time_probability_at_least(self, workload: WorkloadSpec) -> None:
        self.roll.probability_at_least(11)

    def time_probability_at_most(self, workload: WorkloadSpec) -> None:
        self.roll.probability_at_most(11)

    def time_variance(self, workload: WorkloadSpec) -> None:
        self.roll.variance()

    def time_standard_deviation(self, workload: WorkloadSpec) -> None:
        self.roll.standard_deviation()

    def time_quantile(self, workload: WorkloadSpec) -> None:
        self.roll.quantile(0.5)

    def time_event_probability(self, workload: WorkloadSpec) -> None:
        self.event.probability()


class RoutingMethods:
    """Measure the public routing methods with the active backend."""

    params: ClassVar = [STANDARD_WORKLOADS]
    param_names: ClassVar = ["workload"]

    def setup(self, workload: WorkloadSpec) -> None:
        self.roll = build_roll(workload)
        self.event = build_event(workload)
        self.destinations = build_route_destinations(workload)

    def time_route_sum(self, workload: WorkloadSpec) -> None:
        self.roll.route_sum(self.destinations, size=6)

    def time_route_multipy(self, workload: WorkloadSpec) -> None:
        self.roll.route_multiply(self.destinations, size=6)

    def time_route_any(self, workload: WorkloadSpec) -> None:
        self.event.route_any(self.destinations, size=6)

    def time_route_all(self, workload: WorkloadSpec) -> None:
        self.event.route_all(self.destinations, size=6)


class PoolMethods:
    """Measure public structural, resolving, and fused Pool methods."""

    params: ClassVar = [STANDARD_WORKLOADS]
    param_names: ClassVar = ["workload"]

    def setup(self, workload: WorkloadSpec) -> None:
        self.pool = build_pool(workload)
        self.single_pool = build_pool(workload, dice=1)
        self.lookup_indices = build_pool_lookup_indices(workload)
        self.select_indices = (0, 2, 4)

    def time_select(self, workload: WorkloadSpec) -> None:
        self.pool.select(self.select_indices)

    def time_lookup(self, workload: WorkloadSpec) -> None:
        self.pool.lookup(self.lookup_indices)

    def time_first(self, workload: WorkloadSpec) -> None:
        self.pool.first()

    def time_single(self, workload: WorkloadSpec) -> None:
        self.single_pool.single()

    def time_sum(self, workload: WorkloadSpec) -> None:
        self.pool.sum()

    def time_min(self, workload: WorkloadSpec) -> None:
        self.pool.min()

    def time_max(self, workload: WorkloadSpec) -> None:
        self.pool.max()

    def time_drop_lowest_sum(self, workload: WorkloadSpec) -> None:
        self.pool.drop_lowest_sum()

    def time_count_at_least(self, workload: WorkloadSpec) -> None:
        self.pool.count_at_least(5)


class PoolKeepDropBranches:
    """Cover direct, identity, and delegated keep/drop branches."""

    params: ClassVar = [
        STANDARD_WORKLOADS,
        KEEP_DROP_BRANCHES,
    ]
    param_names: ClassVar = ["workload", "branch"]

    def setup(self, workload: WorkloadSpec, branch: str) -> None:
        self.pool = build_pool(workload)
        self.k = {
            "direct": 3,
            "identity": 12,
            "delegated": 9,
        }[branch]
        self.drop_k = {
            "direct": 3,
            "identity": 0,
            "delegated": 9,
        }[branch]

    def time_keep_highest(self, workload: WorkloadSpec, branch: str) -> None:
        self.pool.keep_highest(self.k)

    def time_drop_highest(self, workload: WorkloadSpec, branch: str) -> None:
        self.pool.drop_highest(self.drop_k)


class PoolRerollOnce:
    """Cover no-match, sparse-match, and dense-match rerolling."""

    params: ClassVar = [
        STANDARD_WORKLOADS,
        REROLL_DENSITIES,
    ]
    param_names: ClassVar = ["workload", "match_density"]
    number = 1

    def setup(self, workload: WorkloadSpec, match_density: str) -> None:
        self.pool = build_pool(workload)
        self.targets = {
            "none": (6,),
            "sparse": (1,),
            "dense": (1, 2, 3, 4, 5),
        }[match_density]
        if match_density == "none":
            template = build_pool(workload)
            self.pool = deepcopy(template)
            self.pool.values[self.pool.values == 6] = 5

    def time_reroll_once(
        self,
        workload: WorkloadSpec,
        match_density: str,
    ) -> None:
        self.pool.reroll_once(self.targets)


class PublicFunctions:
    """Measure public conditional and homogeneous assembly functions."""

    params: ClassVar = [STANDARD_WORKLOADS]
    param_names: ClassVar = ["workload"]

    def setup(self, workload: WorkloadSpec) -> None:
        self.rolls: tuple[Roll, Roll] = (
            build_roll(workload),
            build_roll(workload, sides=12),
        )
        self.events: tuple[Event, Event] = (
            self.rolls[0] >= 11,
            self.rolls[1] >= 6,
        )
        self.pools: tuple[Pool, Pool] = (
            build_pool(workload),
            build_pool(workload),
        )

    def time_where(self, workload: WorkloadSpec) -> None:
        where(self.events[0], self.rolls[0], self.rolls[1])

    def time_stack_roll(self, workload: WorkloadSpec) -> None:
        stack(self.rolls)

    def time_stack_event(self, workload: WorkloadSpec) -> None:
        stack(self.events)

    def time_stack_pool(self, workload: WorkloadSpec) -> None:
        stack(self.pools)

    def time_concatenate_roll(self, workload: WorkloadSpec) -> None:
        concatenate(self.rolls)

    def time_concatenate_event(self, workload: WorkloadSpec) -> None:
        concatenate(self.events)

    def time_concatenate_pool(self, workload: WorkloadSpec) -> None:
        concatenate(self.pools)
