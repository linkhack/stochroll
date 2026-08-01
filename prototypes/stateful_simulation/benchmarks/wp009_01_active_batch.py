from __future__ import annotations

from typing import ClassVar

import numpy as np

from stochroll import Event
from stochroll._prototypes.stateful_simulation._shared import Roller
from stochroll._prototypes.stateful_simulation.wp009_01_active_batch.reference import (
    dragon_hunt_active,
    dragon_hunt_dense,
    lantern_run_active,
    lantern_run_dense,
)


class ActiveFraction:
    params: ClassVar[list[float]] = [1.0, 0.75, 0.25, 0.01, 0.0]
    param_names: ClassVar[list[str]] = ["active_fraction"]

    def time_dense_draw(self, active_fraction: float) -> None:
        roller = Roller(repetitions=10_000, seed=42)
        roller.d(20, shape=4)

    def time_packed_draw(self, active_fraction: float) -> None:
        repetitions = 10_000
        active_count = round(repetitions * active_fraction)
        values = np.zeros(repetitions, dtype=np.bool_)
        values[:active_count] = True
        roller = Roller(repetitions=repetitions, seed=42)
        batch = roller.active_batch(Event(values))
        if batch is not None:
            batch.d(20, shape=4)


class ReferenceScenarios:
    params: ClassVar[list[int]] = [2_000, 10_000, 100_000]
    param_names: ClassVar[list[str]] = ["repetitions"]

    def time_lantern_run_dense(self, repetitions: int) -> None:
        lantern_run_dense(repetitions)

    def time_lantern_run_active(self, repetitions: int) -> None:
        lantern_run_active(repetitions)

    def time_dragon_hunt_dense(self, repetitions: int) -> None:
        dragon_hunt_dense(repetitions)

    def time_dragon_hunt_active(self, repetitions: int) -> None:
        dragon_hunt_active(repetitions)
