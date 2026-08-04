"""Direct generic callback runner layered on the shared ActiveBatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from stochroll import Event, Roll

from .._shared import ActiveBatch, Roller, validate_max_steps


class StateAdapter[StateT](Protocol):
    """Explicitly pack and merge every repetition-indexed state field."""

    def take(self, state: StateT, batch: ActiveBatch, /) -> StateT: ...

    def merge(
        self,
        base: StateT,
        update: StateT,
        batch: ActiveBatch,
        /,
    ) -> StateT: ...


type ActivityCallback[StateT] = Callable[[StateT, int], Event]
type StepCallback[StateT] = Callable[[StateT, ActiveBatch, int], StateT]


@dataclass(frozen=True, slots=True)
class SimulationResult[StateT]:
    """Dense final or partial state and its termination metadata."""

    state: StateT
    steps: int
    termination_step: Roll


class SimulationLimitExceeded[StateT](RuntimeError):
    """Raised with recoverable partial state when the bound is exhausted."""

    def __init__(
        self,
        result: SimulationResult[StateT],
        active: Event,
    ) -> None:
        message = f"simulation still has active repetitions after {result.steps} steps"
        super().__init__(message)
        self.result = result
        self.active = active


def _validated_activity(activity: object, repetitions: int) -> Event:
    if not isinstance(activity, Event):
        raise TypeError("is_active must return an Event")
    if activity.values.shape != (repetitions,):
        raise ValueError(
            "is_active must return exact shape "
            f"({repetitions},), got {activity.values.shape}"
        )
    return activity


def run_simulation[StateT](
    roller: Roller,
    initial_state: StateT,
    *,
    adapter: StateAdapter[StateT],
    is_active: ActivityCallback[StateT],
    step: StepCallback[StateT],
    max_steps: int,
) -> SimulationResult[StateT]:
    """Run compact transitions until every repetition terminates or time runs out."""
    limit = validate_max_steps(max_steps)
    state = initial_state
    active = _validated_activity(is_active(state, 0), roller.repetitions)
    termination = np.full(roller.repetitions, -1, dtype=np.int64)
    termination[~active.values] = 0

    for transition_index in range(limit):
        batch = roller.active_batch(active)
        if batch is None:
            return SimulationResult(state, transition_index, Roll(termination))

        compact_state = adapter.take(state, batch)
        compact_update = step(compact_state, batch, transition_index)
        state = adapter.merge(state, compact_update, batch)

        next_active = _validated_activity(
            is_active(state, transition_index + 1),
            roller.repetitions,
        )
        if np.any(next_active.values & ~active.values):
            raise ValueError(
                "activity must be monotonic; repetitions cannot reactivate"
            )

        newly_terminal = active.values & ~next_active.values
        termination[newly_terminal] = transition_index + 1
        active = next_active

    result = SimulationResult(state, limit, Roll(termination))
    if np.any(active.values):
        raise SimulationLimitExceeded(result, active)
    return result
