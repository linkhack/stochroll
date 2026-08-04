"""Stable forward benchmark scenarios owned by the WP-009 evidence series."""

from __future__ import annotations

from ._shared import RecordingRNG, Roller
from .wp009_02_callback_runner import dragon_hunt, lantern_run

SCENARIO_CONTRACT_VERSION = 1
IMPLEMENTATION_NAME = "wp009-02-callback-runner"
DRAGON_STATE_PROFILE = "hp-only-event-masked"


def run_lantern_raw(repetitions: int, *, seed: int = 20260729) -> None:
    lantern_run.simulate_lantern_callbacks(repetitions, seed=seed)


def run_lantern_instrumented(
    repetitions: int,
    *,
    seed: int = 20260729,
) -> None:
    roller = Roller(repetitions=repetitions, seed=seed)
    roller.rng = RecordingRNG(seed)  # type: ignore[assignment]
    lantern_run.run_lantern_callbacks(
        roller, lantern_run.initial_lantern_state(repetitions)
    )


def run_dragon_event_masked_raw(
    repetitions: int,
    *,
    seed: int = 20260730,
) -> None:
    dragon_hunt.simulate_dragon_hunt_callbacks(repetitions, seed=seed)


def run_dragon_event_masked_instrumented(
    repetitions: int,
    *,
    seed: int = 20260730,
) -> None:
    roller = Roller(repetitions=repetitions, seed=seed)
    roller.rng = RecordingRNG(seed)  # type: ignore[assignment]
    dragon_hunt.run_dragon_hunt_callbacks(
        roller, dragon_hunt.initial_dragon_hunt_state(repetitions)
    )
