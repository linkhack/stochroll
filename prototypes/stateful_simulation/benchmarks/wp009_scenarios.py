"""Cross-milestone scenario benchmarks owned by the WP-009 umbrella."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import ClassVar, cast

ScenarioRunner = Callable[[int], object]

_PROTOTYPE_ROOT = "stochroll._prototypes"
_STABLE_MODULE = "stochroll._prototypes.stateful_simulation.wp009_scenarios"
_ACTIVE_BATCH_MODULE = (
    "stochroll._prototypes.stateful_simulation.wp009_01_active_batch.reference"
)

_EVENT_MASKED_DRAGON_MODULE = (
    "stochroll._prototypes.stateful_simulation.wp012_event_sampling.dragon_hunt"
)


class _ScenarioUnavailable(ImportError):
    """The requested runner does not exist in the benchmarked revision."""


def _unavailable(name: str, import_error: ImportError) -> ScenarioRunner:
    from asv_runner.benchmarks.mark import (  # type: ignore[import-untyped]
        SkipNotImplemented,
    )

    def run(_repetitions: int) -> None:
        raise SkipNotImplemented(
            f"{name} is unavailable in this revision"
        ) from import_error

    return run


def _target_module_is_missing(
    error: ModuleNotFoundError,
    module_name: str,
) -> bool:
    """Return whether ``module_name`` or one of its prototype parents is absent."""
    missing_name = error.name
    return bool(
        missing_name
        and missing_name.startswith(_PROTOTYPE_ROOT)
        and (missing_name == module_name or module_name.startswith(f"{missing_name}."))
    )


def _load_runner(module_name: str, runner_name: str) -> ScenarioRunner:
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        if not _target_module_is_missing(error, module_name):
            raise
        raise _ScenarioUnavailable(f"module {module_name!r} is unavailable") from error

    try:
        runner = getattr(module, runner_name)
    except AttributeError as error:
        raise _ScenarioUnavailable(
            f"runner {module_name}.{runner_name} is unavailable"
        ) from error

    if not callable(runner):
        raise TypeError(f"{module_name}.{runner_name} is not callable")
    return cast(ScenarioRunner, runner)


def _load_optional_runner(
    module_name: str,
    runner_name: str,
    *,
    label: str,
) -> ScenarioRunner:
    try:
        return _load_runner(module_name, runner_name)
    except _ScenarioUnavailable as error:
        return _unavailable(label, error)


def _load_runner_with_fallback(
    current_name: str,
    *,
    fallback_module: str,
    fallback_name: str,
    label: str,
) -> ScenarioRunner:
    try:
        return _load_runner(_STABLE_MODULE, current_name)
    except _ScenarioUnavailable:
        return _load_optional_runner(
            fallback_module,
            fallback_name,
            label=label,
        )


def _get_lantern_instrumented() -> ScenarioRunner:
    return _load_runner_with_fallback(
        "run_lantern_instrumented",
        fallback_module=_ACTIVE_BATCH_MODULE,
        fallback_name="lantern_run_active",
        label="WP-009-01 instrumented Lantern Run",
    )


def _get_dragon_event_masked_instrumented() -> ScenarioRunner:
    return _load_runner_with_fallback(
        "run_dragon_event_masked_instrumented",
        fallback_module=_EVENT_MASKED_DRAGON_MODULE,
        fallback_name="dragon_hunt_event_masked",
        label="WP-012 instrumented event-masked Dragon Hunt",
    )


def _get_dragon_fixed_slot_instrumented() -> ScenarioRunner:
    return _load_optional_runner(
        _ACTIVE_BATCH_MODULE,
        "dragon_hunt_active",
        label="WP-009-01 fixed-slot Dragon Hunt",
    )


def _get_lantern_raw() -> ScenarioRunner:
    return _load_optional_runner(
        _STABLE_MODULE,
        "run_lantern_raw",
        label="WP-009-02 raw Lantern Run",
    )


def _get_dragon_event_masked_raw() -> ScenarioRunner:
    return _load_optional_runner(
        _STABLE_MODULE,
        "run_dragon_event_masked_raw",
        label="WP-009-02 raw event-masked Dragon Hunt",
    )


class StatefulScenarioHistory:
    params: ClassVar[list[int]] = [2_000, 10_000, 100_000]
    param_names: ClassVar[list[str]] = ["repetitions"]

    run_lantern_instrumented: ScenarioRunner
    run_dragon_event_masked_instrumented: ScenarioRunner
    run_dragon_fixed_slot_instrumented: ScenarioRunner
    run_lantern_raw: ScenarioRunner
    run_dragon_event_masked_raw: ScenarioRunner

    def setup(self, repetitions: int) -> None:
        del repetitions
        self.run_lantern_instrumented = _get_lantern_instrumented()
        self.run_dragon_event_masked_instrumented = (
            _get_dragon_event_masked_instrumented()
        )
        self.run_dragon_fixed_slot_instrumented = _get_dragon_fixed_slot_instrumented()
        self.run_lantern_raw = _get_lantern_raw()
        self.run_dragon_event_masked_raw = _get_dragon_event_masked_raw()

    def time_lantern_instrumented(self, repetitions: int) -> None:
        self.run_lantern_instrumented(repetitions)

    def time_dragon_event_masked_instrumented(self, repetitions: int) -> None:
        self.run_dragon_event_masked_instrumented(repetitions)

    def time_dragon_fixed_slot_instrumented(self, repetitions: int) -> None:
        self.run_dragon_fixed_slot_instrumented(repetitions)

    def time_lantern_raw(self, repetitions: int) -> None:
        self.run_lantern_raw(repetitions)

    def time_dragon_event_masked_raw(self, repetitions: int) -> None:
        self.run_dragon_event_masked_raw(repetitions)
