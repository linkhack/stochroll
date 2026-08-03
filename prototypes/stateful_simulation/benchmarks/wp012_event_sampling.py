from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, cast

import numpy as np
from numpy.typing import NDArray

from stochroll import Event
from stochroll._prototypes.stateful_simulation._shared import Roller
from stochroll._prototypes.stateful_simulation.wp009_01_active_batch.reference import (
    dragon_hunt_active,
)
from stochroll._prototypes.stateful_simulation.wp012_event_sampling import (
    sampling as sampling_backend,
)
from stochroll._prototypes.stateful_simulation.wp012_event_sampling.dragon_hunt import (
    dragon_hunt_event_masked,
)
from stochroll._prototypes.stateful_simulation.wp012_event_sampling.sampling import (
    PreparedSampling,
    prepare_sampling,
    sample_reference,
    sample_vectorized,
)

type _OrdinalDrawer = Callable[
    [np.random.Generator, PreparedSampling], NDArray[np.intp]
]
type _OrdinalMapper = Callable[[PreparedSampling, NDArray[np.intp]], NDArray[np.intp]]


def _legacy_draw_ordinals(
    rng: np.random.Generator,
    prepared: PreparedSampling,
) -> NDArray[np.intp]:
    """Reproduce the draw operation used before the optimized helper existed."""
    return cast(
        NDArray[np.intp],
        rng.integers(
            0,
            prepared.counts[..., None],
            size=prepared.moved_output_shape,
            dtype=np.intp,
        ),
    )


# ASV loads this benchmark suite once and installs each requested project
# revision underneath it. Resolve optional optimized entry points at import
# time so the same suite measures both the previous and current sampler.
_draw_ordinals = cast(
    _OrdinalDrawer,
    getattr(sampling_backend, "draw_ordinals", _legacy_draw_ordinals),
)
_internal_mapper = cast(
    _OrdinalMapper,
    getattr(sampling_backend, "sample_vectorized_unchecked", sample_vectorized),
)


class EventSampling:
    params: ClassVar[list[list[str] | list[int]]] = [
        ["final", "interior"],
        ["100%", "75%", "25%", "one"],
        [1, 3],
    ]
    param_names: ClassVar[list[str]] = ["axis_kind", "eligible_fraction", "size"]

    def setup(self, axis_kind: str, eligible_fraction: str, size: int) -> None:
        # Exercise both the common final-axis layout and the extra moveaxis work
        # required when candidates occupy an interior structural axis.
        shape: tuple[int, ...]
        if axis_kind == "final":
            shape = (2_048, 32)
            axis = -1
        else:
            shape = (256, 16, 32)
            axis = 1

        selected_extent = shape[axis]
        eligible_count = {
            "100%": selected_extent,
            "75%": max(1, selected_extent * 3 // 4),
            "25%": max(1, selected_extent // 4),
            "one": 1,
        }[eligible_fraction]
        eligibility = np.zeros(shape, dtype=np.bool_)
        selection = [slice(None)] * len(shape)
        selection[axis] = slice(0, eligible_count)
        eligibility[tuple(selection)] = True

        # Preparation and ordinal creation happen here so the two mapping
        # benchmarks consume identical inputs and measure mapping only.
        self.prepared: PreparedSampling = prepare_sampling(
            eligibility,
            repetitions=shape[0],
            size=size,
            axis=axis,
        )
        rng = np.random.default_rng(20260803)
        self.ordinals = rng.integers(
            0,
            self.prepared.counts[..., None],
            size=self.prepared.moved_output_shape,
            dtype=np.intp,
        )
        self.eligibility = Event(eligibility)
        self.roller = Roller(repetitions=shape[0], seed=20260803)
        batch = self.roller.active_batch(Event(np.ones(shape[0], dtype=np.bool_)))
        assert batch is not None
        self.batch = batch

    def time_prepare(
        self,
        axis_kind: str,
        eligible_fraction: str,
        size: int,
    ) -> None:
        prepare_sampling(
            self.eligibility.values,
            repetitions=self.eligibility.values.shape[0],
            size=size,
            axis=self.prepared.axis,
        )

    def time_ordinal_generation(
        self,
        axis_kind: str,
        eligible_fraction: str,
        size: int,
    ) -> None:
        _draw_ordinals(self.roller.rng, self.prepared)

    def time_end_to_end(
        self,
        axis_kind: str,
        eligible_fraction: str,
        size: int,
    ) -> None:
        self.batch.sample_indices(
            self.eligibility,
            size=size,
            axis=self.prepared.axis,
        )

    def time_vectorized(
        self,
        axis_kind: str,
        eligible_fraction: str,
        size: int,
    ) -> None:
        sample_vectorized(self.prepared, self.ordinals)

    def time_internal_mapping(
        self,
        axis_kind: str,
        eligible_fraction: str,
        size: int,
    ) -> None:
        _internal_mapper(self.prepared, self.ordinals)

    def time_reference(
        self,
        axis_kind: str,
        eligible_fraction: str,
        size: int,
    ) -> None:
        sample_reference(self.prepared, self.ordinals)


class CandidateExtent:
    params: ClassVar[list[list[int] | list[str]]] = [
        [4, 16, 64, 256],
        ["100%", "25%", "one"],
        [1, 3],
    ]
    param_names: ClassVar[list[str]] = [
        "candidate_extent",
        "eligible_fraction",
        "size",
    ]

    def setup(
        self,
        candidate_extent: int,
        eligible_fraction: str,
        size: int,
    ) -> None:
        # Keep the number of independent slices fixed so preparation and
        # mapping expose candidate-axis and output-size scaling separately.
        repetitions = 8_192
        eligible_count = {
            "100%": candidate_extent,
            "25%": max(1, candidate_extent // 4),
            "one": 1,
        }[eligible_fraction]
        eligibility = np.zeros((repetitions, candidate_extent), dtype=np.bool_)
        eligibility[:, :eligible_count] = True
        self.eligibility = eligibility
        self.prepared = prepare_sampling(
            eligibility,
            repetitions=repetitions,
            size=size,
            axis=-1,
        )
        self.ordinals = np.random.default_rng(20260803).integers(
            0,
            self.prepared.counts[..., None],
            size=self.prepared.moved_output_shape,
            dtype=np.intp,
        )

        # Retain explicit coverage of the compressed fallback now that the
        # established prefix masks can map ordinals directly.
        scattered = np.zeros_like(eligibility)
        step = candidate_extent // eligible_count
        scattered[:, step - 1 :: step] = True
        self.scattered_eligibility = scattered
        self.scattered_prepared = prepare_sampling(
            scattered,
            repetitions=repetitions,
            size=size,
            axis=-1,
        )
        self.scattered_ordinals = np.random.default_rng(20260803).integers(
            0,
            self.scattered_prepared.counts[..., None],
            size=self.scattered_prepared.moved_output_shape,
            dtype=np.intp,
        )

    def time_vectorized(
        self,
        candidate_extent: int,
        eligible_fraction: str,
        size: int,
    ) -> None:
        sample_vectorized(self.prepared, self.ordinals)

    def time_internal_mapping(
        self,
        candidate_extent: int,
        eligible_fraction: str,
        size: int,
    ) -> None:
        _internal_mapper(self.prepared, self.ordinals)

    def time_internal_mapping_scattered(
        self,
        candidate_extent: int,
        eligible_fraction: str,
        size: int,
    ) -> None:
        _internal_mapper(self.scattered_prepared, self.scattered_ordinals)

    def time_prepare(
        self,
        candidate_extent: int,
        eligible_fraction: str,
        size: int,
    ) -> None:
        prepare_sampling(
            self.eligibility,
            repetitions=self.eligibility.shape[0],
            size=size,
            axis=-1,
        )

    def time_prepare_scattered(
        self,
        candidate_extent: int,
        eligible_fraction: str,
        size: int,
    ) -> None:
        prepare_sampling(
            self.scattered_eligibility,
            repetitions=self.scattered_eligibility.shape[0],
            size=size,
            axis=-1,
        )

    def time_reference(
        self,
        candidate_extent: int,
        eligible_fraction: str,
        size: int,
    ) -> None:
        sample_reference(self.prepared, self.ordinals)

    def time_vectorized_scattered(
        self,
        candidate_extent: int,
        eligible_fraction: str,
        size: int,
    ) -> None:
        sample_vectorized(self.scattered_prepared, self.scattered_ordinals)


class OrdinalDrawing:
    params: ClassVar[list[str]] = [
        "uniform-four",
        "mixed",
        "singleton-heavy",
        "singletons",
    ]
    param_names: ClassVar[list[str]] = ["eligibility_kind"]

    def setup(self, eligibility_kind: str) -> None:
        repetitions = 8_192
        patterns = {
            "uniform-four": [4, 4, 4, 4],
            "mixed": [1, 2, 3, 4],
            "singleton-heavy": [1, 1, 1, 4],
            "singletons": [1, 1, 1, 1],
        }
        counts = np.resize(patterns[eligibility_kind], repetitions)
        eligible = np.arange(4) < counts[:, None]
        self.prepared = prepare_sampling(
            eligible,
            repetitions=repetitions,
            size=3,
            axis=-1,
        )
        self.rng = np.random.default_rng(20260803)

    def time_draw_ordinals(self, eligibility_kind: str) -> None:
        _draw_ordinals(self.rng, self.prepared)


class DragonHuntScenario:
    params: ClassVar[list[int]] = [2_000, 10_000, 100_000]
    param_names: ClassVar[list[str]] = ["repetitions"]

    def time_fixed_slot_targeting(self, repetitions: int) -> None:
        # This is a performance baseline, not an outcome-equivalence oracle:
        # it may select dead slots and therefore follows a different RNG path.
        dragon_hunt_active(repetitions)

    def time_event_masked_targeting(self, repetitions: int) -> None:
        dragon_hunt_event_masked(repetitions)
