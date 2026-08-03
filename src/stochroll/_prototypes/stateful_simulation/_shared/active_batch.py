"""Isolated whole-repetition packing prototype for WP-009-01."""

from __future__ import annotations

from typing import Self, SupportsIndex, cast, overload

import numpy as np
from numpy.typing import NDArray

from stochroll import Event, Pool, Roll
from stochroll import Roller as ProductionRoller
from stochroll._typing import EventArray, PoolArray, RollArray, ShapeLike

from ..wp012_event_sampling.sampling import (
    draw_ordinals,
    prepare_sampling,
    sample_vectorized_unchecked,
)


class Roller(ProductionRoller):
    """Prototype Roller with an explicit whole-repetition batch boundary."""

    @classmethod
    def _with_shared_rng(
        cls,
        repetitions: int,
        rng: np.random.Generator,
    ) -> Self:
        """Construct an internal compact view without creating a discarded RNG."""
        roller = cls.__new__(cls)
        roller.repetitions = repetitions
        roller.rng = rng
        return roller

    def active_batch(self, active: Event) -> ActiveBatch | None:
        if not isinstance(active, Event):
            raise TypeError("active must be an Event")
        if active.values.shape != (self.repetitions,):
            raise ValueError(
                "active must have exact shape "
                f"({self.repetitions},), got {active.values.shape}"
            )

        positions = np.flatnonzero(active.values)
        if positions.size == 0:
            return None
        return ActiveBatch(self, positions)


class ActiveBatch:
    """A compact view of stable active repetition positions."""

    def __init__(self, parent: Roller, positions: NDArray[np.intp]) -> None:
        self._parent = parent
        self._positions = positions
        self._positions.flags.writeable = False
        self._roller = Roller._with_shared_rng(len(positions), parent.rng)

    @property
    def repetitions(self) -> int:
        return len(self._positions)

    @property
    def positions(self) -> NDArray[np.intp]:
        return self._positions

    def d(self, sides: int, *, shape: ShapeLike | None = None) -> Roll:
        return self._roller.d(sides, shape=shape)

    def pool(
        self,
        dice: int,
        *,
        d: int,
        shape: ShapeLike | None = None,
    ) -> Pool:
        return self._roller.pool(dice, d=d, shape=shape)

    def sample_indices(
        self,
        eligible: Event,
        *,
        size: SupportsIndex = 1,
        axis: SupportsIndex = -1,
    ) -> Roll:
        """Sample eligible structural indices for each preserved coordinate."""
        if not isinstance(eligible, Event):
            raise TypeError("eligible must be an Event")

        prepared = prepare_sampling(
            eligible.values,
            repetitions=self.repetitions,
            size=size,
            axis=axis,
        )
        if prepared.counts.size == 0:
            # A zero-sized preserved axis means there are no eligibility
            # slices to validate or sample, so the shared RNG stays untouched.
            return Roll(np.empty(prepared.output_shape, dtype=np.intp))

        ordinals = draw_ordinals(self._roller.rng, prepared)
        return Roll(sample_vectorized_unchecked(prepared, ordinals))

    @overload
    def take(self, value: Roll) -> Roll: ...

    @overload
    def take(self, value: Event) -> Event: ...

    @overload
    def take(self, value: Pool) -> Pool: ...

    def take(self, value: Roll | Event | Pool) -> Roll | Event | Pool:
        self._validate_dense(value, name="value")
        values = value.values[self._positions]

        if type(value) is Roll:
            return Roll(cast(RollArray, values))
        if type(value) is Event:
            return Event(cast(EventArray, values))

        pool = cast(Pool, value)
        return Pool(
            cast(PoolArray, values),
            sides=pool.sides,
            roller=self._roller,
        )

    @overload
    def merge(self, base: Roll, update: Roll) -> Roll: ...

    @overload
    def merge(self, base: Event, update: Event) -> Event: ...

    @overload
    def merge(self, base: Pool, update: Pool) -> Pool: ...

    def merge(
        self,
        base: Roll | Event | Pool,
        update: Roll | Event | Pool,
    ) -> Roll | Event | Pool:
        self._validate_merge(base, update)

        if type(base) is Roll:
            roll_update = cast(Roll, update)
            dtype = np.result_type(base.values.dtype, roll_update.values.dtype)
            values = base.values.astype(dtype, copy=True)
            values[self._positions] = roll_update.values
            return Roll(cast(RollArray, values))

        if type(base) is Event:
            event_update = cast(Event, update)
            values = base.values.copy()
            values[self._positions] = event_update.values
            return Event(values)

        base_pool = cast(Pool, base)
        update_pool = cast(Pool, update)
        values = base_pool.values.copy()
        values[self._positions] = update_pool.values
        return Pool(
            values,
            sides=base_pool.sides,
            roller=self._parent,
        )

    def _validate_dense(self, value: object, *, name: str) -> None:
        if type(value) not in (Roll, Event, Pool):
            raise TypeError(f"{name} must be a Roll, Event, or Pool")
        wrapper = cast(Roll | Event | Pool, value)
        if wrapper.values.shape[0] != self._parent.repetitions:
            raise ValueError(f"{name} must have {self._parent.repetitions} repetitions")
        if type(wrapper) is Pool and wrapper.roller is not self._parent:
            raise ValueError(f"{name} Pool must reference the parent Roller")

    def _validate_merge(self, base: object, update: object) -> None:
        self._validate_dense(base, name="base")
        if type(update) not in (Roll, Event, Pool):
            raise TypeError("update must be a Roll, Event, or Pool")
        if type(base) is not type(update):
            raise TypeError("base and update must have the same wrapper type")

        dense = cast(Roll | Event | Pool, base)
        compact = cast(Roll | Event | Pool, update)
        if compact.values.shape[0] != self.repetitions:
            raise ValueError(f"update must have {self.repetitions} repetitions")
        if dense.values.shape[1:] != compact.values.shape[1:]:
            raise ValueError("base and update trailing shapes must match exactly")

        if type(dense) is Pool:
            dense_pool = dense
            compact_pool = cast(Pool, compact)
            if compact_pool.roller is not self._roller:
                raise ValueError("update Pool must reference this active batch")
            if compact_pool.sides != dense_pool.sides:
                raise ValueError("Pool sides must match")
            if compact_pool.values.dtype != dense_pool.values.dtype:
                raise ValueError("Pool dtypes must match")
