"""Shared deterministic fixtures and metadata for benchmark scenarios."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from stochroll import Event, Pool, Roll, Roller

type SizeName = Literal["small", "large"]

REPETITIONS: Final[dict[SizeName, int]] = {
    "small": 1_000,
    "large": 250_000,
}
STRUCTURAL_SHAPE: Final = (6,)
POOL_DICE: Final = 12
LOOKUP_SIZE: Final = 3
SEED: Final = 42
KEEP_DROP_BRANCHES: Final = ("direct", "identity", "delegated")
REROLL_DENSITIES: Final = ("none", "sparse", "dense")


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """Stable ASV parameter containing public-workload metadata."""

    name: SizeName
    repetitions: int

    def __repr__(self) -> str:
        return f"{self.name}[R={self.repetitions},roll=(R,6),pool=(R,6,12)]"


STANDARD_WORKLOADS: Final = tuple(
    WorkloadSpec(name=name, repetitions=repetitions)
    for name, repetitions in REPETITIONS.items()
)


def make_roller(workload: WorkloadSpec, *, seed: int = SEED) -> Roller:
    """Create a seeded Roller for a standard public workload."""
    return Roller(repetitions=workload.repetitions, seed=seed)


@lru_cache(maxsize=None, typed=True)
def build_roll(workload: WorkloadSpec, *, sides: int = 20) -> Roll:
    """Build a standard shaped Roll through the public domain API."""
    return make_roller(workload).d(sides, shape=STRUCTURAL_SHAPE)


@lru_cache(maxsize=None, typed=True)
def build_event(workload: WorkloadSpec) -> Event:
    """Build a standard shaped Event through a public comparison."""
    return build_roll(workload) >= 11


@lru_cache(maxsize=None, typed=True)
def build_pool(
    workload: WorkloadSpec,
    *,
    dice: int = POOL_DICE,
    sides: int = 6,
) -> Pool:
    """Build a standard shaped Pool through the public domain API."""
    return make_roller(workload).pool(dice, d=sides, shape=STRUCTURAL_SHAPE)


@lru_cache(maxsize=None, typed=True)
def build_roll_lookup_indices(workload: WorkloadSpec) -> Roll:
    """Build length-three per-repetition Roll lookup indices."""
    return make_roller(workload, seed=SEED + 1).d(6, shape=LOOKUP_SIZE) - 1


@lru_cache(maxsize=None, typed=True)
def build_pool_lookup_indices(
    workload: WorkloadSpec,
) -> NDArray[np.int16]:
    """Build full-rank Pool lookup indices shaped ``(R, 3, 12)``."""
    indices = (
        make_roller(workload, seed=SEED + 1).d(
            6,
            shape=(LOOKUP_SIZE, POOL_DICE),
        )
        - 1
    )
    return indices.values.astype(np.int16, copy=False)
