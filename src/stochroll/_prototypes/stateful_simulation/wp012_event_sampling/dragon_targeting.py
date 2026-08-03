"""Dragon Hunt targeting expressed through event-masked sampling."""

from __future__ import annotations

from stochroll import Roll

from .._shared.active_batch import ActiveBatch

DRAGON_ATTACKS = 3


def dragon_targets(batch: ActiveBatch, player_hp: Roll) -> Roll:
    """Choose three targets alive at the beginning of the attack phase."""
    return batch.sample_indices(player_hp > 0, size=DRAGON_ATTACKS)
