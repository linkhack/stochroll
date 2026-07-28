import numpy as np
from numpy.typing import NDArray

from stochroll import Roller, where


def adv_guidance() -> NDArray[np.float64]:
    r = Roller(repetitions=1_000_000)

    # -------------------------
    # Enemy setup
    # -------------------------

    ac = (r.d(10) + 14).broadcast_to(6)  # enemy AC 15..24, shared across attacks
    shield = r.d(8, shape=2).sum() + 10  # shield pool: 2d8 + 10

    # -------------------------
    # Attack rolls
    # -------------------------

    d20 = r.d(20, shape=6)
    attacks = d20 + 14

    nat1 = d20 == 1
    crits = d20 == 20
    hits = crits | ((attacks >= ac) & ~nat1)
    normal_hits = hits & ~crits

    num_crits = crits.count()
    armor_cracked = num_crits >= 2

    # -------------------------
    # Damage rolls
    # -------------------------

    normal_damage = r.d(10, shape=6) + 9
    crit_damage = normal_damage + r.d(10, shape=6)

    # If armor cracked, every normal hit gets extra 1d6.
    crack_bonus = where(
        armor_cracked.broadcast_to(6),
        r.d(6, shape=6),
        0,
    )

    damage_per_attack = where(
        crits,
        crit_damage,
        where(normal_hits, normal_damage + crack_bonus, 0),
    )

    raw_damage = damage_per_attack.sum()

    # -------------------------
    # Shield absorption
    # -------------------------

    hp_damage = where(raw_damage > shield, raw_damage - shield, 0)

    # -------------------------
    # Overkill splash
    # -------------------------

    overkill = hp_damage >= 30
    splash_damage = where(
        overkill,
        r.d(12) + num_crits,
        0,
    )

    total_effective_damage = hp_damage + splash_damage
    return total_effective_damage.expected()
