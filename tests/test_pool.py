import numpy as np
from helpers import FixedRNG

from stochroll import Pool, Roller


def test_pool_shape_and_bounds() -> None:
    roller = Roller(repetitions=100, seed=42)

    pool = roller.pool(4, d=6, shape=3)

    assert pool.values.shape == (100, 3, 4)
    assert pool.values.min() >= 1
    assert pool.values.max() <= 6


def test_keep_highest_matches_drop_lowest() -> None:
    roller = Roller(repetitions=3, seed=42)
    pool = roller.pool(4, d=6)

    kept = np.sort(pool.keep_highest(3).values, axis=-1)
    dropped = np.sort(pool.drop_lowest(1).values, axis=-1)

    np.testing.assert_array_equal(kept, dropped)


def test_pool_selectors_keep_the_requested_values() -> None:
    pool = Pool(
        np.array(
            [
                [1, 6, 3, 2],
                [4, 2, 5, 1],
            ],
            dtype=np.int8,
        ),
        sides=6,
        roller=Roller(repetitions=2, seed=42),
    )

    np.testing.assert_array_equal(
        np.sort(pool.keep_highest(2).values, axis=-1),
        [[3, 6], [4, 5]],
    )
    np.testing.assert_array_equal(
        np.sort(pool.keep_lowest(2).values, axis=-1),
        [[1, 2], [1, 2]],
    )
    np.testing.assert_array_equal(
        np.sort(pool.drop_highest(2).values, axis=-1),
        [[1, 2], [1, 2]],
    )
    np.testing.assert_array_equal(
        np.sort(pool.drop_lowest(2).values, axis=-1),
        [[3, 6], [4, 5]],
    )


def test_pool_reductions_are_exact() -> None:
    pool = Pool(
        np.array([[1, 6, 3], [4, 2, 5]], dtype=np.int8),
        sides=6,
        roller=Roller(repetitions=2, seed=42),
    )

    np.testing.assert_array_equal(pool.first().values, [1, 4])
    np.testing.assert_array_equal(pool.last().values, [3, 5])
    np.testing.assert_array_equal(pool.sum().values, [10, 11])
    np.testing.assert_array_equal(pool.min().values, [1, 2])
    np.testing.assert_array_equal(pool.max().values, [6, 5])
    np.testing.assert_array_equal(pool.drop_lowest_sum().values, [9, 9])
    np.testing.assert_array_equal(pool.count_at_least(4).values, [1, 2])


def test_single_die_pool_can_be_resolved_directly() -> None:
    pool = Pool(
        np.array([[6], [2]], dtype=np.int8),
        sides=6,
        roller=Roller(repetitions=2, seed=42),
    )

    np.testing.assert_array_equal(pool.single().values, [6, 2])
    np.testing.assert_array_equal(pool.sum().values, [6, 2])
    np.testing.assert_array_equal(pool.min().values, [6, 2])
    np.testing.assert_array_equal(pool.max().values, [6, 2])


def test_pool_selectors_support_identity_and_symmetric_paths() -> None:
    pool = Pool(
        np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int8),
        sides=6,
        roller=Roller(repetitions=2, seed=42),
    )

    np.testing.assert_array_equal(pool.keep_highest(4).values, pool.values)
    np.testing.assert_array_equal(pool.keep_lowest(4).values, pool.values)
    np.testing.assert_array_equal(pool.drop_lowest(0).values, pool.values)
    np.testing.assert_array_equal(pool.drop_highest(0).values, pool.values)
    np.testing.assert_array_equal(
        np.sort(pool.keep_lowest(3).values, axis=-1),
        [[1, 2, 3], [1, 2, 3]],
    )
    np.testing.assert_array_equal(
        np.sort(pool.drop_highest(3).values, axis=-1),
        [[1], [1]],
    )


def test_reroll_once_replaces_each_matching_die_once() -> None:
    roller = Roller(repetitions=2, seed=42)
    roller.rng = FixedRNG()  # type: ignore[assignment]
    pool = Pool(
        np.array([[1, 2, 1, 6], [3, 1, 5, 1]], dtype=np.int8),
        sides=6,
        roller=roller,
    )

    rerolled = pool.reroll_once([1])

    expected = np.array([[6, 2, 6, 6], [3, 6, 5, 6]], dtype=np.int8)
    np.testing.assert_array_equal(rerolled.values, expected)


def test_reroll_without_matching_values_preserves_pool() -> None:
    pool = Pool(
        np.array([[2, 3], [4, 5]], dtype=np.int8),
        sides=6,
        roller=Roller(repetitions=2, seed=42),
    )

    np.testing.assert_array_equal(pool.reroll_once([1]).values, pool.values)


def test_pool_reductions_do_not_overflow_at_uint8_boundary() -> None:
    pool = Pool(
        np.array([[255, 255], [254, 255]], dtype=np.uint8),
        sides=255,
        roller=Roller(repetitions=2, seed=42),
    )

    np.testing.assert_array_equal(pool.sum().values, [510, 509])
    np.testing.assert_array_equal(pool.drop_lowest_sum().values, [255, 255])


def test_pool_reductions_handle_large_and_small_dice_axes() -> None:
    one_die = Pool(
        np.array([[7], [9]], dtype=np.int16),
        sides=10,
        roller=Roller(repetitions=2, seed=42),
    )
    many_dice = Pool(
        np.array([list(range(1, 18)), list(range(17, 0, -1))], dtype=np.int16),
        sides=17,
        roller=Roller(repetitions=2, seed=42),
    )

    np.testing.assert_array_equal(one_die.sum().values, [7, 9])
    np.testing.assert_array_equal(many_dice.sum().values, [153, 153])
    np.testing.assert_array_equal(many_dice.min().values, [1, 1])
    np.testing.assert_array_equal(many_dice.max().values, [17, 17])
