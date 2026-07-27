import numpy as np

from stochroll import Roller


def test_seeded_rolls_are_reproducible() -> None:
    first = Roller(repetitions=20, seed=42)
    second = Roller(repetitions=20, seed=42)

    np.testing.assert_array_equal(first.d(20).values, second.d(20).values)
    np.testing.assert_array_equal(
        first.d(8, shape=(2, 3)).values,
        second.d(8, shape=(2, 3)).values,
    )
    np.testing.assert_array_equal(
        first.pool(4, d=6, shape=2).values,
        second.pool(4, d=6, shape=2).values,
    )


def test_scalar_and_shaped_rolls_have_expected_shapes_and_bounds() -> None:
    roller = Roller(repetitions=100, seed=42)

    scalar = roller.d(6)
    shaped = roller.d(6, shape=(2, 3))

    assert scalar.values.shape == (100,)
    assert shaped.values.shape == (100, 2, 3)
    assert scalar.values.min() >= 1
    assert scalar.values.max() <= 6
    assert shaped.values.min() >= 1
    assert shaped.values.max() <= 6
