from stochroll import Roller

roller = Roller(repetitions=1_000, seed=42)
result = roller.pool(4, d=6).drop_lowest().sum()

assert result.values.shape == (1_000,)
assert result.values.min() >= 3
assert result.values.max() <= 18

print("StochRoll distribution smoke test passed.")
