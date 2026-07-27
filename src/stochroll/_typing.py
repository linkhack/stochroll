from typing import Any

import numpy as np
from numpy.typing import NDArray

# Values accepted through the public Python API.
type NumericScalar = int | float | np.integer[Any] | np.floating[Any]

# NumPy dtype scalar classes
type NumericDTypeScalar = np.integer[Any] | np.floating[Any]

# General ndarray categories
type IntegerArray = NDArray[np.integer[Any]]
type FloatingArray = NDArray[np.floating[Any]]
type NumericArray = NDArray[NumericDTypeScalar]
type BooleanArray = NDArray[np.bool_]

# Domain-specific ndarray categories
type RollArray = NumericArray
type PoolArray = IntegerArray
type EventArray = BooleanArray


type ShapeLike = int | tuple[int, ...]
type AxisLike = int | tuple[int, ...] | None
