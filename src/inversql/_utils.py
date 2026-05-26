# Copyright (c) The InverSQL Authors - All Rights Reserved


import numpy as np
from numpy import typing as npt

__all__ = ["IntArray", "FloatArray", "BoolArray"]

type IntArray = npt.NDArray[np.int_]
type FloatArray = npt.NDArray[np.floating]
type BoolArray = npt.NDArray[np.bool_]
