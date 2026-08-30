# Copyright (c) InverSQL Authors - All Rights Reserved


import typing
import numpy as np
from numpy import typing as npt

__all__ = ["IntArray", "FloatArray", "BoolArray"]

IntArray: typing.TypeAlias = npt.NDArray[np.int_]
FloatArray: typing.TypeAlias = npt.NDArray[np.floating]
BoolArray: typing.TypeAlias = npt.NDArray[np.bool_]
