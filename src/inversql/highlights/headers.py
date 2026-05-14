# Copyright (c) The InverSQL Authors - All Rights Reserved

import dataclasses as dcls
import typing

import numpy as np
from numpy import typing as npt

from .highlights import AxisInfo, HighLight

__all__ = ["HeaderHighLight", "HeaderAxisInfo"]


@dcls.dataclass(init=False)
class HeaderHighLight(HighLight):
    rows: npt.NDArray
    """
    The rows selected.
    """

    cols: npt.NDArray
    """
    The columns selected.
    """

    @typing.override
    def __init__(self, num_rows: int, num_cols: int) -> None:
        self.rows = np.zeros([num_rows], dtype=bool)
        self.cols = np.zeros([num_cols], dtype=bool)

    @typing.override
    def _call(self, axis: typing.Literal[0, 1]):
        return HeaderAxisInfo(state=self, axis=axis)


@dcls.dataclass(frozen=True)
class HeaderAxisInfo(AxisInfo):
    state: HeaderHighLight
    """
    The state of the current board.
    """

    axis: typing.Literal[0, 1]
    """
    Axis which this accessor is concerned about.
    """

    @typing.override
    def __int__(self):
        return self.axis

    @typing.override
    def __getitem__(self, idx: int):
        return self.header[idx]

    def __setitem__(self, idx: int, to: bool) -> None:
        self.header[idx] = to

    @typing.override
    def __len__(self) -> int:
        return len(self.header)

    @property
    def header(self) -> npt.NDArray:
        """
        The header for the table.
        """

        return self.axes[self.axis]

    @property
    def axes(self) -> tuple[npt.NDArray, npt.NDArray]:
        """
        The accessor for row/col,
        organized this way so that it can be accessed with `axis` as index.
        """

        return self.state.rows, self.state.cols
