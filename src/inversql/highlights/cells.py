# Copyright (c) The InverSQL Authors - All Rights Reserved

import dataclasses as dcls
import typing

import numpy as np
from numpy import typing as npt

from .highlights import AxisInfo, HighLight

__all__ = ["CellHighLight", "CellAxisInfo"]


@dcls.dataclass(init=False)
class CellHighLight(HighLight):
    """
    A cell-based highlight that supports highlighting individual cells.
    """

    cells: npt.NDArray
    """
    The individual cells selected.
    """

    @typing.override
    def __init__(self, num_rows: int, num_cols: int) -> None:
        self.cells = np.zeros([num_rows, num_cols], dtype=bool)

    @typing.override
    def _call(self, axis: typing.Literal[0, 1]):
        return CellAxisInfo(state=self, axis=axis)

    def __getitem__(self, idx: tuple[int, int]) -> bool:
        return self.cells[idx]

    def __setitem__(self, idx: tuple[int, int], to: bool) -> None:
        self.cells[idx] = to


@dcls.dataclass(frozen=True)
class CellAxisInfo(AxisInfo):
    state: CellHighLight
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
        return self.reduced[idx]

    @typing.override
    def __len__(self) -> int:
        return self.cells.shape[self.axis]

    @property
    def cells(self):
        return self.state.cells

    @property
    def reduced(self):
        # Using `1 - axis` because we reduce the "other" axis.
        return self.cells.sum(axis=1 - self.axis)
