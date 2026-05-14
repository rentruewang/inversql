# Copyright (c) The InverSQL Authors - All Rights Reserved

import abc
import typing
from collections import abc as cabc

__all__ = ["AxisInfo", "HighLight"]


@typing.runtime_checkable
class AxisInfo(typing.Protocol):
    """
    The `AxisInfo` object is used to retrieve the information about each axis.
    Mainly, it tells you which index is selected on this axis (by subscription.)
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        The length of each axis.

        Returns:
            The number of rows or number of columns.
        """

    @abc.abstractmethod
    def __int__(self) -> typing.Literal[0, 1]:
        """
        The ID of the axis. 0 for row and 1 for column.
        """

    @abc.abstractmethod
    def __getitem__(self, idx: int, /) -> bool:
        """
        Whether or not the idx is selected.
        """

        raise NotImplementedError

    def __str__(self) -> typing.Literal["row", "col"]:
        """
        The name of the axis. Can be used to check it's a row or col.
        """

        # Since `int(self)` must be `0` or `1`, this is ok.
        return "col" if int(self) else "row"

    def __iter__(self) -> cabc.Iterator[int]:
        """
        The chosen pairs for each axis.

        Returns:
            The axes ids.
            `[a, b, c]` if row a, b, c are selected,
            assuming the `AxisInfo` is representing a row.
        """

        for idx in range(len(self)):
            if self[idx]:
                yield idx


@typing.runtime_checkable
class HighLight(typing.Protocol):
    """
    `HighLight` represents the cells that are currently highlighted on screen,
    based on user selection of cells or rows or columns.

    It aims to mimic how Google sheet / Excel's cell selection logic,
    where selecting cells / rows / columns would affect which cells are highlighed.

    Note:
        Originally this class looks like `AxisInfo`,
        with every method taking an `axis` parameter,
        making writing an elegant interface burdensome.

        However, that just feels not very elegant.

        Naturally, the simplest way is to move `axis` parameter to `self`,
        and therefore `AxisInfo` is born.
    """

    def __init__(self, num_rows: int, num_cols: int) -> None:
        """
        As highlight all start with a state of 0,
        only `num_rows` and `num_cols` is needed in
        """

    def __call__(self, axis: typing.Literal[0, 1]) -> AxisInfo:
        """
        Return information on a specific axis.
        Each axis supports random access to check whether it is highlighted.
        """

        if axis not in [0, 1]:
            raise IndexError(f"Axis must be one of 0, 1. Got {axis=}.")

        return self._call(axis=axis)

    @abc.abstractmethod
    def _call(self, axis: typing.Literal[0, 1]) -> AxisInfo:
        """
        Return the `AxisInfo` for each axis.
        """

        raise NotImplementedError
