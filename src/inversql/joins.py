# Copyright (c) The InverSQL Authors - All Rights Reserved

import typing

import pandas as pd

__all__ = ["Joiner", "cross_join"]


@typing.runtime_checkable
class Joiner(typing.Protocol):
    "The interface for joining 2 dataframes, in every ways you can imagine."

    def __call__(self, left: pd.DataFrame, right: pd.DataFrame, /) -> pd.DataFrame: ...


def cross_join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """
    Cross join gives the cartesian product.
    """

    return left.merge(right, how="cross")
