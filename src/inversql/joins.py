# Copyright (c) The InverSQL Authors - All Rights Reserved

import abc
import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import pandas as pd

__all__ = ["Joiner", "joiner_dcls", "CrossJoin", "SharedColNameJoiner"]


@dcls.dataclass(frozen=True)
class JoinResult:
    """
    `JoinResult` yields the join type and how the joins are performed.
    """

    df: pd.DataFrame
    "The dataframe that is joined."

    how: str
    "The join type."

    on: cabc.Sequence[str]
    "The columns that act as key during the join."

    def __bool__(self):
        return self.not_empty

    @property
    def not_empty(self) -> bool:
        "If the dataframe is empty, it's treated as invalid."

        return bool(len(self.df))


@typing.dataclass_transform()
def joiner_dcls(cls):
    return dcls.dataclass()(cls)


@joiner_dcls
class Joiner(abc.ABC):
    """
    The interface for joining 2 dataframes (so far), in every ways you can imagine.

    If the returned `pd.DataFrame` has `len` == 0, it means the join failed.

    A `Joiner` can give multiple results, and it's data dependent.
    If nothing is yielded, that means no valid table can be had from this joiner.
    """

    dataframes: list[pd.DataFrame]
    """
    `pd.DataFrame`s to join.
    """

    def __call__(self) -> cabc.Iterator[JoinResult]:
        for result in self.join():
            if not result:
                continue
            yield result

    @abc.abstractmethod
    def join(self) -> cabc.Iterator[JoinResult]:
        """
        Yields all the potential tables that this `Joiner` knows.
        """

        raise NotImplementedError


@joiner_dcls
class CrossJoiner(Joiner):

    @typing.override
    def join(self) -> cabc.Generator[JoinResult]:
        df = functools.reduce(self.cross_join, self.dataframes)
        yield JoinResult(df, how="cross", on=[])

    @staticmethod
    def cross_join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        """
        Cross join gives the cartesian product.
        """

        return left.merge(right, how="cross")


@joiner_dcls
class SharedColNameJoiner(Joiner):
    """
    Join 2 dataframes with their shared columns.

    If left.a and left.b are both joinable to right.a right.b,
    this explores the case of joining only a, only b, both a and b.

    So this would yield 2**n - 1 join results.
    """

    @typing.override
    def join(self):
        same_cols = list(self.same_column_names())

        for subset in all_subsets(same_cols):
            dfs = [df.set_index(subset) for df in self.dataframes]
            join: typing.Any = pd.DataFrame.join
            joined = functools.reduce(join, dfs)
            yield JoinResult(joined, how="inner", on=subset)

    def same_column_names(self) -> set[str]:
        names = [set(df.columns) for df in self.dataframes]
        shared = functools.reduce(set.intersection, names)
        assert isinstance(shared, set) and all(isinstance(k, str) for k in shared)
        return shared


def all_subsets(sequence: cabc.Sequence[str]) -> cabc.Generator[cabc.Sequence[str]]:
    """
    Yields all possible subsets recursively.
    """

    if not sequence:
        yield []
        return

    *drop_last, last = sequence

    for subsets in all_subsets(drop_last):
        yield subsets
        yield *subsets, last
