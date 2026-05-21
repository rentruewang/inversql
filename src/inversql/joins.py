# Copyright (c) The InverSQL Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections import abc as cabc

import pandas as pd

__all__ = ["Joiner", "cross_join"]


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
        return self.valid()

    def valid(self) -> bool:
        "If the dataframe is empty, it's treated as invalid."

        return bool(len(self.df))


@typing.runtime_checkable
class Joiner(typing.Protocol):
    """
    The interface for joining 2 dataframes, in every ways you can imagine.

    If the returned `pd.DataFrame` has `len` == 0, it means the join failed.

    A `Joiner` can give multiple results, and it's data dependent.
    If nothing is yielded, that means no valid table can be had from this joiner.
    """

    def __call__(
        self, left: pd.DataFrame, right: pd.DataFrame, /
    ) -> cabc.Iterator[JoinResult]:
        """
        Yields all the potential tables that this `Joiner` knows.
        """

        ...


def cross_join(left: pd.DataFrame, right: pd.DataFrame):
    """
    Cross join gives the cartesian product.
    """

    yield JoinResult(df=left.merge(right, how="cross"), how="cross", on=[])


class SharedColNameJoiner(Joiner):
    """
    Join 2 dataframes with their shared columns.

    If left.a and left.b are both joinable to right.a right.b,
    this explores the case of joining only a, only b, both a and b.

    So this would yield 2**n - 1 join results.
    """

    @typing.override
    def __call__(self, left: pd.DataFrame, right: pd.DataFrame, /):
        same_cols = list(self.same_column_names(left, right))

        for subset in all_subsets(same_cols):
            # Don't yield the case where no columns are joined.
            # Coupled with "inner" join empty columns would cause result to be empty.
            if not subset:
                continue

            left_with_idx = left.set_index(same_cols)
            right_with_idx = right.set_index(same_cols)
            joined = left_with_idx.join(right_with_idx)
            yield JoinResult(joined, how="inner", on=subset)

    def same_column_names(self, left: pd.DataFrame, right: pd.DataFrame) -> set[str]:
        left_names = set(left.columns)
        right_names = set(right.columns)
        return left_names & right_names


def all_subsets(sequence: cabc.Sequence[str]) -> cabc.Generator[cabc.Sequence[str]]:
    if not sequence:
        yield []
        return

    *drop_last, last = sequence

    for subsets in all_subsets(drop_last):
        yield subsets
        yield *subsets, last
