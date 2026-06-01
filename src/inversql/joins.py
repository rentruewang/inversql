# Copyright (c) The InverSQL Authors - All Rights Reserved

import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import pandas as pd

__all__ = [
    "Joiner",
    "JoinerList",
    "FilteredJoiner",
    "cross_joiner",
    "shared_col_name_joiner",
]


@dcls.dataclass(frozen=True)
class JoinOp:
    """
    `JoinOp` tracks the join operations s.t. we can reconstruct the joins.
    """

    how: str
    "The join type."

    on: list[str] | None = None
    "The columns that act as key during the join."

    left_on: list[str] | None = None
    "The `left_on` that `pandas` uses."

    right_on: list[str] | None = None
    "The `left_on` that `pandas` uses."


@dcls.dataclass(frozen=True)
class JoinResult:
    """
    `JoinResult` yields the join type and how the joins are performed.
    """

    df: pd.DataFrame
    "The dataframe that is joined."

    sources: dict[str, pd.DataFrame]
    "The original dataframe."

    ops: JoinOp | list[JoinOp]
    """
    The original join operations. Would have length `len(sources) - 1`.
    """

    def __bool__(self):
        return self.not_empty

    @property
    def not_empty(self) -> bool:
        "If the dataframe is empty, it's treated as invalid."

        return bool(len(self.df))

    @property
    def join_ops(self) -> list[JoinOp]:
        if isinstance(self.ops, JoinOp):
            return [self.ops] * (len(self.sources) - 1)
        else:
            assert len(self.ops) == len(self.sources) - 1
            return self.ops


@typing.runtime_checkable
class Joiner(typing.Protocol):
    """
    The interface for joining 2 dataframes (so far), in every ways you can imagine.

    If the returned `pd.DataFrame` has `len` == 0, it means the join failed.

    A `Joiner` can give multiple results, and it's data dependent.
    If nothing is yielded, that means no valid table can be had from this joiner.
    """

    def __call__(
        self, dataframes: dict[str, pd.DataFrame], /
    ) -> cabc.Iterator[JoinResult]:
        """
        Yields all the potential tables that this `Joiner` knows.
        """

        ...


@dcls.dataclass
class FilteredJoiner(Joiner):
    """
    `FilteredJoiner` filters the empty `Joiner`s, as those outputs are considered invalid.
    """

    joiner: Joiner
    "The joiner whose output we want to filter."

    def __call__(
        self, dataframes: dict[str, pd.DataFrame]
    ) -> cabc.Generator[JoinResult]:
        for result in self.joiner(dataframes):
            if result:
                yield result

    @classmethod
    def wrap(cls, joiner: Joiner) -> typing.Self:
        "Wrap the `joiner` s.t. invalid outputs are discarded."
        return joiner if isinstance(joiner, cls) else cls(joiner)


@dcls.dataclass
class JoinerList(Joiner):
    """
    A list of `Joiner`s.
    """

    joiners: list[Joiner]
    "The joiners to iterate over."

    @typing.override
    def __call__(
        self, dataframes: dict[str, pd.DataFrame]
    ) -> cabc.Generator[JoinResult]:
        for joiner in self._filtered_joiners:
            yield from joiner(dataframes)

    @property
    def _filtered_joiners(self) -> cabc.Generator[Joiner]:
        for joiner in self.joiners:
            yield FilteredJoiner.wrap(joiner)


def cross_joiner(
    dataframes: dict[str, pd.DataFrame],
) -> cabc.Generator[JoinResult]:
    """
    Cross join gives the cartesian product.
    """

    def cross_join(l: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
        return pd_merge_with_suffix(l, r, how="cross", dataframes=dataframes)

    df = functools.reduce(cross_join, dataframes.values())
    yield JoinResult(df, sources=dataframes, ops=JoinOp(how="cross"))


def shared_col_name_joiner(
    dataframes: dict[str, pd.DataFrame],
) -> cabc.Generator[JoinResult]:
    """
    Join 2 dataframes with their shared columns.

    If left.a and left.b are both joinable to right.a right.b,
    this explores the case of joining only a, only b, both a and b.

    So this would yield 2**n - 1 join results.
    """

    same_cols = list(_same_column_names(dataframes.values()))

    for subset in all_subsets(same_cols):
        if not subset:
            continue

        dfs = [df.set_index(subset) for df in dataframes.values()]

        def inner_join(left, right):
            return pd_join_with_suffix(left, right, "inner", dataframes=dataframes)

        joined = functools.reduce(inner_join, dfs)

        yield JoinResult(
            df=joined, sources=dataframes, ops=JoinOp(how="inner", on=subset)
        )


def _same_column_names(dataframes: cabc.Iterable[pd.DataFrame]) -> set[str]:
    "Get the column names that are shared."
    names = [set(df.columns) for df in dataframes]
    shared: set[str] = functools.reduce(set.intersection, names)
    assert isinstance(shared, set) and all(isinstance(k, str) for k in shared)
    return shared


def all_subsets(sequence: cabc.Sequence[str]) -> cabc.Generator[list[str]]:
    """
    Yields all possible subsets recursively.
    """

    if not sequence:
        yield []
        return

    *drop_last, last = sequence

    for subsets in all_subsets(drop_last):
        yield subsets
        yield [*subsets, last]
