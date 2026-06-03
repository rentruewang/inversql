# Copyright (c) The InverSQL Authors - All Rights Reserved

import itertools
import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import pandas as pd

from inversql.rels import JoinRelation, Relation, SourceRelation

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

    def __call__(self, *sources: SourceRelation) -> cabc.Iterator[Relation]:
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

    def __call__(self, *sources: SourceRelation) -> cabc.Iterator[Relation]:
        for result in self.joiner(*sources):
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
    def __call__(self, *sources: SourceRelation) -> cabc.Iterator[Relation]:
        for joiner in self._filtered_joiners:
            yield from joiner(*sources)

    @property
    def _filtered_joiners(self) -> cabc.Generator[Joiner]:
        for joiner in self.joiners:
            yield FilteredJoiner.wrap(joiner)


def cross_joiner(*sources: SourceRelation) -> cabc.Iterator[Relation]:
    """
    Cross join gives the cartesian product.
    """

    def cross_join(l, r):
        return JoinRelation(l, r, "cross")

    yield functools.reduce(cross_join, sources)


def shared_col_name_joiner(
    *sources: SourceRelation, limit: int = 32
) -> cabc.Generator[Relation]:
    """
    Join 2 dataframes with their shared columns.

    If left.a and left.b are both joinable to right.a right.b,
    this explores the case of joining only a, only b, both a and b.

    So this would yield 2**n - 1 join results.

    Only supports the cases when there are multiple tables.
    Single table not allowed, since `cross_joiner` does it already.

    Args:
        *sources: The source relations.
        limit: The maximum number of subsets to show. Default to 32.
    """

    if len(sources) <= 1:
        return

    same_cols = list(_same_column_names(sources))

    # Using the zip to limit how many subsets we want.
    for _, subset in zip(range(limit), all_subsets(same_cols)):
        if not subset:
            continue

        def inner_join(l, r):
            join_key = tuple(subset)
            return JoinRelation(l, r, "inner", left_on=join_key, right_on=join_key)

        yield functools.reduce(inner_join, sources)


def _same_column_names(sources: cabc.Iterable[Relation]) -> set[str]:
    "Get the column names that are shared."

    names = [{c.column for c in source.columns} for source in sources]
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

    for cnt in reversed(range(0, len(sequence) + 1)):
        for combo in itertools.combinations(sequence, cnt):
            yield list(combo)
