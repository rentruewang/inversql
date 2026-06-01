# Copyright (c) The InverSQL Authors - All Rights Reserved

import pandas as pd
import pytest

from inversql.joins import (
    FilteredJoiner,
    Joiner,
    JoinerList,
    all_subsets,
    cross_joiner,
    shared_col_name_joiner,
)


def _sets():
    yield "abcde"
    yield "abcd"


@pytest.fixture(params=_sets())
def column_set(request):
    return request.param


@pytest.fixture
def tables(join_left_df: pd.DataFrame, join_right_df: pd.DataFrame):
    return {
        "join_left_df": join_left_df,
        "join_right_df": join_right_df,
    }


def test_subsets(column_set):
    assert len(list(all_subsets(column_set))) == 2 ** len(column_set)


def test_joiner_instances():
    assert isinstance(cross_joiner, Joiner)
    assert isinstance(shared_col_name_joiner, Joiner)

    assert issubclass(FilteredJoiner, Joiner)
    assert issubclass(JoinerList, Joiner)


def test_cross_join(
    tables: dict[str, pd.DataFrame],
    join_left_df: pd.DataFrame,
    join_right_df: pd.DataFrame,
) -> None:
    # Unpack because this should only yield 1 result.
    [joined] = FilteredJoiner(cross_joiner)(tables)
    assert len(joined.df) == len(join_left_df) * len(join_right_df)


def test_shared_col_name_joiner(tables: dict[str, pd.DataFrame]) -> None:
    # Unpack because this should only yield 1 result.
    [joined] = FilteredJoiner(shared_col_name_joiner)(tables)
    assert len(joined.df) == 5
    assert list(joined.df.index) == [1, 2, 2, 3, 4]
