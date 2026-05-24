# Copyright (c) The InverSQL Authors - All Rights Reserved

import pandas as pd
import pytest

from inversql.joins import (
    Joiner,
    HintJoiner,
    SharedColNameJoiner,
    all_subsets,
    cross_join,
)


def _sets():
    yield "abcde"
    yield "abcd"


@pytest.fixture(params=_sets())
def column_set(request):
    return request.param


def test_subsets(column_set):
    assert len(list(all_subsets(column_set))) == 2 ** len(column_set)


@pytest.fixture
def shared_cn_joiner():
    return SharedColNameJoiner()


@pytest.fixture
def hint_joiner():
    return HintJoiner([("id", "user_id")])


def test_cross_join(join_left_df: pd.DataFrame, join_right_df: pd.DataFrame) -> None:
    # Unpack because this should only yield 1 result.
    [joined] = cross_join(join_left_df, join_right_df)
    assert len(joined.df) == len(join_left_df) * len(join_right_df)


def test_cross_join_is_joiner():
    assert isinstance(cross_join, Joiner)


def test_shared_col_name_joiner(
    shared_cn_joiner: Joiner, join_left_df: pd.DataFrame, join_right_df: pd.DataFrame
) -> None:
    # Unpack because this should only yield 1 result.
    [joined] = shared_cn_joiner(join_left_df, join_right_df)
    assert len(joined.df) == 5
    assert list(joined.df.index) == [1, 2, 2, 3, 4]


def test_shared_col_name_joiner_is_joiner(shared_cn_joiner: Joiner):
    assert isinstance(shared_cn_joiner, Joiner)


def test_hint_joiner(hint_joiner: Joiner):
    left = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
    right = pd.DataFrame({"user_id": [1, 2, 2, 4], "amount": [45, 120, 15, 89]})

    [joined] = hint_joiner(left, right)

    assert len(joined.df) == 3
    assert list(joined.df["id"]) == [1, 2, 2]
    assert list(joined.df["user_id"]) == [1, 2, 2]
    assert joined.how == "inner"
    assert joined.on == []
    assert joined.left_on == ["id"]
    assert joined.right_on == ["user_id"]


def test_hint_joiner_is_joiner(hint_joiner: Joiner):
    assert isinstance(hint_joiner, Joiner)
