# Copyright (c) The InverSQL Authors - All Rights Reserved

import pandas as pd
import pytest

from inversql.rels import JoinRelation, NumericDF, Relation, SourceRelation


@pytest.fixture
def left_rel(join_left_df: pd.DataFrame):
    return SourceRelation("join_left", join_left_df)


@pytest.fixture
def right_rel(join_right_df: pd.DataFrame):
    return SourceRelation("join_right", join_right_df)


@pytest.fixture(params=["inner", "cross"])
def how(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def join_rel(left_rel: Relation, right_rel: Relation):
    return JoinRelation(
        left_rel, right_rel, how="inner", left_on="user_id", right_on="user_id"
    )


def test_left_rel(left_rel: SourceRelation):
    assert len(left_rel.data()) == 4
    assert {col.table for col in left_rel.columns} == {"join_left"}
    labels = [f"join_left.{x}" for x in "user_id,name,email".split(",")]
    assert {str(col.ref()) for col in left_rel.columns} == set(labels)


def test_right_rel(right_rel: SourceRelation):
    assert len(right_rel.data()) == 4
    assert {col.table for col in right_rel.columns} == {"join_right"}
    labels = [f"join_right.{x}" for x in "user_id,order_id,amount".split(",")]
    assert {str(col.ref()) for col in right_rel.columns} == set(labels)


def test_join(
    join_rel: JoinRelation, left_rel: SourceRelation, right_rel: SourceRelation
):
    assert join_rel.columns == {*left_rel.columns, *right_rel.columns}


def test_join_len(
    join_rel: JoinRelation, left_rel: SourceRelation, right_rel: SourceRelation
):
    if join_rel.how == "inner":
        assert len(join_rel.data()) <= min(len(left_rel.data()), len(right_rel.data()))

    elif join_rel.how == "cross":
        assert len(join_rel.data()) == len(left_rel.data()) * len(right_rel.data())

    else:
        raise ValueError("Not present in test cases.")


@pytest.fixture
def num_join_left(join_left_df: pd.DataFrame):
    return NumericDF(join_left_df)


@pytest.fixture
def num_join_right(join_right_df: pd.DataFrame):
    return NumericDF(join_right_df)


@pytest.mark.parametrize(
    "num_df,df",
    [
        (num_join_left.name, "join_left_df"),
        (num_join_right.name, "join_right_df"),
    ],
)
def test_numeric_df_left(request: pytest.FixtureRequest, num_df, df):
    num_df: NumericDF = request.getfixturevalue(num_df)
    df: pd.DataFrame = request.getfixturevalue(df)

    assert not len(num_df.numeric().select_dtypes(exclude=["number"]).columns)
    assert (num_df.original() == df).all().all()
