# Copyright (c) The InverSQL Authors - All Rights Reserved

import pandas as pd
import pytest

from inversql.refs import NumericDF


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
    assert (num_df.revert(num_df.numeric()) == df).all().all()
