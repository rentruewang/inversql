# Copyright (c) The InverSQL Authors - All Rights Reserved

import pandas as pd

from inversql.joins import Joiner, cross_join


def test_cross_join(join_left_df: pd.DataFrame, join_right_df: pd.DataFrame) -> None:
    joined = cross_join(join_left_df, join_right_df)
    assert len(joined) == len(join_left_df) * len(join_right_df)


def test_cross_join_is_joiner():
    assert isinstance(cross_join, Joiner)
