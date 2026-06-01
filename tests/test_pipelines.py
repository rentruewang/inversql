# Copyright (c) The InverSQL Authors - All Rights Reserved

import pandas as pd
import pytest

from inversql.pipelines import Pipeline
from inversql.refs import AnnotatedDF


@pytest.fixture
def pipeline():
    return Pipeline()


def annot_left(join_left_df: pd.DataFrame):
    return AnnotatedDF("join_left_df", join_left_df)
