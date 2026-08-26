# Copyright (c) InverSQL Authors - All Rights Reserved

import pytest

from inversql.pipelines import Pipeline
from inversql.rels import SourceRelation


@pytest.fixture
def pipeline():
    return Pipeline()


@pytest.fixture
def test_unary_pipeline(left_rel: SourceRelation, pipeline: Pipeline):
    for sql in pipeline(left_rel):
        assert isinstance(sql, str)
        assert sql.startswith("SELECT")


def test_binary_pipeline(
    left_rel: SourceRelation, right_rel: SourceRelation, pipeline: Pipeline
):
    for sql in pipeline(left_rel, right_rel):
        assert isinstance(sql, str)
        assert sql.startswith("SELECT")
