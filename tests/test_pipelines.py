# Copyright (c) The InverSQL Authors - All Rights Reserved

import pytest

from inversql.pipelines import Pipeline
from inversql.rels import SourceRelation


@pytest.fixture
def pipeline():
    return Pipeline()


def test_unary_pipeline(left_rel: SourceRelation, pipeline: Pipeline):
    left_rel.toggle((1, 0))
    for sql in pipeline(left_rel):
        assert isinstance(sql, str)
        assert sql.startswith("SELECT")


def test_binary_pipeline(
    left_rel: SourceRelation, right_rel: SourceRelation, pipeline: Pipeline
):
    left_rel.toggle((0, 0))
    right_rel.toggle((0, 0))
    for sql in pipeline(left_rel, right_rel):
        assert isinstance(sql, str)
        assert sql.startswith("SELECT")
        breakpoint()
