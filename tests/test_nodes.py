# Copyright (c) The InverSQL Authors - All Rights Reserved

import pypika
import pytest

from inversql.nodes import FilterNode, InputNode, JoinNode, QueryNode, SelectNode


def _input_sources():
    yield "table"


def _to_select():
    yield "abc"
    yield "de"
    yield "f"


def _conditions():
    yield "a > 0"
    yield "a = b"


def _on_fields():
    yield "aa"
    yield "ab"


def _join_type():
    yield "INNER"
    yield "OUTER"


def _input_node(table: str | pypika.Table):
    return InputNode(table)


def _select_node(input_node: QueryNode, selected: str):
    return SelectNode(input_node, selected=selected)


def _filter_node(input_node: QueryNode, condition: str):
    return FilterNode(input_node, where=condition)


def _join_node(input_node: QueryNode, left_on: str, right_on: str, how: str):
    return JoinNode(
        left=input_node, right=input_node, left_on=left_on, right_on=right_on, how=how
    )


@pytest.fixture(params=_input_sources())
def table(request) -> str:
    return request.param


@pytest.fixture
def input_node(table):
    return _input_node(table)


@pytest.fixture(params=_to_select())
def selected(request) -> str:
    return request.param


@pytest.fixture(params=_conditions())
def condition(request) -> str:
    return request.param


@pytest.fixture(params=_on_fields())
def on_fields(request) -> tuple[str, str]:
    return request.param


@pytest.fixture(params=_join_type())
def how(request) -> str:
    return request.param


def test_input(table):
    _input_node(table)


def test_input_sql(table):
    str(_input_node(table))


def test_filter(input_node, condition):
    _filter_node(input_node, condition=condition)


def test_filter_sql(input_node, condition):
    str(_filter_node(input_node, condition=condition))


def test_select(input_node, selected):
    _select_node(input_node=input_node, selected=selected)


def test_select_sql(input_node, selected):
    str(_select_node(input_node=input_node, selected=selected))


def test_join(input_node, on_fields, how):
    left_on, right_on = on_fields
    _join_node(input_node=input_node, left_on=left_on, right_on=right_on, how=how)


def test_join_sql(input_node, on_fields, how):
    left_on, right_on = on_fields
    str(_join_node(input_node=input_node, left_on=left_on, right_on=right_on, how=how))
