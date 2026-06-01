# Copyright (c) The InverSQL Authors - All Rights Reserved

import pytest

from inversql.exprs import (
    AndExpr,
    CmpExpr,
    CmpOp,
    Expr,
    OrExpr,
    feature_name,
    parse_feature_name,
    parse_sympy_expr,
)


@pytest.fixture(params=range(10))
def feat_idx(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def feat_name(feat_idx: int) -> str:
    return feature_name(feat_idx)


@pytest.fixture
def parsed_feature_name(feat_name: str) -> int:
    return parse_feature_name(feat_name)


def test_feature_name_parsing(feat_idx: int, parsed_feature_name: int):
    assert feat_idx == parsed_feature_name


@pytest.fixture
def cmp_1():
    return CmpExpr(1, CmpOp.EQ, 1)


@pytest.fixture
def cmp_2():
    return CmpExpr(2, CmpOp.GT, 1)


@pytest.fixture
def cmp_3():
    return CmpExpr(0, CmpOp.GT, 1)


@pytest.fixture
def and_1(cmp_1: CmpExpr, cmp_2: CmpExpr):
    return AndExpr(cmp_1, cmp_2)


@pytest.fixture
def or_1(and_1: CmpExpr, cmp_3: CmpExpr):
    return OrExpr(and_1, cmp_3)


def test_parse_sympy(or_1: Expr):
    sympy_expr = or_1.to_sympy()
    parsed = parse_sympy_expr(sympy_expr)
    assert parsed == or_1
