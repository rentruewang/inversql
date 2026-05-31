# Copyright (c) The InverSQL Authors - All Rights Reserved

import abc
import dataclasses as dcls
import enum
import operator
import re
import typing
from collections import abc as cabc

import sympy

from inversql._utils import FloatArray

__all__ = [
    "CmpOp",
    "Expr",
    "CmpExpr",
    "AndExpr",
    "OrExpr",
    "DontCareExpr",
    "feature_name",
]


def feature_name(idx: int) -> str:
    return f"feature_{idx}"


_FEATURE_REGEX = re.compile("feature_(\d+)")


def parse_feature_name(name: str) -> int:
    if m := _FEATURE_REGEX.match(name):
        return int(m.group(1))

    raise ValueError("Cannot parse the feature.")


@typing.dataclass_transform()
def expr_dcls(cls):
    return dcls.dataclass()(cls)


@expr_dcls
class Expr(abc.ABC):
    """
    Base class for boolean expression.
    """

    @abc.abstractmethod
    def __invert__(self) -> Expr:
        "`~self`"
        raise NotImplementedError

    def __and__(self, other: Expr) -> Expr:
        "`self & other`"
        return AndExpr(left=self, right=other)

    def __or__(self, other: Expr) -> Expr:
        "`self | other`"
        return OrExpr(left=self, right=other)

    @abc.abstractmethod
    def eval(self, sample: FloatArray) -> bool:
        "Evaluate with `sample` to give `True` or `False`."
        raise NotImplementedError

    def to_sympy(self, simplify: bool) -> sympy.Expr:
        expr = self._to_sympy(simplify)

        if simplify:
            expr = sympy.simplify(expr)

        return expr

    @abc.abstractmethod
    def _to_sympy(self, simplify: bool) -> sympy.Expr:
        "Convert to a `sympy.Expr`."
        raise NotImplementedError


def parse_sympy_expr(expr: sympy.Expr) -> Expr:
    args = expr.args

    if not _is_expr_list(args):
        raise TypeError(
            f"{expr.args} are not symbolic. Impossible for `inversql` usage."
        )

    match expr:
        case sympy.And():
            left, right = args

            return AndExpr(
                left=parse_sympy_expr(left),
                right=parse_sympy_expr(right),
            )
        case sympy.Or():
            left, right = args

            return OrExpr(
                left=parse_sympy_expr(left),
                right=parse_sympy_expr(right),
            )

        # We don't have a `NotExpr`, but directly invoke the `__invert__`.
        case sympy.Not():
            [expr] = args

            return ~parse_sympy_expr(expr)

        case (
            sympy.Eq() | sympy.Ne() | sympy.Ge() | sympy.Gt() | sympy.Le() | sympy.Lt()
        ):
            left, right = args
            assert isinstance(left, sympy.Symbol)
            assert isinstance(right, int | float | bool)
            return CmpExpr(
                feat_idx=parse_feature_name(str(left)),
                cmp=CmpOp.from_sympy_type(type(expr)),
                threshold=right,
            )

    raise ValueError(f"Unsupported expression: {expr}")


def _is_expr_list(args, /) -> typing.TypeIs[cabc.Sequence[sympy.Expr]]:
    if not isinstance(args, cabc.Sequence):
        return False
    if any(not isinstance(arg, sympy.Expr) for arg in args):
        return False
    return True


@expr_dcls
class DontCareExpr(Expr):
    """
    Singalling the don't care values.

    Evals to `NotImplemented` means `True` in `AND`, and `False` in `OR` (default values).
    """

    @typing.override
    def __repr__(self) -> str:
        return f"x"

    @typing.override
    def __invert__(self) -> Expr:
        return self

    @typing.override
    def eval(self, sample: FloatArray) -> bool:
        return NotImplemented

    @typing.override
    def _to_sympy(self, simplify: bool) -> sympy.Expr:
        return NotImplemented


class CmpOp(enum.StrEnum):
    "The comparison operators."

    EQ = "=="
    NE = "!="
    GE = ">="
    GT = ">"
    LE = "<="
    LT = "<"

    def __invert__(self) -> CmpOp:
        match self:
            case CmpOp.EQ:
                return CmpOp.NE
            case CmpOp.NE:
                return CmpOp.EQ
            case CmpOp.GE:
                return CmpOp.LT
            case CmpOp.GT:
                return CmpOp.LE
            case CmpOp.LE:
                return CmpOp.GT
            case CmpOp.LT:
                return CmpOp.GE

    @property
    def op(self):
        match self:
            case CmpOp.EQ:
                return operator.eq
            case CmpOp.NE:
                return operator.ne
            case CmpOp.GE:
                return operator.ge
            case CmpOp.GT:
                return operator.gt
            case CmpOp.LE:
                return operator.le
            case CmpOp.LT:
                return operator.lt

    @classmethod
    def from_sympy_type(cls, typ: type[sympy.Expr]) -> CmpOp:
        if typ == sympy.Eq:
            return cls.EQ

        if typ == sympy.Ne:
            return cls.NE

        if typ == sympy.Ge:
            return cls.GE

        if typ == sympy.Gt:
            return cls.GT

        if typ == sympy.Le:
            return cls.LE

        if typ == sympy.Lt:
            return cls.LT

        raise ValueError(f"Operator type {typ} has no `CmpOp` correspondence.")


@expr_dcls
class CmpExpr(Expr):
    "The boolean expression that can be true or false."

    feat_idx: int
    "The node predicts the branch based on the feature at `feat_idx`."

    cmp: CmpOp
    "The comparison operator. Sklearn uses <= by default."

    threshold: float
    "The value that the feature at `feat_idx` compares against."

    features: list[str] | None = None
    """
    If given, the features are named. Else use "feature_{i}".
    """

    def __post_init__(self):
        if not isinstance(self.feat_idx, int) or self.feat_idx < 0:
            raise ValueError(f"{self.feat_idx=} not an integer >= 0.")

        if not isinstance(self.cmp, CmpOp):
            raise TypeError(f"{self.cmp=} should be `CmpOp`, got {type(self.cmp)=}.")

        if not isinstance(self.threshold, float):
            raise TypeError(f"{self.threshold=} should be float.")

    @typing.override
    def __repr__(self) -> str:
        return f"{feature_name(self.feat_idx)} {self.cmp.value} {self.threshold}"

    @typing.override
    def __invert__(self) -> typing.Self:
        return dcls.replace(self, cmp=~self.cmp)

    @typing.override
    def eval(self, sample: FloatArray) -> bool:
        return self.cmp.op(sample[self.feat_idx], self.threshold)

    @typing.override
    def _to_sympy(self, simplify: bool) -> sympy.Expr:
        symbol = sympy.Symbol(feature_name(self.feat_idx))
        return self.cmp.op(symbol, self.threshold)


@expr_dcls
class AndExpr(Expr):
    "`left & right` expression."

    left: Expr
    "The LHS expression."

    right: Expr
    "The RHS expression."

    @typing.override
    def __repr__(self) -> str:
        return f"({self.left}) & ({self.right})"

    @typing.override
    def __invert__(self) -> Expr:
        return ~self.left or ~self.right

    @typing.override
    def eval(self, sample: FloatArray) -> bool:
        left = self.left.eval(sample)
        right = self.right.eval(sample)
        return _binop_handle_notimplemented(left, right, lambda l, r: l and r)

    @typing.override
    def _to_sympy(self, simplify: bool) -> sympy.Expr:
        left = self.left.to_sympy(simplify)
        right = self.right.to_sympy(simplify)
        return _binop_handle_notimplemented(left, right, operator.and_)


@expr_dcls
class OrExpr(Expr):
    "`left | right` expression."

    left: Expr
    "The LHS expression."

    right: Expr
    "The RHS expression."

    @typing.override
    def __repr__(self) -> str:
        return f"({self.left}) | ({self.right})"

    @typing.override
    def __invert__(self) -> Expr:
        return ~self.left and ~self.right

    @typing.override
    def eval(self, sample: FloatArray) -> bool:
        left = self.left.eval(sample)
        right = self.right.eval(sample)
        return _binop_handle_notimplemented(left, right, lambda l, r: l or r)

    @typing.override
    def _to_sympy(self, simplify: bool) -> sympy.Expr:
        left = self.left.to_sympy(simplify)
        right = self.right.to_sympy(simplify)
        return _binop_handle_notimplemented(left, right, operator.or_)


def _binop_handle_notimplemented(
    left, right, func: cabc.Callable[[object, object], typing.Any]
):
    """
    Check if one side is `NotImplemented`, then return the otherside.
    If both sides are `NotImplemented`, return `NotImplemented`.
    If both sides are given, ues `func` to evalute the boolean expression.

    This handles both sympy symbols and boolean evaluation.

    `NotImplemented` values are given by `DontCareExpr.*`.
    """

    if left is NotImplemented and right is NotImplemented:
        return NotImplemented

    if left is NotImplemented:
        return right

    if right is NotImplemented:
        return left

    return func(left, right)
