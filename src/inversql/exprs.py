# Copyright (c) The InverSQL Authors - All Rights Reserved

import abc
import dataclasses as dcls
import enum
import functools
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
    "parse_feature_name",
]


_FEATURE_REGEX = re.compile(r"feature_(\d+)")


def feature_name(idx: int) -> str:
    return f"feature_{idx}"


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

    def to_sympy(self, simplify: bool = False) -> sympy.Expr:
        expr = self._to_sympy(simplify)

        if simplify:
            expr = sympy.simplify(expr)

        return expr

    @abc.abstractmethod
    def _to_sympy(self, simplify: bool) -> sympy.Expr:
        "Convert to a `sympy.Expr`."
        raise NotImplementedError


@typing.no_type_check
def parse_sympy_expr(expr: sympy.Expr) -> Expr:
    """
    Parse the given sympy expression.
    Convert the `sympy.Expr` into our `Expr` (only AND / OR / comparison),
    this is fine because those are the only features we use from `sympy`.

    Due to our limitation (that our expressions are binary), AND / OR are `fold`ed.

    At some point we might transition to full `sympy`.
    """

    args = expr.args

    match expr:
        case sympy.And():
            return functools.reduce(AndExpr, map(parse_sympy_expr, args))
        case sympy.Or():
            return functools.reduce(OrExpr, map(parse_sympy_expr, args))

        # We don't have a `NotExpr`, but directly invoke the `__invert__`.
        case sympy.Not():
            [expr] = args

            return ~parse_sympy_expr(expr)

        case (
            sympy.Eq() | sympy.Ne() | sympy.Ge() | sympy.Gt() | sympy.Le() | sympy.Lt()
        ):
            left, right = args
            assert isinstance(left, sympy.Symbol), left
            return CmpExpr(
                feat_idx=parse_feature_name(str(left)),
                cmp=CmpOp.from_sympy_type(type(expr)),
                threshold=float(right),
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

    @property
    def sympy_type(self):
        match self:
            case CmpOp.EQ:
                return sympy.Eq
            case CmpOp.NE:
                return sympy.Ne
            case CmpOp.GE:
                return sympy.Ge
            case CmpOp.GT:
                return sympy.Gt
            case CmpOp.LE:
                return sympy.Le
            case CmpOp.LT:
                return sympy.Lt

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

    def __post_init__(self):
        if not isinstance(self.feat_idx, int) or self.feat_idx < 0:
            raise ValueError(f"{self.feat_idx=} not an integer >= 0.")

        if not isinstance(self.cmp, CmpOp):
            raise TypeError(f"{self.cmp=} should be `CmpOp`, got {type(self.cmp)=}.")

        if not isinstance(self.threshold, int | float):
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
        return self.cmp.sympy_type(symbol, self.threshold)


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

    def __eq__(self, other: object):
        if isinstance(other, AndExpr):
            in_order = self.left == other.left and self.right == other.right
            inverted = self.left == other.right and self.right == other.left
            return in_order or inverted
        return NotImplemented

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

    def __eq__(self, other: object):
        if isinstance(other, OrExpr):
            in_order = self.left == other.left and self.right == other.right
            inverted = self.left == other.right and self.right == other.left
            return in_order or inverted
        return NotImplemented

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
