# Copyright (c) The InverSQL Authors - All Rights Reserved

import abc
import dataclasses as dcls
import enum
import typing
from collections import abc as cabc

from inversql._utils import FloatArray

__all__ = ["CmpOp", "Expr", "CmpExpr", "AndExpr", "OrExpr", "DontCareExpr"]


@typing.dataclass_transform(frozen_default=True)
def expr_dcls(cls):
    return dcls.dataclass(frozen=True)(cls)


@expr_dcls
class Expr(abc.ABC):
    """
    Base class for boolean expression.
    """

    @abc.abstractmethod
    def __invert__(self) -> Expr:
        raise NotImplementedError

    def __and__(self, other: Expr) -> Expr:
        return AndExpr(self, other)

    def __or__(self, other: Expr) -> Expr:
        return OrExpr(self, other)

    @abc.abstractmethod
    def eval(self, sample: FloatArray) -> bool:
        raise NotImplementedError


@expr_dcls
class DontCareExpr(Expr):
    """
    Singalling the don't care values.

    Evals to `NotImplemented` means `True` in `AND`, and `False` in `OR` (default values).
    """

    @typing.override
    def __invert__(self) -> Expr:
        return self

    @typing.override
    def eval(self, sample: FloatArray) -> bool:
        return NotImplemented


class CmpOp(enum.StrEnum):
    "The comparison operators."

    EQ = "=="
    NE = "!="
    GE = ">="
    GT = ">"
    LE = "<="
    LT = "<"

    def __call__(self, left: float, right: float) -> bool:
        match self:
            case CmpOp.EQ:
                return left == right
            case CmpOp.NE:
                return left != right
            case CmpOp.GE:
                return left >= right
            case CmpOp.GT:
                return left > right
            case CmpOp.LE:
                return left <= right
            case CmpOp.LT:
                return left < right

    def __invert__(self):
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

        if not isinstance(self.threshold, float):
            raise TypeError(f"{self.threshold=} should be float.")

    @typing.override
    def __invert__(self) -> typing.Self:
        return dcls.replace(self, cmp=~self.cmp)

    @typing.override
    def eval(self, sample: FloatArray) -> bool:
        "Evaluate the current expression to true or false."
        return self.cmp(sample[self.feat_idx], self.threshold)


@expr_dcls
class AndExpr(Expr):
    "`left & right` expression."

    left: Expr
    "The LHS expression."

    right: Expr
    "The RHS expression."

    @typing.override
    def __invert__(self) -> Expr:
        return ~self.left or ~self.right

    @typing.override
    def eval(self, sample: FloatArray) -> bool:
        left = self.left.eval(sample)
        right = self.right.eval(sample)

        return _binop_handle_notimplemented(left, right, lambda l, r: l and r)


@expr_dcls
class OrExpr(Expr):
    "`left | right` expression."

    left: Expr
    "The LHS expression."

    right: Expr
    "The RHS expression."

    @typing.override
    def __invert__(self) -> Expr:
        return ~self.left and ~self.right

    @typing.override
    def eval(self, sample: FloatArray) -> bool:
        left = self.left.eval(sample)
        right = self.right.eval(sample)

        return _binop_handle_notimplemented(left, right, lambda l, r: l or r)


def _binop_handle_notimplemented(
    left: bool, right: bool, func: cabc.Callable[[bool, bool], bool]
) -> bool:
    """
    Check if one side is `NotImplemented`, then return the otherside.
    If both sides are `NotImplemented`, return `NotImplemented`.
    If both sides are given, ues `func` to evalute the boolean expression.
    `NotImplemented` values are given by `DontCareExpr.eval`.
    """

    if left is NotImplemented and right is NotImplemented:
        return NotImplemented

    if left is NotImplemented:
        return right

    if right is NotImplemented:
        return left

    return func(left, right)
