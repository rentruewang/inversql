# Copyright (c) The InverSQL Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing, enum
from collections import abc as cabc

__all__ = ["TreeNode", "BranchNode"]


class TreeNode(abc.ABC):
    __match_args__: typing.ClassVar[tuple[str, ...]]

    @abc.abstractmethod
    def children(self) -> cabc.Iterator[TreeNode]:
        raise NotImplementedError


class CmpOp(enum.StrEnum):
    "The comparison operators."

    EQ = "=="
    NE = "!="
    GE = ">="
    GT = ">"
    LE = "<="
    LT = "<"


@dcls.dataclass(frozen=True)
class NumCmp:
    """
    Comparison of numeric data.

    Since `sklearn` decision tree (our only implementation currently) uses numeric data,
    this is the only currently supported comparison operator.
    This may change in the future.
    """

    feature: str
    "The name of the column in the `pd.DataFrame` (must be in `.columns`)."

    compare: CmpOp
    "How the left side compares to the right side."

    value: float
    "The value that the decision tree compares against."


@typing.final
@dcls.dataclass(frozen=True)
class BranchNode(TreeNode):
    condition: NumCmp
    "The condition of the branch."

    yes: TreeNode
    "The branch where condition is `True`."

    no: TreeNode
    "The branch where condition is `False`."

    @typing.override
    def children(self) -> cabc.Iterator[TreeNode]:
        yield self.yes
        yield self.no


@typing.final
@dcls.dataclass(frozen=True)
class LeafNode(TreeNode):
    """
    The leaf node in a decision tree, corresponding to a category prediction.
    In this case, it corresponds to a binary condition, reflected in `.value`.
    """

    value: bool
    "Each node corresponds to one of the values, yes or no."

    @typing.override
    def children(self) -> cabc.Iterator[TreeNode]:
        return
        yield
