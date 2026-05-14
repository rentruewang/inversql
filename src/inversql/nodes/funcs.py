# Copyright (c) The InverSQL Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections import abc as cabc

from .nodes import QueryNode
import enum

__all__ = ["FilterNode", "SelectNode"]


class BooleanOp(enum.StrEnum):
    EQ = "=="
    NE = "!="
    GE = ">="
    GT = ">"
    LE = "<="
    LT = "<"

    @classmethod
    def parse(cls, text: str):
        match text:
            case "=" | "==":
                return cls.EQ
            case "<>" | "!=":
                return cls.NE
            case ">=":
                return cls.GE
            case ">":
                return cls.GT
            case "<=":
                return cls.LE
            case "<":
                return cls.LT
            case _:
                raise ValueError(f"Unknown {text=}.")


@dcls.dataclass(frozen=True)
class BooleanExpr:
    var: str
    op: BooleanOp
    var_or_num: str | typing.Any


@dcls.dataclass(frozen=True)
class FilterNode(QueryNode, key="FILTER"):
    """
    Implements the selection operator ("FILTER").
    """

    select: QueryNode
    """
    The source of the current query node.
    """

    where: BooleanExpr
    """
    The query conversion function.
    """

    @typing.override
    def __str__(self) -> str:
        return f"SELECT * FROM {self.select} WHERE {self.where}"

    @typing.override
    def children(self) -> tuple[QueryNode]:
        return (self.select,)


@dcls.dataclass(frozen=True)
class SelectNode(QueryNode, key="SELECT"):
    source: QueryNode
    selected: cabc.Sequence[str]

    @typing.override
    def __str__(self) -> str:
        return f"SELECT {', '.join(self.selected)} FROM {self.source}"

    @typing.override
    def children(self) -> tuple[QueryNode]:
        return (self.source,)
