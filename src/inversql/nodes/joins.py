# Copyright (c) The InverSQL Authors - All Rights Reserved

import dataclasses as dcls
import typing

from .nodes import QueryNode

__all__ = ["JoinNode"]


@dcls.dataclass(frozen=True)
class JoinNode(QueryNode, key="JOIN"):
    """
    The node representing joining two tables.
    """

    left: QueryNode
    """
    The LHS of the joining expression.
    """

    right: QueryNode
    """
    The RHS of the joining expression.
    """

    left_on: str
    """
    Left column to join on.

    """

    right_on: str
    """
    Right column to join on.
    """

    how: str = ""
    """
    How the tables are joined.
    """

    @typing.override
    def __str__(self) -> str:
        return f"SELECT * FROM {self.left} {self.how} JOIN {self.right} ON {self.left_on} = {self.right_on}"

    @typing.override
    def children(self) -> tuple[QueryNode, QueryNode]:
        return self.left, self.right
