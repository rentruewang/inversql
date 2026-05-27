# Copyright (c) The InverSQL Authors - All Rights Reserved

import abc
import collections
import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import numpy as np
import sympy
from sklearn import tree
from sklearn.utils import validation

from inversql._utils import BoolArray, FloatArray, IntArray

from .exprs import AndExpr, CmpExpr, CmpOp, Expr, OrExpr

__all__ = ["TreeNode", "BranchNode", "LeafNode", "sklearn_binary_tree_to_nodes"]


@typing.dataclass_transform(kw_only_default=True)
def tree_dcls(cls):
    return dcls.dataclass(kw_only=True)(cls)


@dcls.dataclass(frozen=True, slots=True)
class AncestryPath:
    """
    Since paths are always [*branches, leaf], we organize both separately
    s.t. no type erasure to `TreeNode` need to happen (like `LeafNode.lineage`).
    """

    branches: collections.deque[BranchNode]
    "The lineage, from root to the closest parent (use `deque` for `.appendleft`)."

    prediction: LeafNode
    "The leaf node (signalling prediction)."

    @property
    def nodes(self) -> cabc.Iterator[TreeNode]:
        "The nodes from root to leaf."

        yield from self.branches
        yield self.prediction

    @property
    def exprs(self) -> Expr:
        "The aggregate expressions."

        return functools.reduce(AndExpr, [node.expr for node in self.branches])


@tree_dcls
class TreeNode(abc.ABC):
    __match_args__: typing.ClassVar[tuple[str, ...]]

    parent: BranchNode | None = None
    """
    The parent of the current node.
    """

    @typing.final
    def predict(self, sample: FloatArray, /) -> bool:
        node = self.walk(sample)
        return node.prediction.val

    @abc.abstractmethod
    def walk(self, sample: FloatArray, /) -> AncestryPath:
        """
        Walk down the tree given the sample, and return the leaf node that is predicted.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def children(self) -> cabc.Iterator[TreeNode]:
        raise NotImplementedError

    def lineage(self) -> cabc.Generator[BranchNode]:
        """
        Get the lineage of the current node (excluding self).
        """

        if self.parent is None:
            return

        # Recursively calls parent's first s.t. the lineage is root first.
        yield from self.parent.lineage()
        yield self.parent

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def truth_exprs(self):
        def sum_branch(nodes: cabc.Iterable[BranchNode]):
            exprs = [node.expr for node in nodes]
            return functools.reduce(AndExpr, exprs)

        # Each leaf is a product, and the truth values of entire tree is a sum of product.
        sum_exprs = [sum_branch(leaf.lineage()) for leaf in self.truth_leaves()]
        tree_expr = functools.reduce(OrExpr, sum_exprs)
        return tree_expr

    def truth_exprs_sympy(self, simplify: bool) -> sympy.Expr:
        return self.truth_exprs().to_sympy(simplify=simplify)

    @abc.abstractmethod
    def truth_leaves(self) -> cabc.Generator[LeafNode]:
        "Get the leaf nodes that are `True`."

        raise NotImplementedError


@typing.final
@tree_dcls
class BranchNode(TreeNode):
    """
    Branching based on the given features.

    Since `sklearn` decision tree (our only implementation currently) uses numeric data,
    this is the only currently supported comparison operator (only feature <= threshold).

    Additionally, we are storing `feat_idx` (feature index) rather than feature itself
    because scikit learn uses integer indices rather than feature names.

    Both of these may change in the future (or new node may be added).
    """

    expr: Expr
    "The expression to compare against."

    yes: TreeNode
    "The branch where `feat cmp threashold` is `True`."

    no: TreeNode
    "The branch where `feat cmp threashold` is `False`."

    def __post_init__(self) -> None:
        if not isinstance(self.yes, TreeNode):
            raise TypeError(f"{self.yes=} should be a tree node.")

        if not isinstance(self.no, TreeNode):
            raise TypeError(f"{self.no=} should be a tree node.")

        # Set the sub nodes' parent.
        self.yes.parent = self.no.parent = self

    @typing.override
    def walk(self, sample: FloatArray) -> AncestryPath:
        # Recursively calls the children, then append `self` to path.
        child = self.yes if self.expr.eval(sample) else self.no
        child_path = child.walk(sample)
        child_path.branches.appendleft(self)
        return child_path

    @typing.override
    def children(self) -> cabc.Iterator[TreeNode]:
        yield self.yes
        yield self.no

    @typing.override
    def truth_leaves(self) -> cabc.Generator[LeafNode]:
        yield from self.yes.truth_leaves()
        yield from self.no.truth_leaves()


@typing.final
@tree_dcls
class LeafNode(TreeNode):
    """
    The leaf node in a decision tree, corresponding to a category prediction.
    In this case, it corresponds to a binary condition, reflected in `.prediction`.
    """

    val: bool
    "The value that this leaf node predicts."

    @typing.override
    def walk(self, sample: FloatArray) -> AncestryPath:
        # If this node is reached, only need to store `self`.
        return AncestryPath(collections.deque(), self)

    @typing.override
    def children(self) -> cabc.Iterator[TreeNode]:
        return
        yield

    @typing.override
    def truth_leaves(self) -> cabc.Generator[LeafNode]:
        if self.val:
            yield self


def sklearn_binary_tree_to_nodes(clf: tree.DecisionTreeClassifier) -> TreeNode:
    "Convert a binary sklearn decision tree to our `TreeNode`."

    validation.check_is_fitted(clf)

    if clf.n_classes_ != 2:
        raise ValueError(
            f"Only binary decision trees are supported here. Got {clf.classes_}"
        )

    num_nodes = clf.tree_.node_count
    num_feats = clf.n_features_in_

    to_left: IntArray = np.array(clf.tree_.children_left)
    to_right: IntArray = np.array(clf.tree_.children_right)
    threshold: IntArray = np.array(clf.tree_.threshold)

    # Feature starts from 1.
    feat_idx: IntArray = np.array(clf.tree_.feature)
    assert np.all(feat_idx < num_feats)
    assert np.all((feat_idx >= 0) | (feat_idx == -2))

    # Leaf node metadata.
    pred_idx, is_one_hot = _process_prediction(clf.tree_.value)

    lengths = {num_nodes, len(to_left), len(to_right), len(threshold), len(feat_idx)}
    if lengths != {num_nodes}:
        raise AssertionError("Some fields have different length. Impossible.")

    def build_tree_node_rec(idx: int = 0) -> TreeNode:
        "Build tree node, recursively, starting at the default root `idx == 0`."

        assert (is_leaf := to_left[idx] < 0) == (to_right[idx] < 0)

        if is_leaf:
            assert feat_idx[idx] < 0, feat_idx[idx]
            assert is_one_hot[idx]
            return LeafNode(val=pred_idx[idx])

        # Internal nodes, create the sub-nodes then create the immutable `BranchNode`.

        yes_sub_node = build_tree_node_rec(idx=to_left[idx])
        no_sub_node = build_tree_node_rec(idx=to_right[idx])
        branch_expr = CmpExpr(
            feat_idx=int(feat_idx[idx]),
            cmp=CmpOp("<="),
            threshold=float(threshold[idx]),
        )

        return BranchNode(
            expr=branch_expr,
            yes=yes_sub_node,
            no=no_sub_node,
        )

    return build_tree_node_rec()


def _process_prediction(value: FloatArray) -> tuple[BoolArray, BoolArray]:
    """
    Return the `prediction, is_one_hot` for the given `value` array,
    whose shape is `nodes, outputs, classes`.
    """

    if value.ndim != 3:
        raise ValueError(
            f"The given value array doesn't have ndim == 3, {value.shape=}."
        )

    if value.shape[1] != 1:
        raise ValueError(
            "The given value array has multiple outputs. Not possible in `inversql`. "
            f"{value.shape=}."
        )

    if value.shape[2] != 2:
        raise ValueError("Only binary prediction task is supported right now.")

    value = value.squeeze(1)

    prediction = np.argmax(value, axis=-1)
    assert np.all((prediction == 0) | (prediction == 1))
    prediction = prediction.astype(bool)

    # All 0 or 1 guarantees to be one-hot, as it always sums to 1.
    is_one_hot = np.all((value == 0) | (value == 1), axis=-1)

    assert len(prediction) == len(is_one_hot), "Sanity check failed."
    return prediction, is_one_hot
