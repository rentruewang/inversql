# Copyright (c) The InverSQL Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import numpy as np
from sklearn import tree
from sklearn.utils import validation

from inversql._utils import BoolArray, FloatArray, IntArray

from .exprs import CmpExpr, CmpOp, Expr

__all__ = ["TreeNode", "BranchNode", "LeafNode", "sklearn_binary_tree_to_nodes"]


@typing.dataclass_transform(kw_only_default=True)
def tree_dcls(cls):
    return dcls.dataclass(kw_only=True)(cls)


@tree_dcls
class TreeNode(abc.ABC):
    __match_args__: typing.ClassVar[tuple[str, ...]]

    parent: BranchNode | None = None
    """
    The parent of the current node.
    """

    @typing.final
    def predict(self, sample: FloatArray, /) -> int:
        node = self.walk(sample)
        return node.pred_idx

    @abc.abstractmethod
    def walk(self, sample: FloatArray, /) -> LeafNode:
        """
        Walk down the tree given the sample, and return the leaf node that is predicted.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def children(self) -> cabc.Iterator[TreeNode]:
        raise NotImplementedError

    def lineage(self) -> cabc.Generator[TreeNode]:
        """
        Get the lineage of the current node.
        """

        # Recursively calls parent's first s.t. the lineage is root first.
        if self.parent is not None:
            yield from self.parent.lineage()

        yield self

    @property
    def is_root(self) -> bool:
        return self.parent is None


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
    def walk(self, sample: FloatArray) -> LeafNode:
        # If `feat cmp threshold`, delegate to `self.yes`
        if self.expr.eval(sample):
            return self.yes.walk(sample)

        # else delegate to `self.no`.
        else:
            return self.no.walk(sample)

    @typing.override
    def children(self) -> cabc.Iterator[TreeNode]:
        yield self.yes
        yield self.no


@typing.final
@tree_dcls
class LeafNode(TreeNode):
    """
    The leaf node in a decision tree, corresponding to a category prediction.
    In this case, it corresponds to a binary condition, reflected in `.prediction`.
    """

    pred_idx: int
    "The feature that this leaf node predicts."

    @typing.override
    def walk(self, sample: FloatArray) -> LeafNode:
        # If this node is reached, already decided.
        return self

    @typing.override
    def children(self) -> cabc.Iterator[TreeNode]:
        return
        yield


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
            return LeafNode(pred_idx=pred_idx[idx])

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


def _process_prediction(value: FloatArray) -> tuple[IntArray, BoolArray]:
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

    value = value.squeeze(1)

    prediction = np.argmax(value, axis=-1)

    # All 0 or 1 guarantees to be one-hot, as it always sums to 1.
    is_one_hot = np.all((value == 0) | (value == 1), axis=-1)

    assert len(prediction) == len(is_one_hot), "Sanity check failed."
    return prediction, is_one_hot
