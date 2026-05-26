# Copyright (c) The InverSQL Authors - All Rights Reserved

import numpy as np
import pytest
from numpy import random
from sklearn import tree

from inversql.trees import BranchNode, TreeNode, sklearn_binary_tree_to_nodes


@pytest.fixture
def train_data(seed: int):
    random.seed(seed)
    x = random.randn(91, 31)
    y = random.randn(len(x)) > 0

    # Ensure that we have both categories.
    assert np.any(y)
    assert np.any(~y)

    return x, y


def _get_classifier(x: np.ndarray, y: np.ndarray):
    assert {*y[y >= 0]} == {0, 1}

    t = tree.DecisionTreeClassifier()
    t.fit(x, y)
    return t


def test_clf(train_data: tuple[np.ndarray, np.ndarray]):
    x, y = train_data
    clf = _get_classifier(x, y)

    assert clf.n_classes_ == 2

    # Need n-1 splits to handle n data, if each leaf node has 1 sample.
    assert clf.tree_.node_count <= len(y) - 1


def _create_node_from_clf(train_data: tuple[np.ndarray, np.ndarray]):
    x, y = train_data
    clf = _get_classifier(x, y)
    node = sklearn_binary_tree_to_nodes(clf)
    return clf, node


def test_create_tree_node(train_data: tuple[np.ndarray, np.ndarray]):
    _, node = _create_node_from_clf(train_data)
    assert isinstance(node, TreeNode)


def test_our_clf(train_data: tuple[np.ndarray, np.ndarray]):
    clf, node = _create_node_from_clf(train_data)
    assert isinstance(node, TreeNode)

    for sample, answer in zip(*train_data):
        clf_pred = clf.predict(sample[None]).squeeze()
        our_pred = node.predict(sample)
        assert clf_pred == answer
        assert our_pred == answer


def test_lineage(train_data: tuple[np.ndarray, np.ndarray]):
    _, node = _create_node_from_clf(train_data)
    assert isinstance(node, TreeNode)

    for sample, _ in zip(*train_data):
        leaf = node.walk(sample)
        lineage = list(leaf.lineage())
        assert lineage[-1] is leaf

        for branch_node in lineage[:-1]:
            assert isinstance(branch_node, BranchNode)
            assert branch_node.yes.parent is branch_node
            assert branch_node.no.parent is branch_node
