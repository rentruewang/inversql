# Copyright (c) The InverSQL Authors - All Rights Reserved

import numpy as np
import pytest
from numpy import random
from sklearn import tree

from inversql.trees import TreeNode, sklearn_binary_tree_to_nodes


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


def test_create_tree_node(train_data: tuple[np.ndarray, np.ndarray]):
    x, y = train_data
    clf = _get_classifier(x, y)
    node = sklearn_binary_tree_to_nodes(clf)
    assert isinstance(node, TreeNode)
