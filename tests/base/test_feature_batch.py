from __future__ import annotations

import unittest

import numpy as np

from maml.base import get_feature_batch
from maml.base._feature_batch import stack_first_dim


class TestFeatureBatch(unittest.TestCase):
    def test_get_feature_batch(self):
        x = [np.random.normal(size=(10, 20)), np.random.normal(size=(10, 20)), np.random.normal(size=(10, 20))]
        y = get_feature_batch("stack_first_dim")(x)
        assert y.shape == (3, 10, 20)
        y = get_feature_batch(stack_first_dim)(x)
        assert y.shape == (3, 10, 20)
        self.assertRaises(KeyError, get_feature_batch, fb_name="unknown_batch")

        y = get_feature_batch()(x)
        np.testing.assert_almost_equal(y[0], x[0])


def test_stack_padded():
    x = [np.array([1.0, 2.0, 3.0]), np.array([4.0]), np.array([5.0, 6.0])]
    y = get_feature_batch("stack_padded")(x)
    np.testing.assert_allclose(y, [[1, 2, 3], [4, 0, 0], [5, 6, 0]])


def test_pandas_concat():
    import pandas as pd

    dfs = [pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [3]})]
    out = get_feature_batch("pandas_concat")(dfs)
    assert out.index.names == ["input_index", None]
    assert out.xs(1, level="input_index")["a"].tolist() == [3]
    assert out.xs(0, level="input_index")["a"].tolist() == [1, 2]


if __name__ == "__main__":
    unittest.main()
