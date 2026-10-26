from __future__ import annotations

import unittest

import numpy as np
import pytest

from maml.models.dl import WeightedAverageLayer, WeightedSet2Set


class TestLayer(unittest.TestCase):
    def test_weighted_average(self):
        alpha = 0.4
        wal = WeightedAverageLayer(alpha=alpha)
        x = np.random.normal(size=(1, 5, 2))
        weights = np.array([0.1, 0.2, 0.1, 0.4, 0.5]).reshape((1, -1))
        indices = np.array([0, 0, 0, 1, 1]).reshape((1, -1))
        out = wal.call([x, weights, indices])

        x = x[0, :3, :]
        weights = weights[0, :3]
        exp_weights = weights**alpha
        out_np = np.sum(x * exp_weights[:, None], axis=0) / np.sum(exp_weights)
        np.testing.assert_array_almost_equal(out.numpy()[0][0], out_np)
        assert wal.get_config().get("alpha") == alpha
        assert wal.compute_output_shape([[1, 5, 2], [1, 5], [1, 5]]) == (1, None, 2)

    def test_weighted_set2set(self):
        maml_set = WeightedSet2Set(T=2, n_hidden=2)
        maml_set.build([[1, 3, 2], [1, 3], [1, 3]])
        x = np.random.normal(size=(1, 3, 2))
        weights = np.array([0.5, 0.5, 1], dtype=np.float64).reshape((1, -1))
        indices = np.array([0, 0, 0], dtype="int32").reshape((1, -1))
        res = maml_set([x, weights, indices])
        assert res.shape == (1, 1, 4)


@pytest.mark.parametrize(("use_bias", "unit_forget_bias"), [(True, False), (False, True)])
def test_weighted_set2set_bias_options(use_bias, unit_forget_bias):
    layer = WeightedSet2Set(T=2, n_hidden=3, use_bias=use_bias, unit_forget_bias=unit_forget_bias)
    layer.build([[1, 4, 2], [1, 4], [1, 4]])
    assert (layer.m_bias is not None) is use_bias
    assert (layer.recurrent_bias is not None) is use_bias
    x = np.random.default_rng(0).normal(size=(1, 4, 2)).astype("float32")
    weights = np.ones((1, 4), dtype="float32")
    indices = np.array([[0, 0, 1, 1]], dtype="int32")
    out = layer([x, weights, indices])
    assert out.shape == (1, 2, 6)
    assert layer.compute_output_shape([[1, 4, 2], [1, 4], [1, 4]]) == (1, None, 6)


def test_weighted_set2set_unit_forget_bias_init():
    layer = WeightedSet2Set(T=1, n_hidden=2)
    layer.build([[1, 3, 2], [1, 3], [1, 3]])
    # forget-gate block of the recurrent bias is initialised to ones, the rest to zeros
    np.testing.assert_allclose(layer.recurrent_bias.numpy(), [0, 0, 1, 1, 0, 0, 0, 0])


def test_weighted_set2set_config_roundtrip():
    layer = WeightedSet2Set(T=4, n_hidden=5, use_bias=False, activation="relu", name="s2s")
    config = layer.get_config()
    assert config["T"] == 4
    assert config["n_hidden"] == 5
    assert config["use_bias"] is False
    assert config["name"] == "s2s"
    clone = WeightedSet2Set.from_config(config)
    assert clone.T == 4
    assert clone.n_hidden == 5
    assert clone.use_bias is False
    assert clone.activation.__name__ == "relu"


if __name__ == "__main__":
    unittest.main()
