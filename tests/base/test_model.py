from __future__ import annotations

import unittest

import numpy as np
import pytest
from monty.tempfile import ScratchDir

from maml.base import KerasModel, SKLModel, is_keras_model, is_sklearn_model


class TestBaseModel(unittest.TestCase):
    def test_sklmodel(self):
        from sklearn.linear_model import LinearRegression

        model = SKLModel(model=LinearRegression())
        x = [[1, 2], [3, 4]]
        y = [3, 7]
        model.fit(x, y)
        model.train(x, y)
        self.assertAlmostEqual(model.predict_objs([[4, 5]])[0], 9)
        with ScratchDir("."):
            model.save("test_model.sav")
            model.fit([[1, 2], [3, 4]], [6, 14])
            x = np.array([[4, 5]])
            self.assertAlmostEqual(model.predict_objs(x)[0], 18)
            model.load("test_model.sav")
            self.assertAlmostEqual(model.predict_objs(x)[0], 9)
            model2 = SKLModel.from_file("test_model.sav")
            self.assertAlmostEqual(model2.predict_objs(x)[0], 9)
            self.assertAlmostEqual(model2.evaluate([[4, 8], [8, 5]], [12, 13]), 1.0)
        assert is_sklearn_model(model)
        assert not is_keras_model(model)

    def test_keras_model(self):
        import tensorflow as tf

        model = KerasModel(model=tf.keras.Sequential([tf.keras.layers.Dense(1, input_dim=2)]))
        model.model.compile("adam", tf.keras.losses.MeanSquaredError())

        x = np.array([[1, 2], [3, 4]])
        y = np.array([3, 7]).reshape((-1, 1))
        model.fit(x, y)
        model.train(x, y)
        model.model.set_weights([np.array([[1.0], [1.0]]), np.array([0])])
        self.assertAlmostEqual(model.predict_objs(np.array([[4, 5]]))[0], 9)

        with ScratchDir("."):
            model.save("test_model.sav")
            model.fit(np.array([[1, 2], [3, 4]]), np.array([6, 14])[:, None])
            model.load("test_model.sav")
            self.assertAlmostEqual(model.predict_objs([[4, 5]])[0], 9)
            model2 = KerasModel.from_file("test_model.sav")
            self.assertAlmostEqual(model2.predict_objs([[4, 5]])[0], 9)
            self.assertAlmostEqual(model2.evaluate([[4, 8], [8, 5]], [12, 13]), 0.0)
        assert not is_sklearn_model(model)
        assert is_keras_model(model)


def test_train_val_targets_without_val_objs():
    from sklearn.linear_model import LinearRegression

    model = SKLModel(model=LinearRegression())
    with pytest.raises(ValueError, match="training objects are none"):
        model.train([[1, 2], [3, 4]], [3, 7], val_targets=[1])


def test_get_input_dim():
    from maml.base import DummyDescriber

    assert KerasModel.get_input_dim(None, None) is None
    assert KerasModel.get_input_dim(DummyDescriber(), 7) == 7


@pytest.fixture
def linear_keras_model():
    import tensorflow as tf

    inp = tf.keras.Input(shape=(2,))
    out = tf.keras.layers.Dense(1)(inp)
    model = KerasModel(model=tf.keras.Model(inp, out))
    model.model.compile("adam", "mse")
    return model


def test_keras_fit_validation_data_and_extra_kwargs(linear_keras_model, monkeypatch):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([[3.0], [7.0]])

    with pytest.raises(ValueError, match="Validation data incorrect"):
        linear_keras_model.fit(x, y, val_targets=y)

    captured = {}

    def fake_get_validation_data(val_features, val_targets, **val_kwargs):
        captured["val_kwargs"] = val_kwargs
        return val_features, val_targets

    monkeypatch.setattr(linear_keras_model, "_get_validation_data", fake_get_validation_data)
    # `not_a_fit_arg` is not an argument of keras Model.fit, so it is routed to _get_validation_data
    linear_keras_model.fit(x, y, val_features=x, val_targets=y, epochs=1, verbose=0, not_a_fit_arg=3)
    assert captured["val_kwargs"] == {"not_a_fit_arg": 3}
    history = linear_keras_model.model.history.history
    assert "val_loss" in history
    assert len(history["loss"]) == 1


def test_keras_train_with_validation_objs(linear_keras_model):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([[3.0], [7.0]])
    linear_keras_model.train(x, y, val_objs=x, val_targets=y, epochs=2, verbose=0)
    assert len(linear_keras_model.model.history.history["val_loss"]) == 2


if __name__ == "__main__":
    unittest.main()
