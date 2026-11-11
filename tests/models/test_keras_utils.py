from __future__ import annotations

import pytest
import tensorflow as tf

from maml.models.dl._keras_utils import deserialize_keras_activation, deserialize_keras_optimizer


@pytest.mark.parametrize(
    "activation",
    [
        "relu",
        {"module": "builtins", "class_name": "function", "config": "relu", "registered_name": "function"},
        tf.keras.activations.relu,
    ],
)
def test_deserialize_activation(activation):
    act = deserialize_keras_activation(activation)
    assert act(tf.constant([-1.0, 2.0])).numpy().tolist() == [0.0, 2.0]


def test_deserialize_activation_callable_passthrough():
    """Regression: tf.keras.activations.Activation does not exist in Keras 3 (AttributeError)."""

    def my_act(x):
        return 2 * x

    assert deserialize_keras_activation(my_act) is my_act


def test_deserialize_activation_invalid():
    with pytest.raises(ValueError, match="activation not recognized"):
        deserialize_keras_activation(1)


def test_deserialize_optimizer_str_and_dict():
    adam = deserialize_keras_optimizer("adam")
    assert isinstance(adam, tf.keras.optimizers.Adam)
    config = tf.keras.optimizers.serialize(tf.keras.optimizers.SGD(learning_rate=0.05))
    sgd = deserialize_keras_optimizer(config)
    assert isinstance(sgd, tf.keras.optimizers.SGD)
    assert float(sgd.learning_rate) == pytest.approx(0.05)


def test_deserialize_optimizer_instance_passthrough():
    opt = tf.keras.optimizers.RMSprop()
    assert deserialize_keras_optimizer(opt) is opt


def test_deserialize_optimizer_invalid():
    with pytest.raises(ValueError, match="optimizer not recognized"):
        deserialize_keras_optimizer(1.0)
