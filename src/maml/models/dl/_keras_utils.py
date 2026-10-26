"""Keras utils for deserialize activations and optimizers."""

from __future__ import annotations


def deserialize_keras_activation(activation):
    """
    Serialize keras activation.

    Args:
        activation (str, dict, or Activation object): activation to serialize

    Returns: Activation object
    """
    import tensorflow as tf

    if isinstance(activation, str):
        return tf.keras.activations.get(activation)

    if isinstance(activation, dict):
        return tf.keras.activations.deserialize(activation)

    if callable(activation):
        return activation

    raise ValueError("activation not recognized")


def deserialize_keras_optimizer(optimizer):
    """
    Serialize keras activation.

    Args:
        optimizer (str, dict, or optimizer object): optimizer to serialize

    Returns: Activation object
    """
    import tensorflow as tf

    if isinstance(optimizer, str):
        return tf.keras.optimizers.get(optimizer)

    if isinstance(optimizer, dict):
        return tf.keras.optimizers.deserialize(optimizer)

    if isinstance(optimizer, tf.keras.optimizers.Optimizer):
        return optimizer

    raise ValueError("optimizer not recognized")
