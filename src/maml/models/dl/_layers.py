"""Deep learning layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import tensorflow as tf
import tensorflow.keras.backend as kb
from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Layer

if TYPE_CHECKING:
    from collections.abc import Sequence


class WeightedAverageLayer(Layer):
    r"""
    Weight average the features using weights.

    result= \sum{w_i^a * value_i} / \sum{w_i^a}

    """

    def __init__(self, alpha: float = 1.0, **kwargs):
        """
        Args:
            alpha (float): exponent in weighting.
            **kwargs: pass to construction of Layer in tensorflow.keras.layers.
        """
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape: Sequence) -> None:
        """
        Build the layer.

        Args:
            input_shape (tuple): input shape tuple
        """
        self.built = True

    def call(self, inputs: Sequence, mask: tf.Tensor | None = None) -> tf.Tensor:
        """
        Core logic of the layer.

        Args:
            inputs (tuple): input tuple of length 3
            mask (tf.Tensor): not used here
        """
        # prop: [1, n, n_feature]
        # weight: [1, n]
        # index: [1, n]
        prop, weights, index = inputs
        expo_weights = weights**self.alpha
        prop = prop * expo_weights[:, :, None]
        prop_sum = self.reduce_sum(prop, index, perm=[1, 0, 2])
        weight_sum = self.reduce_sum(expo_weights, index, perm=[1, 0])
        return prop_sum / weight_sum[:, :, None]

    @staticmethod
    def reduce_sum(prop: tf.Tensor, index: tf.Tensor, perm: Sequence) -> tf.Tensor:
        """
        Reduce sum the tensors using index.

        Args:
            prop (tf.Tensor): tensor with shape [1, n, ...]
            index (tf.Tensor): integer tensor with shape [1, n]
            perm (list): permutation for transpose.
        """
        index = tf.reshape(index, (-1,))
        prop = tf.transpose(a=prop, perm=perm)
        out = tf.math.segment_sum(prop, index)
        return tf.transpose(a=out, perm=perm)

    @staticmethod
    def compute_output_shape(input_shape: Sequence) -> tuple:
        """
        Compute output shape from input shape
        Args:
            input_shape (tuple/list): input shape list.
        """
        prop_shape = input_shape[0]
        return prop_shape[0], None, prop_shape[-1]

    def get_config(self) -> dict:
        """Get layer configuration."""
        config = {"alpha": self.alpha}
        base_config = super().get_config()
        config.update(base_config)
        return config  # type: ignore


class WeightedSet2Set(Layer):
    """
    This is a modified version from megnet.layers.readout.Set2Set.
    Here, in addition to taking features and indices as inputs, we also
    take a weight tensor. The input to the core logic is
    [features, weights, indices].
    """

    def __init__(
        self,
        T=3,
        n_hidden=512,
        activation=None,
        activation_lstm="tanh",
        recurrent_activation="hard_sigmoid",
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        use_bias=True,
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        """
        Args:
            T: (int) recurrent step
            n_hidden: (int) number of hidden units
            activation: (str or object) activation function
            activation_lstm: (str or object) activation function for lstm
            recurrent_activation: (str or object) activation function for recurrent step
            kernel_initializer: (str or object) initializer for kernel weights
            recurrent_initializer: (str or object) initializer for recurrent weights
            bias_initializer: (str or object) initializer for biases
            use_bias: (bool) whether to use biases
            unit_forget_bias: (bool) whether to use basis in forget gate
            kernel_regularizer: (str or object) regularizer for kernel weights
            recurrent_regularizer: (str or object) regularizer for recurrent weights
            bias_regularizer: (str or object) regularizer for biases
            kernel_constraint: (str or object) constraint for kernel weights
            recurrent_constraint: (str or object) constraint for recurrent weights
            bias_constraint:(str or object) constraint for biases
            kwargs: other inputs for keras Layer class.
        """
        super().__init__(**kwargs)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.activation_lstm = activations.get(activation_lstm)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.unit_forget_bias = unit_forget_bias
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.T = T
        self.n_hidden = n_hidden

    def build(self, input_shape: Sequence) -> None:
        """
        Build the output shape from input shapes
        Args:
            input_shape (tuple/list): input shape list.
        """
        feature_shape, _weight_shape, _index_shape = input_shape
        self.m_weight = self.add_weight(
            shape=(feature_shape[-1], self.n_hidden),
            initializer=self.kernel_initializer,
            name="x_to_m_weight",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.m_bias = self.add_weight(
                shape=(self.n_hidden,),
                initializer=self.bias_initializer,
                name="x_to_m_bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.m_bias = None

        self.recurrent_kernel = self.add_weight(
            shape=(2 * self.n_hidden, 4 * self.n_hidden),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return kb.concatenate(
                        [
                            self.bias_initializer((self.n_hidden,), *args, **kwargs),
                            initializers.Ones()((self.n_hidden,), *args, **kwargs),
                            self.bias_initializer((self.n_hidden * 2,), *args, **kwargs),
                        ]
                    )

            else:
                bias_initializer = self.bias_initializer
            self.recurrent_bias = self.add_weight(
                shape=(self.n_hidden * 4,),
                name="recurrent_bias",
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.recurrent_bias = None
        self.built = True

    def compute_output_shape(self, input_shape):
        """
        Compute output shapes from input shapes
        Args:
            input_shape (sequence of tuple): input shapes.

        Returns: sequence of tuples output shapes

        """
        feature_shape, _index_shape, _ = input_shape
        return feature_shape[0], None, 2 * self.n_hidden

    def call(self, inputs):
        """
        Core logic of the layer.

        Args:
            inputs (tuple): input tuple of length 3
        """
        features, weights, feature_graph_index = inputs
        feature_graph_index = tf.reshape(feature_graph_index, (-1,))
        counts = tf.math.bincount(feature_graph_index)
        n_count = tf.shape(counts)[0]
        n_feature = tf.shape(features)[0]
        m = kb.dot(features, self.m_weight)
        if self.use_bias:
            m += self.m_bias

        self.h = tf.zeros(tf.stack([n_feature, n_count, self.n_hidden]))
        self.c = tf.zeros(tf.stack([n_feature, n_count, self.n_hidden]))
        q_star = tf.zeros(tf.stack([n_feature, n_count, 2 * self.n_hidden]))
        for _i in range(self.T):
            self.h, _c = self._lstm(q_star, self.c)
            e_i_t = tf.reduce_sum(input_tensor=m * tf.repeat(self.h, repeats=counts, axis=1), axis=-1)
            maxes = tf.math.segment_max(e_i_t[0], feature_graph_index)
            maxes = tf.repeat(maxes, repeats=counts)
            e_i_t -= tf.expand_dims(maxes, axis=0)
            exp = tf.exp(e_i_t) * weights
            seg_sum = tf.transpose(
                a=tf.math.segment_sum(tf.transpose(a=exp, perm=[1, 0]), feature_graph_index),
                perm=[1, 0],
            )
            seg_sum = tf.expand_dims(seg_sum, axis=-1)
            intermediate = tf.repeat(seg_sum, repeats=counts, axis=1)
            a_i_t = exp / intermediate[..., 0]
            r_t = tf.transpose(
                a=tf.math.segment_sum(
                    tf.transpose(a=tf.multiply(m, a_i_t[:, :, None]), perm=[1, 0, 2]),
                    feature_graph_index,
                ),
                perm=[1, 0, 2],
            )
            q_star = kb.concatenate([self.h, r_t], axis=-1)
        return q_star

    def _lstm(self, h, c):
        # lstm implementation here
        z = kb.dot(h, self.recurrent_kernel)
        if self.use_bias:
            z += self.recurrent_bias
        z0 = z[:, :, : self.n_hidden]
        z1 = z[:, :, self.n_hidden : 2 * self.n_hidden]
        z2 = z[:, :, 2 * self.n_hidden : 3 * self.n_hidden]
        z3 = z[:, :, 3 * self.n_hidden :]
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        # print(z.shape, f.shape, c.shape, z2.shape)
        c = f * c + i * self.activation_lstm(z2)
        o = self.recurrent_activation(z3)
        h = o * self.activation_lstm(c)
        return h, c

    def get_config(self):
        """
         Part of keras layer interface, where the signature is converted into a dict
        Returns:
            configurational dictionary.
        """
        config = {
            "T": self.T,
            "n_hidden": self.n_hidden,
            "activation": activations.serialize(self.activation),
            "activation_lstm": activations.serialize(self.activation_lstm),
            "recurrent_activation": activations.serialize(self.recurrent_activation),
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "recurrent_initializer": initializers.serialize(self.recurrent_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "use_bias": self.use_bias,
            "unit_forget_bias": self.unit_forget_bias,
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "recurrent_regularizer": regularizers.serialize(self.recurrent_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(self.recurrent_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
