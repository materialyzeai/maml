"""Multi-layer perceptron models."""

from __future__ import annotations

from maml.base import BaseDescriber, KerasModel


def construct_mlp(
    input_dim: int,
    n_neurons: tuple = (64, 64),
    activation: str = "relu",
    n_targets: int = 1,
    is_classification: bool = False,
    optimizer: str = "adam",
    loss: str = "mse",
    compile_metrics: tuple = (),
):
    """
    Constructor for multi-layer perceptron models.

    Args:
        input_dim (int): input dimension, i.e., feature dimension
        n_neurons (tuple): list of hidden neuron sizes
        activation (str): activation function
        n_targets (int): number of targets
        is_classification (bool): whether the target is a classification problem
        optimizer (str): optimizer
        loss (str): loss function
        compile_metrics (tuple): metrics to evaluate during epochs
    """
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Model

    inp = Input(shape=(input_dim,))
    out_ = inp
    for n_neuron in n_neurons:
        out_ = Dense(n_neuron, activation=activation)(out_)

    if is_classification:
        final_act: str | None = "sigmoid"
    else:
        final_act = None
    out = Dense(n_targets, activation=final_act)(out_)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer, loss, metrics=compile_metrics)
    return model


class MLP(KerasModel):
    """This class implements the multi-layer perceptron models."""

    def __init__(
        self,
        input_dim: int | None = None,
        describer: BaseDescriber | None = None,
        n_neurons: tuple = (64, 64),
        activation: str = "relu",
        n_targets: int = 1,
        is_classification: bool = False,
        optimizer: str = "adam",
        loss: str = "mse",
        compile_metrics: tuple = (),
        **kwargs,
    ):
        """
        Constructor for multi-layer perceptron models.

        Args:
            input_dim (int): input dimension, i.e., feature dimension
            describer: a BaseDescriber featurizer.
            n_neurons: number of neurons in each layer of MLP.
            activation (str): activation function
            n_targets (int): number of targets
            is_classification (bool): whether the target is a classification problem
            optimizer (str): optimizer
            loss (str): loss function
            compile_metrics (tuple): metrics to evaluate during epochs
            **kwargs: pass to constructor of KerasModel.
        """
        input_dim = self.get_input_dim(describer, input_dim)
        if input_dim is None:
            raise ValueError("input_dim is not known and cannot be inferred")
        model = construct_mlp(
            input_dim=input_dim,
            n_neurons=n_neurons,
            activation=activation,
            n_targets=n_targets,
            is_classification=is_classification,
            optimizer=optimizer,
            loss=loss,
            compile_metrics=compile_metrics,
        )
        super().__init__(describer=describer, model=model, **kwargs)
