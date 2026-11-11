"""
Test models
"""

from __future__ import annotations

from unittest import TestCase, main

import numpy as np
import pytest

from maml.describers import ElementStats
from maml.models import MLP


class TestAtomSets(TestCase):
    x = ["H2O", "Fe2O3"]
    y = [0.1, 0.2]
    model = MLP(describer=ElementStats.from_data("megnet_3"), n_neurons=(2, 2))

    def test_train(self):
        self.model.train(self.x, self.y, epochs=0)
        assert self.model.predict_objs(["H2O"]).shape == (1, 1)


def test_mlp_requires_input_dim():
    with pytest.raises(ValueError, match="input_dim is not known"):
        MLP()


def test_mlp_classification_output_in_unit_interval():
    model = MLP(input_dim=3, n_neurons=(4,), n_targets=2, is_classification=True, loss="binary_crossentropy")
    assert model.model.layers[-1].activation.__name__ == "sigmoid"
    out = model.predict_objs(np.random.default_rng(0).normal(size=(5, 3)) * 100)
    assert out.shape == (5, 2)
    assert np.all((out >= 0) & (out <= 1))


@pytest.mark.xfail(
    reason="KerasMixin.save/load use the legacy .hdf5 format; with Keras 3, load_model cannot deserialize a model "
    "compiled with a string loss such as MLP's default 'mse' (\"Could not deserialize 'keras.metrics.mse'\").",
    raises=ValueError,
    strict=True,
)
def test_mlp_save_and_load(tmp_path):
    model = MLP(input_dim=3, n_neurons=(2,))
    x = np.random.default_rng(0).normal(size=(4, 3))
    fname = str(tmp_path / "mlp")
    model.save(fname)
    other = MLP(input_dim=3, n_neurons=(2,))
    other.load(fname)
    np.testing.assert_allclose(other.predict_objs(x), model.predict_objs(x), rtol=1e-6)


@pytest.mark.xfail(
    reason="KerasMixin.from_file calls cls(model=None, ...), but MLP.__init__ builds its own model and forwards "
    "kwargs to KerasModel, so `model` is passed twice (TypeError).",
    raises=TypeError,
    strict=True,
)
def test_mlp_from_file(tmp_path):
    model = MLP(input_dim=3, n_neurons=(2,))
    fname = str(tmp_path / "mlp")
    model.save(fname)
    MLP.from_file(fname, input_dim=3)


if __name__ == "__main__":
    main()
