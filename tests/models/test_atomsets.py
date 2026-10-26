"""
Test models
"""

from __future__ import annotations

import json
from unittest import TestCase, main

import numpy as np
import pytest

from maml.describers import SiteElementProperty
from maml.models import AtomSets
from maml.models.dl._atomsets import construct_atom_sets


class TestAtomSets(TestCase):
    def test_predict(self):
        x = np.array([[0, 1, 0, 1, 0, 1]], dtype=np.int32).reshape((1, -1))
        x_vec = np.random.normal(size=(1, 6, 20))
        indices = np.array([[0, 0, 0, 1, 1, 1]], dtype=np.int32).reshape((1, -1))
        y = np.array([[0.1, 0.2]]).reshape((1, 2, 1))

        model1 = AtomSets(
            describer=SiteElementProperty(),
            is_embedding=True,
            symmetry_func="mean",
            n_neurons=(8, 8),
            n_neurons_final=(4, 4),
            n_targets=1,
        )
        model2 = AtomSets(
            input_dim=20,
            is_embedding=False,
            symmetry_func="set2set",
            n_neurons=(4, 4),
            n_neurons_final=(4, 4),
            T=2,
            n_hidden=10,
        )
        res = model1.predict_objs(["H2O", "FeO"])
        print(res.shape, " res.shape")
        assert res.shape == (2, 1)
        res3 = model2.model.predict([x_vec, np.ones_like(indices), indices])
        assert res3.shape == (1, 2, 1)


# all two-site compositions, so that describer features stack into a regular array
OBJS = ["FeO", "NaCl", "LiF", "MgO", "CaS"]
TARGETS = [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.fixture
def model():
    return AtomSets(describer=SiteElementProperty(), n_neurons=(4,), n_neurons_final=(4,), embedding_dim=4)


def test_construct_embedding_with_input_dim_raises():
    with pytest.raises(ValueError, match="input dim needs to be None"):
        construct_atom_sets(input_dim=3, is_embedding=True)


def test_construct_unsupported_symmetry_func():
    with pytest.raises(ValueError, match="symmetry function not supported"):
        construct_atom_sets(symmetry_func="max")


def test_construct_multiple_symmetry_funcs_classification():
    m = construct_atom_sets(
        input_dim=3,
        is_embedding=False,
        n_neurons=(4,),
        n_neurons_final=(),
        n_targets=2,
        symmetry_func=["mean", "set2set"],
        is_classification=True,
        T=1,
        n_hidden=3,
    )
    x = np.random.default_rng(0).normal(size=(1, 5, 3)).astype("float32")
    weights = np.ones((1, 5), dtype="float32")
    indices = np.array([[0, 0, 1, 1, 1]], dtype="int32")
    out = m.predict([x, weights, indices], verbose=0)
    # one row per set, two sigmoid outputs each
    assert out.shape == (1, 2, 2)
    assert np.all((out > 0) & (out < 1))


def test_predict_multiple_batches(model):
    """Regression: _predict crashed with "'tuple' object is not an iterator" for more than one batch."""
    features = model.describer.transform(OBJS)
    one_batch = model._predict(features)
    multi_batch = model._predict(features, batch_size=2)
    assert multi_batch.shape == (len(OBJS), 1)
    np.testing.assert_allclose(multi_batch, one_batch, rtol=1e-5)


def test_fit_with_validation_and_evaluate(model):
    features = model.describer.transform(OBJS)
    # 5 objects in batches of 3 -> 6 and 4 sites, so Keras sees a variable site axis
    history = model.fit(
        features, TARGETS, val_features=features, val_targets=TARGETS, epochs=2, batch_size=3, verbose=0
    )
    assert len(history.history["val_loss"]) == 2

    preds = model.predict_objs(OBJS)
    expected_mse = np.mean((preds[:, 0] - np.array(TARGETS)) ** 2)
    assert model.evaluate(OBJS, TARGETS, batch_size=len(OBJS)) == pytest.approx(expected_mse, rel=1e-4)
    # evaluate with precomputed features
    assert model.evaluate(features, TARGETS, is_feature=True, batch_size=len(OBJS)) == pytest.approx(
        expected_mse, rel=1e-4
    )


def test_data_generator_with_weights_and_shuffle(model):
    features = [[np.array([[1], [2]]), np.array([0.5, 0.5])], [np.array([[3]]), np.array([1.0])]]
    gen = model._get_data_generator(features, [1.0, 2.0], batch_size=1, is_shuffle=True)
    assert len(gen) == 2
    (x, weights, index), y = gen[0]
    np.testing.assert_array_equal(x, [[1, 2]])
    np.testing.assert_array_equal(weights, [[0.5, 0.5]])
    np.testing.assert_array_equal(index, [[0, 0]])
    np.testing.assert_array_equal(y, [[1.0]])

    np.random.seed(1)
    gen.on_epoch_end()
    # shuffling keeps features, weights and targets aligned
    pairs = {float(t): float(np.sum(w)) for t, w in zip(gen.targets, gen.weights, strict=True)}
    assert pairs == {1.0: 1.0, 2.0: 1.0}
    assert model._get_data_generator(None, None) is None


def test_save_and_from_dir(model, tmp_path):
    model.fit(model.describer.transform(OBJS), TARGETS, epochs=1, verbose=0)
    dirname = tmp_path / "atomsets"
    model.save(str(dirname))
    with open(dirname / "config.json") as f:
        config = json.load(f)
    assert config["n_neurons"] == [4]
    assert config["activation"] == "relu"

    loaded = AtomSets.from_dir(str(dirname))
    np.testing.assert_allclose(loaded.predict_objs(OBJS), model.predict_objs(OBJS), rtol=1e-5)
    # saving into an existing directory overwrites
    model.save(str(dirname))


@pytest.mark.xfail(
    reason="BaseModel.train calls to_array on describer output; per-structure site features of different "
    "lengths are ragged and np.array raises. AtomSets needs its own train() that keeps features as a list.",
    raises=ValueError,
    strict=True,
)
def test_train_with_different_number_of_sites(model):
    model.train(["H2O", "FeO"], [0.1, 0.2], epochs=1, verbose=0)


@pytest.mark.xfail(
    reason="_DataGenerator puts each batch on axis 1 behind a dummy leading axis of size 1. Keras 3 infers the "
    "dataset signature from the first two batches only, so if they have the same number of sites, a later "
    "batch with a different count fails with an InvalidArgumentError.",
    strict=True,
)
def test_fit_partial_last_batch(model):
    # 5 two-site objects in batches of 2 -> 4, 4, 2 sites
    model.fit(model.describer.transform(OBJS), TARGETS, epochs=1, batch_size=2, verbose=0)


if __name__ == "__main__":
    main()
