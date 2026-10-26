from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from maml.base import SKLModel, TargetScalerMixin
from maml.utils import DummyScaler, StandardScaler


class _ScaledSKLModel(TargetScalerMixin, SKLModel):
    pass


@pytest.fixture
def data():
    # each "object" is a feature vector; len(obj) plays the role of the number of atoms (2 here)
    x = [[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]
    y = [3.0, 7.0, 12.0]
    return x, y


def test_default_scaler_is_dummy():
    model = _ScaledSKLModel(model=LinearRegression())
    assert isinstance(model.target_scaler, DummyScaler)


def test_train_scales_targets_and_predict_inverts(data):
    x, y = data
    scaler = StandardScaler(mean=1.0, std=2.0, is_intensive=False)
    model = _ScaledSKLModel(model=LinearRegression(), target_scaler=scaler)
    assert model.train(x, y) is model

    # the underlying estimator is fitted on (y / n - mean) / std with n = 2
    scaled = (np.array(y) / 2 - 1.0) / 2.0
    np.testing.assert_allclose(model.model.predict(np.array(x)), scaled, atol=1e-8)
    # predict_objs applies the inverse transform and recovers the original targets
    np.testing.assert_allclose(model.predict_objs(x), y, atol=1e-8)


def test_train_with_validation_scales_val_targets(data, monkeypatch):
    x, y = data
    scaler = StandardScaler(mean=0.0, std=10.0, is_intensive=True)
    model = _ScaledSKLModel(model=LinearRegression(), target_scaler=scaler)

    captured = {}

    def fake_fit(self, features, targets=None, val_features=None, val_targets=None, **kwargs):
        captured.update(targets=targets, val_targets=val_targets)
        return self

    monkeypatch.setattr(SKLModel, "fit", fake_fit)
    model.train(x, y, val_objs=x[:1], val_targets=y[:1])
    np.testing.assert_allclose(captured["targets"], np.array(y) / 10)
    np.testing.assert_allclose(captured["val_targets"], [y[0] / 10])


def test_train_without_targets_delegates(monkeypatch):
    model = _ScaledSKLModel(model=LinearRegression())
    sentinel = object()
    monkeypatch.setattr(SKLModel, "train", lambda self, *args, **kwargs: sentinel)
    assert model.train([[1.0]]) is sentinel
