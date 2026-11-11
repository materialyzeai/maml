from __future__ import annotations

import unittest

import numpy as np
import pytest

from maml.apps.bowsr.preprocessing import DummyScaler, StandardScaler


class ScalerTest(unittest.TestCase):
    def setUp(self):
        self.target = np.random.random((12, 6))
        self.scaler1 = StandardScaler()
        self.scaler2 = DummyScaler()

    def test_attributes(self):
        assert self.scaler1.mean is None
        assert self.scaler1.std is None

    def test_fit(self):
        self.scaler1.fit(self.target)
        self.scaler2.fit(self.target)

        assert np.all(abs(self.scaler1.mean - np.mean(self.target, axis=0)) < 0.0001)
        assert np.all(abs(self.scaler1.std - np.std(self.target, axis=0)) < 0.0001)

    def test_transform(self):
        self.assertRaises(ValueError, self.scaler1.transform, self.target)
        self.scaler1.fit(self.target)
        self.scaler2.fit(self.target)

        transformed_target1 = self.scaler1.transform(self.target)
        assert transformed_target1.shape == self.target.shape
        transformed_target2 = (self.target - np.mean(self.target, axis=0)) / np.std(self.target, axis=0)
        assert np.all(abs(transformed_target1 - transformed_target2) < 1e-05)

        transformed_target3 = self.scaler2.transform(self.target)
        assert np.all(abs(transformed_target3 - self.target) < 1e-05)

    def test_inverse_transform(self):
        self.assertRaises(ValueError, self.scaler1.inverse_transform, self.target)
        self.scaler1.fit(self.target)
        self.scaler2.fit(self.target)

        transformed_target1 = self.scaler1.transform(self.target)
        target1 = self.scaler1.inverse_transform(transformed_target1)
        assert np.all(abs(target1 - self.target) < 0.0001)

        target2 = self.scaler2.inverse_transform(self.target)
        assert np.all(abs(target2 - self.target) < 0.0001)

    def test_as_dict(self):
        self.scaler1.fit(self.target)
        scaler1_dict = self.scaler1.as_dict()
        assert scaler1_dict["@class"] == "StandardScaler"
        assert np.all(abs(scaler1_dict["params"]["mean"] - np.mean(self.target, axis=0)) < 0.0001)
        assert np.all(abs(scaler1_dict["params"]["std"] - np.std(self.target, axis=0)) < 0.0001)

        scaler2_dict = self.scaler2.as_dict()
        assert scaler2_dict["@class"] == "DummyScaler"

    def test_from_dict(self):
        self.scaler1.fit(self.target)
        scaler1_dict = self.scaler1.as_dict()
        scaler1 = StandardScaler.from_dict(scaler1_dict)
        assert np.all(abs(scaler1.mean - self.scaler1.mean) < 0.0001)
        assert np.all(abs(scaler1.std - self.scaler1.std) < 0.0001)

        scaler2_dict = self.scaler2.as_dict()
        scaler2 = DummyScaler.from_dict(scaler2_dict)
        transformed_target2 = scaler2.transform(self.target)
        target2 = scaler2.inverse_transform(self.target)
        assert np.all(abs(transformed_target2 - self.target) < 0.0001)
        assert np.all(abs(target2 - self.target) < 0.0001)


def test_scaler_repr():
    assert repr(DummyScaler()) == "DummyScaler"
    assert repr(StandardScaler(mean=1.0, std=2.0)) == "StandardScaler(mean=1.0, std=2.0)"


@pytest.mark.parametrize(
    ("mean", "std"),
    [(None, None), ([0.0, 1.0], [1.0, 2.0]), (np.array([0.0, 1.0]), np.array([1.0, 2.0]))],
)
def test_standard_scaler_as_dict_roundtrip(mean, std):
    """Regression: as_dict raised AttributeError for an unfitted scaler or list parameters."""
    scaler = StandardScaler(mean=mean, std=std)
    d = scaler.as_dict()
    assert d["params"] == {
        "mean": None if mean is None else list(mean),
        "std": None if std is None else list(std),
    }
    restored = StandardScaler.from_dict(d)
    if mean is None:
        assert restored.mean is None
        assert restored.std is None
        with pytest.raises(ValueError, match="No parameters"):
            restored.transform(np.zeros((1, 2)))
    else:
        x = np.array([[1.0, 3.0]])
        assert np.allclose(restored.transform(x), scaler.transform(x))


if __name__ == "__main__":
    unittest.main()
