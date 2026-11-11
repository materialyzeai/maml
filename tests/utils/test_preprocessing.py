"""
Tests for preprocessing
"""

from __future__ import annotations

import unittest

import pytest
from pymatgen.core import Lattice, Structure

from maml.utils import DummyScaler, Scaler, StandardScaler


class TestPreprocessing(unittest.TestCase):
    def test_ss(self):
        ss = StandardScaler(mean=0, std=10, is_intensive=False)
        self.assertAlmostEqual(ss.transform(10, n=10), 0.1)

        s = Structure(Lattice.cubic(3.16), ["Mo", "Mo"], [[0, 0, 0], [0.5, 0.5, 0.5]])

        ss2 = StandardScaler.from_training_data(structures=[s, s], targets=[0, 0], is_intensive=True)
        self.assertAlmostEqual(ss2.transform(10, n=10), 10)
        self.assertAlmostEqual(ss2.inverse_transform(10, n=10), 10)

    def test_dummy(self):
        dummy = DummyScaler()
        self.assertAlmostEqual(dummy.transform(100), 100)
        self.assertAlmostEqual(DummyScaler.from_training_data([0, 0], [0, 0]).inverse_transform(100), 100)


@pytest.mark.parametrize("method", ["transform", "inverse_transform"])
def test_base_scaler_not_implemented(method):
    with pytest.raises(NotImplementedError):
        getattr(Scaler(), method)(1.0)


def test_standard_scaler_str_and_zero_std():
    ss = StandardScaler(mean=1.23456, std=0.0, is_intensive=False)
    assert ss.std == 1.0
    assert str(ss) == "StandardScaler(mean=1.235, std=1.000, is_intensive=False)"
    assert repr(ss) == str(ss)


def test_standard_scaler_roundtrip_extensive():
    ss = StandardScaler(mean=0.5, std=2.0, is_intensive=False)
    assert ss.inverse_transform(ss.transform(12.0, n=4), n=4) == pytest.approx(12.0)


if __name__ == "__main__":
    unittest.main()
