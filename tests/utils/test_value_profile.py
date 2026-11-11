from __future__ import annotations

import unittest

import numpy as np
import pytest

from maml.utils import ConstantValue, LinearProfile, ValueProfile


class TestValueProfile(unittest.TestCase):
    def test_constant(self):
        constant_profile = ConstantValue(0.1)
        constant_profile.increment_step()
        assert constant_profile.step == 1
        self.assertAlmostEqual(constant_profile.get_value(), 0.1)

    def test_linearprofile(self):
        linear = LinearProfile(value_start=100, value_end=1, max_steps=99)

        self.assertAlmostEqual(linear.rate, -1)
        for _ in range(99):
            linear.increment_step()
        self.assertAlmostEqual(linear.get_value(), 1)


def test_max_steps_exceeded():
    profile = ConstantValue(1.0, max_steps=2)
    profile.increment_step()
    profile.increment_step()
    with pytest.raises(RuntimeError, match="Step exceeding maximum"):
        profile.increment_step()


def test_base_get_value_not_implemented():
    with pytest.raises(NotImplementedError):
        ValueProfile().get_value()


def test_linear_profile_with_rate_is_unbounded():
    linear = LinearProfile(value_start=1.0, rate=0.5)
    assert linear.max_steps == np.inf
    for _ in range(1000):
        linear.increment_step()
    assert linear.get_value() == pytest.approx(501.0)


if __name__ == "__main__":
    unittest.main()
