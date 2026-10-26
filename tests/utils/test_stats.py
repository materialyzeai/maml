from __future__ import annotations

import unittest

import numpy as np
import pytest
from scipy.stats import kurtosis, skew

from maml.utils import Stats, get_full_stats_and_funcs, stats_list_conversion


class TestStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = [1, 1, 2, 3, 4, 5, 2]
        cls.weights = [0.1, 0.2, 0.3, 0.1, 0.2, 0.05, 0.05]

    def test_mean(self):
        self.assertAlmostEqual(Stats.mean(self.x), np.mean(self.x))
        self.assertAlmostEqual(Stats.mean(self.x, self.weights), np.average(self.x, weights=self.weights))
        self.assertAlmostEqual(Stats.inverse_mean(self.x), np.mean(1.0 / np.array(self.x)))

        self.assertAlmostEqual(Stats.mean(self.x, self.weights), Stats.average(self.x, self.weights))

    def test_max_min_range(self):
        assert Stats.max(self.x) == 5
        assert Stats.min(self.x) == 1
        assert Stats.range(self.x) == 4

    def test_moment(self):
        np.testing.assert_almost_equal(Stats.moment(self.x, max_order=2), [np.mean(self.x), np.std(self.x)])
        self.assertAlmostEqual(Stats.moment(self.x, order=1), np.mean(self.x))

        self.assertAlmostEqual(Stats.moment(self.x), np.mean(self.x))

    def test_kurtosis_skew(self):
        self.assertAlmostEqual(Stats.skewness(self.x), skew(self.x))
        self.assertAlmostEqual(Stats.kurtosis(self.x), kurtosis(self.x, fisher=False))

    def test_geometric_mean(self):
        self.assertAlmostEqual(
            Stats.geometric_mean(self.x, self.weights),
            np.prod(np.power(self.x, self.weights)) ** (1.0 / np.sum(self.weights)),
        )

    def test_shifted_geometric_mean(self):
        self.assertAlmostEqual(Stats.shifted_geometric_mean([0.12, -0.08, 0.02], shift=1), 0.0167215)

    def test_harmonic_mean(self):
        self.assertAlmostEqual(
            Stats.harmonic_mean(self.x, self.weights),
            np.sum(self.weights) / np.sum(np.array(self.weights) / np.array(self.x)),
        )

    def test_allowed_stats(self):
        assert "best" not in Stats.allowed_stats
        assert "average" in Stats.allowed_stats

    def test_power_mean(self):
        p1 = Stats.power_mean(self.x, self.weights, p=0)
        self.assertAlmostEqual(p1, Stats.geometric_mean(self.x, self.weights))
        p2 = Stats.power_mean(self.x, self.weights, p=2)

        holder_mean_2 = np.sum(np.array(self.weights) * np.array(self.x) ** 2) ** (1.0 / 2)
        self.assertAlmostEqual(p2, holder_mean_2)

    def test_mode(self):
        self.assertAlmostEqual(Stats.mode(self.x), 1.5)
        self.assertAlmostEqual(Stats.mode(self.x, self.weights), 2)

    def test_mae(self):
        self.assertAlmostEqual(
            Stats.mean_absolute_deviation(self.x), np.mean(np.abs(np.array(self.x) - np.mean(self.x)))
        )


def test_mean_absolute_error_alias():
    x = [1.0, 2.0, 6.0]
    assert Stats.mean_absolute_error(x) == pytest.approx(Stats.mean_absolute_deviation(x))
    assert Stats.mean_absolute_error(x) == pytest.approx(2.0)


def test_moment_max_order_overrides_order():
    x = [1.0, 2.0, 6.0]
    assert Stats.moment(x, order=3, max_order=2) == pytest.approx([np.mean(x), np.std(x)])


@pytest.mark.parametrize("func", [Stats.skewness, Stats.kurtosis])
def test_skew_kurtosis_constant_data(func):
    # std ~ 0 is replaced by 1 to avoid division by zero; all central moments are 0
    assert func([2.0, 2.0, 2.0]) == pytest.approx(0.0)


@pytest.mark.parametrize("data", [[1.0, -1.0], [0.0, 2.0]])
def test_power_mean_non_positive_raises(data):
    with pytest.raises(ValueError, match="Not possible to calculate geometric means"):
        Stats.power_mean(data)


def test_power_mean_unweighted():
    assert Stats.power_mean([1.0, 3.0], p=1) == pytest.approx(2.0)
    assert Stats.power_mean([1.0, 4.0], p=0) == pytest.approx(2.0)


def test_weighted_std_uses_weighted_mean():
    """Regression: higher-order moments were centred on the unweighted mean."""
    # all weight on a single point -> zero spread
    assert Stats.std([0.0, 10.0], weights=[1.0, 0.0]) == pytest.approx(0.0)


_WEIGHTED_CASES = [
    ([1.0, 2.0, 4.0], [1, 2, 1]),
    ([0.5, 3.0, -1.0, 7.0], [3, 1, 2, 1]),
    ([1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0], [2, 4, 6, 2, 4, 1, 1]),
]


def _expand(data, weights):
    return np.repeat(np.array(data), weights)


@pytest.mark.parametrize(("data", "weights"), _WEIGHTED_CASES)
@pytest.mark.parametrize("scale", [1.0, 0.1])  # integer counts and the same pmf as fractions
def test_weighted_stats_match_expanded_reference(data, weights, scale):
    w = [i * scale for i in weights]
    expanded = _expand(data, weights)
    assert Stats.mean(data, w) == pytest.approx(np.mean(expanded))
    assert Stats.std(data, w) == pytest.approx(np.std(expanded))
    assert Stats.skewness(data, w) == pytest.approx(skew(expanded))
    assert Stats.kurtosis(data, w) == pytest.approx(kurtosis(expanded, fisher=False))
    assert Stats.mean_absolute_deviation(data, w) == pytest.approx(np.mean(np.abs(expanded - expanded.mean())))
    mean_w = np.average(data, weights=w)
    for order in (2, 3, 4):
        central = np.average((np.array(data) - mean_w) ** order, weights=w)
        expected = np.sign(central) * np.abs(central) ** (1.0 / order)
        assert Stats.moment(data, w, order=order) == pytest.approx(expected)
    np.testing.assert_allclose(
        Stats.moment(data, w, max_order=3),
        [Stats.moment(data, w, order=i) for i in (1, 2, 3)],
    )


@pytest.mark.parametrize(("data", "weights"), _WEIGHTED_CASES)
def test_uniform_weights_equal_unweighted(data, weights):
    uniform = [1.0] * len(data)
    for func in (Stats.std, Stats.skewness, Stats.kurtosis):
        assert func(data, uniform) == pytest.approx(func(data))
    assert Stats.std(data) == pytest.approx(np.std(data))
    assert Stats.skewness(data) == pytest.approx(skew(data))
    assert Stats.kurtosis(data) == pytest.approx(kurtosis(data, fisher=False))


@pytest.mark.parametrize(
    ("stats", "expected"),
    [
        (["mean", "std"], ["mean", "std"]),
        (["moment:None:3"], ["moment:1:None", "moment:2:None", "moment:3:None"]),
        (["moment:2:None"], ["moment:2:None"]),
        (["moment:None:0"], ["moment:0:None"]),
        (["shifted_geometric_mean:50"], ["shifted_geometric_mean:50"]),
    ],
)
def test_stats_list_conversion(stats, expected):
    assert stats_list_conversion(stats) == expected


def test_get_full_stats_and_funcs():
    x = [1.0, 2.0, 6.0]
    names, funcs = get_full_stats_and_funcs(["mean", "moment:None:2", "shifted_geometric_mean:10"])
    assert names == ["mean", "moment:1:None", "moment:2:None", "shifted_geometric_mean:10"]
    values = [f(x) for f in funcs]
    assert values[0] == pytest.approx(3.0)
    assert values[1] == pytest.approx(3.0)
    assert values[2] == pytest.approx(np.std(x))
    assert values[3] == pytest.approx(Stats.shifted_geometric_mean(x, shift=10.0))


@pytest.mark.parametrize("stats", [["best"], ["best:1"]])
def test_get_full_stats_and_funcs_unknown(stats):
    with pytest.raises(ValueError, match="best not in available Stats"):
        get_full_stats_and_funcs(stats)


if __name__ == "__main__":
    unittest.main()
