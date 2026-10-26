from __future__ import annotations

import os
import unittest

import numpy as np
import pandas as pd
import pytest
from pymatgen.util.testing import PymatgenTest

from maml.apps.symbolic import ISIS, SIS, L0BrutalForce
from maml.apps.symbolic._selectors import BaseSelector, DantzigSelector
from maml.apps.symbolic._sis import _best_combination, _get_coeff

CWD = os.path.abspath(os.path.dirname(__file__))


class TestSIS(PymatgenTest):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        df = pd.read_csv(os.path.join(CWD, "sis_test.csv"), index_col=[0])
        cls.x = df[["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x", "pow2", "exp"]]
        cls.y = df["y"]

    def test_sis(self):
        sis = SIS(gamma=0.5, selector=L0BrutalForce(1e-4))
        selected = sis.run(self.x.values, self.y)
        np.testing.assert_almost_equal(selected, [12, 10, 9])
        sis.set_gamma(0.1)
        assert sis.gamma == 0.1
        sis.update_gamma(ratio=0.5)
        assert sis.gamma == pytest.approx(0.15)


class testISIS(PymatgenTest):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        df = pd.read_csv(os.path.join(CWD, "sis_test.csv"), index_col=[0])
        cls.x = df[["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x", "pow2", "exp"]]
        cls.y = df["y"]

    def test_coeff(self):
        ISIS(SIS(gamma=0.5, selector=L0BrutalForce(1e-4)))
        coeff = _get_coeff(self.x.values, self.y)
        assert np.allclose(coeff[10:], [1, 1, 1])

    def test_evaluate(self):
        isis = ISIS(SIS(gamma=0.5, selector=L0BrutalForce(1e-4)), l0_regulate=False)
        _ = isis.run(self.x.values, self.y, max_p=4)
        mae = isis.evaluate(self.x.values, self.y)
        assert mae == pytest.approx(0)

    def test_best_combination(self):
        isis = ISIS(SIS(gamma=0.5, selector=L0BrutalForce(1e-4)), l0_regulate=False)
        _ = isis.run(self.x.values, self.y, max_p=4)
        comb_best, coeff_best, score_best = _best_combination(
            self.x.values, self.y, np.array([10, 11]), np.array([12, 0])
        )
        assert np.allclose(comb_best, [10, 11, 12])
        assert np.allclose(coeff_best, [1, 1, 1])
        assert score_best == pytest.approx(0)

    def test_isis(self):
        # Test update gamma
        np.random.seed(42)
        isis = ISIS(SIS(gamma=0.1, selector=L0BrutalForce(1e-4)))
        selected = isis.run(self.x.values, self.y, max_p=3)
        assert isis.sis.gamma == pytest.approx(0.1)
        np.testing.assert_equal(selected, [10, 11, 12])


@pytest.fixture(scope="module")
def sparse_problem() -> tuple[np.ndarray, np.ndarray]:
    """Y = 3 * x0 - 2 * x3 + 1.5 * x7 + small noise, 10 features, 100 samples."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 10))
    y = 3 * x[:, 0] - 2 * x[:, 3] + 1.5 * x[:, 7] + 0.01 * rng.standard_normal(100)
    return x, y


def test_screen_uses_correlation_magnitude(sparse_problem):
    """Regression: screening ranked on signed x^T y, dropping strongly anti-correlated features."""
    x, y = sparse_problem
    omega = x.T @ y
    assert omega[3] < 0
    sis = SIS(gamma=0.3, verbose=False)
    screened = sis.screen(x, y)
    np.testing.assert_array_equal(screened, np.argsort(np.abs(omega))[::-1][:3])
    assert set(screened) == {0, 3, 7}


def test_sis_run_recovers_support(sparse_problem):
    x, y = sparse_problem
    sis = SIS(gamma=0.5, selector=L0BrutalForce(1e-3))
    assert set(sis.run(x, y)) == {0, 3, 7}
    new_selector = L0BrutalForce(1e-4)
    sis.set_selector(new_selector)
    assert sis.selector is new_selector


@pytest.mark.xfail(
    reason="SIS.compute_residual applies selector indices (relative to the screened columns) to the "
    "unscreened x, so the residual after SIS.run uses the wrong features",
    strict=True,
)
def test_sis_compute_residual_after_run(sparse_problem):
    x, y = sparse_problem
    sis = SIS(gamma=0.3, selector=L0BrutalForce(1e-3), verbose=False)
    selected = sis.run(x, y)
    coef = np.linalg.lstsq(x[:, selected], y, rcond=None)[0]
    np.testing.assert_allclose(sis.compute_residual(x, y), y - x[:, selected] @ coef, atol=1e-8)


def test_isis_default_sis():
    """Regression: ISIS() with no SIS raised AttributeError on None.selector."""
    isis = ISIS()
    assert isinstance(isis.sis, SIS)
    assert isis.selector is isis.sis.selector
    assert isinstance(isis.selector, DantzigSelector)


def test_isis_evaluate_after_early_return(sparse_problem):
    """Regression: when the first selection already has max_p features, find_sel was never stored."""
    x, y = sparse_problem
    isis = ISIS(SIS(gamma=0.5, selector=L0BrutalForce(1e-3), verbose=False))
    selected = isis.run(x, y, max_p=2)
    assert len(selected) == 2
    np.testing.assert_array_equal(isis.find_sel, selected)
    assert len(isis.coeff) == 2
    assert isis.evaluate(x, y) < 0  # neg MAE of a 2-feature fit to a 3-feature target


def test_isis_updates_gamma_when_screening_is_empty(sparse_problem):
    """With gamma * n < 1 the screen is empty, the selector raises ValueError and ISIS grows gamma."""
    x, y = sparse_problem
    gamma = 0.05
    isis = ISIS(SIS(gamma=gamma, selector=L0BrutalForce(1e-3), verbose=False))
    selected = isis.run(x, y, max_p=4, step=0.5)
    assert len(selected) == 4
    assert {0, 3, 7} <= set(selected)
    assert isis.sis.gamma > gamma
    # gamma only grows until a single feature survives screening of the 7 remaining columns
    assert int(7 * isis.sis.gamma) == 1
    assert isis.evaluate(x, y) == pytest.approx(0, abs=0.02)


def test_isis_raises_when_nothing_is_ever_selected(sparse_problem):
    """A selector that zeroes every coefficient must not make ISIS loop forever."""

    class _SelectNothing(BaseSelector):
        def select(self, x, y, options=None):
            self.coef_ = np.zeros(x.shape[1])
            self.indices = np.array([], dtype=int)
            return self.indices

    x, y = sparse_problem
    isis = ISIS(SIS(gamma=0.1, selector=_SelectNothing(), verbose=False))
    with pytest.raises(RuntimeError, match="no features"):
        isis.run(x, y, max_p=2)
    assert isis.sis.gamma >= 1


def test_best_combination_single_new_feature(sparse_problem):
    x, y = sparse_problem
    comb, coeff, score = _best_combination(x, y, np.array([0, 3]), np.array([7]))
    np.testing.assert_array_equal(comb, [0, 3, 7])
    np.testing.assert_allclose(coeff, [3, -2, 1.5], atol=0.01)
    assert score == pytest.approx(0, abs=0.02)


if __name__ == "__main__":
    unittest.main()
