# flake8: noqa

import json
import os
import unittest

import numpy as np
import pytest
from pymatgen.util.testing import PymatgenTest
from scipy.optimize import check_grad

try:
    import cvxpy as cp

    from maml.apps.symbolic._selectors_cvxpy import (
        AdaptiveLassoCP,
        DantzigSelectorCP,
        LassoCP,
        cp,
    )

except ImportError:
    cp = None

from maml.apps.symbolic._selectors import (
    SCAD,
    AdaptiveLasso,
    BaseSelector,
    DantzigSelector,
    L0BrutalForce,
    Lasso,
    PenalizedLeastSquares,
)

CWD = os.path.abspath(os.path.dirname(__file__))


class TestSelectors(PymatgenTest):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        with open(os.path.join(CWD, "test_data.json")) as f:
            djson = json.load(f)
        cls.x = np.array(djson["x"])
        cls.beta = np.array(djson["beta"])
        cls.y = cls.x.dot(cls.beta)
        cls.lasso_alpha = 0.1

    def test_dantzig(self):
        dt = DantzigSelector(1)
        selected = dt.select(self.x, self.y)
        # selected only [4, 5, 6]
        np.testing.assert_allclose(selected, [4, 5, 6])
        self.assertAlmostEqual(-dt.evaluate(self.x, self.y), 0.40040170)
        np.testing.assert_almost_equal(dt.get_coef(), dt.coef_)

        np.testing.assert_almost_equal(dt.get_feature_indices(), selected)
        self.assertTrue(dt.compute_residual(self.x, self.y).shape == self.y.shape)

        assert dt._get_param_names() == ["lambd", "sigma"]
        assert dt.get_params() == {"lambd": 1, "sigma": 1.0}
        dt.set_params(**{"lambd": 0.1})
        assert dt.get_params() == {"lambd": 0.1, "sigma": 1.0}

    @unittest.skipIf(cp is None, "cvxpy not installed")
    def test_dantzigcp(self):
        dt = DantzigSelectorCP(1)
        selected = dt.select(self.x, self.y)
        np.testing.assert_allclose(selected, [4, 5, 6])

    def test_lasso(self):
        lasso = Lasso(self.lasso_alpha)
        selected = lasso.select(self.x, self.y, options={"maxiter": 1e5, "ftol": 1e-15})
        np.testing.assert_allclose(selected, [4, 5, 6, 9])

    @unittest.skipIf(cp is None, "cvxpy not installed")
    def test_lassocp(self):
        lasso = LassoCP(self.lasso_alpha)
        selected = lasso.select(self.x, self.y)
        np.testing.assert_allclose(selected, [4, 5, 6, 9])

        from sklearn.linear_model import Lasso

        lasso = Lasso(self.lasso_alpha, fit_intercept=False)
        lasso.fit(self.x, self.y)
        np.testing.assert_allclose(np.where(np.abs(lasso.coef_) > 1e-10)[0], [4, 5, 6, 9])

    def test_adaptive_lasso(self):
        lasso = AdaptiveLasso(self.lasso_alpha, gamma=0.1)
        selected = lasso.select(self.x, self.y, options={"maxiter": 1e4, "ftol": 1e-12})

        np.testing.assert_allclose(selected, [4, 5, 6, 9])

    @unittest.skipIf(cp is None, "cvxpy not installed")
    def test_adaptive_lassocp(self):
        lasso = AdaptiveLassoCP(self.lasso_alpha, gamma=0.1)
        selected = lasso.select(self.x, self.y)
        np.testing.assert_allclose(selected, [4, 5, 6, 9])

    def test_scad(self):
        # Regression: a wrong SCAD gradient used to make SLSQP fail here with status 9.
        scad = SCAD(1.0)
        selected = scad.select(self.x, self.y)
        assert scad.is_fitted
        # all true |beta| < lambd, so the SCAD penalty shrinks everything to zero
        np.testing.assert_array_equal(selected, [])

    def test_l0(self):
        lambd = 1e-6
        l0 = L0BrutalForce(lambd)
        selected = l0.select(self.x, self.y)
        self.assertListEqual(selected.tolist(), [6, 7, 8, 9])


TRUE_SUPPORT = [0, 3]
TRUE_COEF = np.array([3.0, 0, 0, -2.0, 0, 0, 0, 0])


@pytest.fixture(scope="module")
def sparse_problem() -> tuple[np.ndarray, np.ndarray]:
    """y = 3 * x0 - 2 * x3 + small noise, 8 features, 100 samples."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 8))
    y = x @ TRUE_COEF + 0.01 * rng.standard_normal(100)
    return x, y


@pytest.mark.parametrize(
    ("selector", "atol"),
    [
        (DantzigSelector(1.0), 0.02),
        (Lasso(0.1), 0.15),
        (AdaptiveLasso(0.1, gamma=0.5), 0.1),  # shrinkage ~ lambd / sqrt(|beta_ols|)
        (SCAD(0.1), 0.02),  # SCAD is nearly unbiased for large coefficients
        (L0BrutalForce(1e-3), 0.01),
    ],
    ids=["dantzig", "lasso", "adaptive_lasso", "scad", "l0"],
)
def test_support_recovery(selector, atol, sparse_problem):
    x, y = sparse_problem
    np.testing.assert_array_equal(selector.select(x, y), TRUE_SUPPORT)
    coef = selector.get_coef()
    assert np.all(np.delete(coef, TRUE_SUPPORT) == 0.0)
    np.testing.assert_allclose(coef[TRUE_SUPPORT], TRUE_COEF[TRUE_SUPPORT], atol=atol)
    np.testing.assert_allclose(selector.compute_residual(x, y), y - x[:, TRUE_SUPPORT] @ coef[TRUE_SUPPORT])


def test_scad_penalty_and_jac():
    lambd, a = 0.5, 3.7
    scad = SCAD(lambd, a=a)
    # three regimes of eq. 12: linear, quadratic and constant
    assert scad.penalty(np.array([0.2])) == pytest.approx(lambd * 0.2)
    b = 1.0
    assert scad.penalty(np.array([b])) == pytest.approx(-(b**2 - 2 * a * lambd * b + lambd**2) / (2 * (a - 1)))
    assert scad.penalty(np.array([-5.0])) == pytest.approx((a + 1) * lambd**2 / 2)
    # the penalty is continuous at the knots
    for knot in (lambd, a * lambd):
        assert scad.penalty(np.array([knot - 1e-9])) == pytest.approx(scad.penalty(np.array([knot + 1e-9])))

    # analytic gradient matches finite differences away from the kinks
    rng = np.random.default_rng(1)
    for _ in range(5):
        beta = rng.uniform(-3, 3, 6)
        assert check_grad(scad.penalty, lambda bb: scad._penalty_jac(None, None, bb), beta) < 1e-6
    # gradient is odd in beta and vanishes beyond a * lambd
    np.testing.assert_allclose(
        scad._penalty_jac(None, None, np.array([-0.2, 0.2, 2.0, -2.0])), [-lambd, lambd, 0.0, 0.0]
    )


@pytest.mark.parametrize(
    "selector",
    [Lasso(0.1), AdaptiveLasso(0.1, gamma=0.5), SCAD(0.1)],
    ids=["lasso", "adaptive_lasso", "scad"],
)
def test_penalized_least_squares_jac(selector, sparse_problem):
    """The full loss Jacobian agrees with finite differences of the loss."""
    x, y = sparse_problem
    if isinstance(selector, AdaptiveLasso):
        selector.w = selector.get_w(x, y)
    jac = selector.construct_jac(x, y)
    beta = np.random.default_rng(2).uniform(0.5, 1.0, x.shape[1])  # away from kinks
    assert check_grad(lambda bb: selector.construct_loss(x, y, bb), jac, beta) < 1e-5


def test_base_selector_defaults(sparse_problem):
    x, y = sparse_problem
    base = BaseSelector()
    assert base.construct_constraints(x, y) is None
    assert base.construct_jac(x, y) is None
    with pytest.raises(NotImplementedError):
        base.construct_loss(x, y, np.zeros(x.shape[1]))
    assert base.get_params() == {"coef_thres": 1e-6, "method": "SLSQP"}

    pls = PenalizedLeastSquares()
    assert pls.penalty(np.ones(3)) == 0.0
    assert pls._penalty_jac(x, y, np.ones(3)) == 0.0
    assert pls.construct_constraints(x, y) == []
    pls.select(x, y)
    np.testing.assert_allclose(pls.coef_, np.linalg.lstsq(x, y, rcond=None)[0], atol=1e-4)


def test_select_raises_when_not_converged(sparse_problem):
    lasso = Lasso(0.1)
    with pytest.raises(RuntimeError, match="Not converged"):
        lasso.select(*sparse_problem, options={"maxiter": 1})
    assert not lasso.is_fitted


class _Nested(BaseSelector):
    def __init__(self, inner: BaseSelector, **kwargs):
        self.inner = inner
        super().__init__(**kwargs)


def test_set_params():
    lasso = Lasso(0.1)
    assert lasso.set_params() is lasso
    with pytest.raises(ValueError, match="Invalid parameter foo"):
        lasso.set_params(foo=1)
    nested = _Nested(Lasso(0.1))
    assert nested.set_params(inner__lambd=0.7) is nested
    assert nested.inner.lambd == 0.7


if __name__ == "__main__":
    unittest.main()
