"""Tests for the cvxpy-backed selectors, checked against known sparse problems and the scipy selectors."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")

from sklearn.linear_model import Lasso as SklearnLasso  # noqa: E402

from maml.apps.symbolic import SIS  # noqa: E402
from maml.apps.symbolic._selectors import AdaptiveLasso, DantzigSelector, Lasso  # noqa: E402
from maml.apps.symbolic._selectors_cvxpy import (  # noqa: E402
    AdaptiveLassoCP,
    BaseSelectorCP,
    DantzigSelectorCP,
    LassoCP,
    PenalizedLeastSquaresCP,
)

TRUE_SUPPORT = [0, 3]
TRUE_COEF = np.array([3.0, 0, 0, -2.0, 0, 0, 0, 0])


@pytest.fixture(scope="module")
def sparse_problem() -> tuple[np.ndarray, np.ndarray]:
    """Y = 3 * x0 - 2 * x3 + small noise, 8 features, 100 samples."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 8))
    y = x @ TRUE_COEF + 0.01 * rng.standard_normal(100)
    return x, y


@pytest.mark.parametrize(
    ("selector", "atol"),
    [
        (DantzigSelectorCP(1.0), 0.02),
        (LassoCP(0.1), 0.15),  # lasso shrinks the coefficients by ~lambd
        (AdaptiveLassoCP(0.1, gamma=1.0), 0.05),
    ],
    ids=["dantzig", "lasso", "adaptive_lasso"],
)
def test_support_recovery(selector, atol, sparse_problem):
    x, y = sparse_problem
    selected = selector.select(x, y)
    np.testing.assert_array_equal(selected, TRUE_SUPPORT)
    assert selector.is_fitted
    np.testing.assert_array_equal(selector.get_feature_indices(), TRUE_SUPPORT)
    coef = selector.get_coef()
    assert coef.shape == (8,)
    # coefficients outside the support are hard-zeroed by coef_thres
    assert np.all(np.delete(coef, TRUE_SUPPORT) == 0.0)
    np.testing.assert_allclose(coef[TRUE_SUPPORT], TRUE_COEF[TRUE_SUPPORT], atol=atol)
    np.testing.assert_allclose(selector.predict(x), x[:, TRUE_SUPPORT] @ coef[TRUE_SUPPORT])
    np.testing.assert_allclose(selector.compute_residual(x, y), y - selector.predict(x))
    assert -selector.evaluate(x, y) < 5 * atol


def test_dantzig_constraint_is_active(sparse_problem):
    x, y = sparse_problem
    lambd, sigma = 2.0, 1.5
    dt = DantzigSelectorCP(lambd, sigma=sigma)
    dt.select(x, y)
    corr = np.linalg.norm(x.T @ (y - x @ dt.coef_), np.inf)
    # L1-minimal solution sits on the boundary of the feasible set
    assert corr == pytest.approx(lambd * sigma, rel=1e-4)


def test_lasso_cp_matches_sklearn(sparse_problem):
    """LassoCP minimises 1/(2n)||y - x b||^2 + lambd ||b||_1, identical to sklearn's objective."""
    x, y = sparse_problem
    lasso = LassoCP(0.1)
    lasso.select(x, y)
    ref = SklearnLasso(alpha=0.1, fit_intercept=False, tol=1e-12, max_iter=100000).fit(x, y)
    np.testing.assert_allclose(lasso.coef_, ref.coef_, atol=1e-5)


def test_adaptive_lasso_cp_matches_rescaled_lasso(sparse_problem):
    """Adaptive lasso is a lasso on features rescaled by 1/w, with coefficients rescaled back."""
    x, y = sparse_problem
    gamma = 1.0
    alasso = AdaptiveLassoCP(0.1, gamma=gamma)
    alasso.select(x, y)
    w = alasso.get_w(x, y)
    beta_ols = np.linalg.lstsq(x, y, rcond=None)[0]
    np.testing.assert_allclose(w, 1.0 / np.abs(beta_ols) ** gamma)
    ref = SklearnLasso(alpha=0.1, fit_intercept=False, tol=1e-12, max_iter=100000).fit(x / w, y)
    np.testing.assert_allclose(alasso.coef_, ref.coef_ / w, atol=1e-5)


@pytest.mark.parametrize(
    ("cp_selector", "scipy_selector", "atol"),
    [
        (DantzigSelectorCP(1.0), DantzigSelector(1.0), 1e-4),
        (LassoCP(0.1), Lasso(0.1), 1e-4),
        (AdaptiveLassoCP(0.1, gamma=0.5), AdaptiveLasso(0.1, gamma=0.5), 1e-3),
    ],
    ids=["dantzig", "lasso", "adaptive_lasso"],
)
def test_cp_agrees_with_scipy(cp_selector, scipy_selector, atol, sparse_problem):
    x, y = sparse_problem
    sel_cp = cp_selector.select(x, y)
    sel_scipy = scipy_selector.select(x, y)
    np.testing.assert_array_equal(sel_cp, sel_scipy)
    np.testing.assert_allclose(cp_selector.coef_, scipy_selector.coef_, atol=atol)


@pytest.mark.parametrize(
    "cls_args",
    [(DantzigSelectorCP, (1.0,)), (LassoCP, (0.1,)), (AdaptiveLassoCP, (0.1, 1.0))],
    ids=["dantzig", "lasso", "adaptive_lasso"],
)
def test_default_solver_is_installed(cls_args, sparse_problem):
    """Regression: the default solver used to be ECOS, which cvxpy >= 1.6 no longer ships."""
    cls, args = cls_args
    selector = cls(*args)
    assert selector.method is None
    np.testing.assert_array_equal(selector.select(*sparse_problem), TRUE_SUPPORT)


@pytest.mark.parametrize(
    ("method", "options", "coef_thres"),
    [("CLARABEL", None, 1e-6), ("SCS", {"eps": 1e-9, "max_iters": 100000}, 1e-3)],
)
def test_explicit_solver_and_options(method, options, coef_thres, sparse_problem):
    lasso = LassoCP(0.1, method=method, coef_thres=coef_thres)
    np.testing.assert_array_equal(lasso.select(*sparse_problem, options=options), TRUE_SUPPORT)


def test_infeasible_problem_raises(sparse_problem):
    """Regression: an infeasible problem used to fail with TypeError on np.abs(None)."""
    dt = DantzigSelectorCP(-1.0)
    with pytest.raises(RuntimeError, match="infeasible"):
        dt.select(*sparse_problem)
    assert not dt.is_fitted


def test_get_set_params():
    alasso = AdaptiveLassoCP(0.1, gamma=1.0)
    assert alasso.get_params() == {"gamma": 1.0, "lambd": 0.1}
    alasso.set_params(lambd=0.5)
    assert alasso.lambd == 0.5
    assert DantzigSelectorCP(1.0).get_params() == {"lambd": 1.0, "sigma": 1.0}


def test_base_classes(sparse_problem):
    x, y = sparse_problem
    beta = cp.Variable(x.shape[1])
    base = BaseSelectorCP()
    assert base.construct_constraints(x, y, beta) is None
    with pytest.raises(NotImplementedError):
        base.construct_loss(x, y, beta)
    with pytest.raises(NotImplementedError):
        base.select(x, y)

    # Unpenalized least squares recovers the OLS solution
    pls = PenalizedLeastSquaresCP()
    assert pls.penalty(beta) == 0.0
    pls.select(x, y)
    np.testing.assert_allclose(pls.coef_, np.linalg.lstsq(x, y, rcond=None)[0], atol=1e-6)


def test_sis_with_cp_selector(sparse_problem):
    x, y = sparse_problem
    sis = SIS(gamma=0.5, selector=DantzigSelectorCP(1.0), verbose=False)
    np.testing.assert_array_equal(np.sort(sis.run(x, y)), TRUE_SUPPORT)
