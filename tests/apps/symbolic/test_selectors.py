# flake8: noqa

import json
import os
import unittest

import numpy as np
from pymatgen.util.testing import PymatgenTest

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
    DantzigSelector,
    L0BrutalForce,
    Lasso,
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


if __name__ == "__main__":
    unittest.main()
