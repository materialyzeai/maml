"""Selectors."""

from __future__ import annotations

import inspect
from collections import defaultdict
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import lstsq
from scipy.optimize import NonlinearConstraint, minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import get_scorer

if TYPE_CHECKING:
    from collections.abc import Callable


class BaseSelector:
    """
    Feature selector. This is meant to work on relatively smaller
    number of features.
    """

    def __init__(self, coef_thres: float = 1e-6, method: str = "SLSQP"):
        """
        Base selector
        Args:
            coef_thres (float): threshold to discard certain coefficients
            method (str): optimization methods in scipy.optimize.minimize.
        """
        self.coef_thres = coef_thres
        self.is_fitted = False
        self.coef_: np.ndarray | None = None
        self.method = method
        self.indices: np.ndarray | None = None

    def select(self, x: np.ndarray, y: np.ndarray, options: dict | None = None) -> np.ndarray | None:
        """
        Select feature indices from x
        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            options (dict): options in the optimizations provided
                to scipy.optimize.minimize
        Returns: list of int indices.
        """
        _n_data, n_dim = x.shape
        options = options or {"maxiter": 1e4, "ftol": 1e-12}
        res = minimize(
            lambda beta: self.construct_loss(x=x, y=y, beta=beta),
            [0] * n_dim,
            jac=self.construct_jac(x=x, y=y),
            method=self.method,
            constraints=self.construct_constraints(x=x, y=y),
            options=options,
        )
        if res.status != 0:
            raise RuntimeError(f"Not converged, status {res.status}")
        self.is_fitted = True
        self.coef_ = res.x
        # output coefficient indices that are above certain thresholds
        self.indices = np.where(np.abs(self.coef_) > self.coef_thres)[0]  # type: ignore
        self.coef_[np.where(np.abs(self.coef_) <= self.coef_thres)[0]] = 0.0  # type: ignore
        return self.indices

    def construct_loss(self, x: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
        """
        Get loss function from data and tentative coefficients beta
        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            beta (np.ndarray): N coefficients
        Returns: loss value.
        """
        raise NotImplementedError

    def construct_constraints(
        self, x: np.ndarray, y: np.ndarray, beta: np.ndarray | None = None
    ) -> dict | list | NonlinearConstraint | None:
        """
        Get constraints dictionary from data, e.g.,
        {"func": lambda beta: fun(x, y, beta), "type": "ineq"}.

        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            beta (np.ndarray): parameter to optimize
        Returns: dict of constraints.
        """
        return None

    def construct_jac(self, x: np.ndarray, y: np.ndarray) -> Callable | None:
        """
        Jacobian of cost function
        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
        Returns: Jacobian function.
        """
        return None

    def evaluate(self, x: np.ndarray, y: np.ndarray, metric: str = "neg_mean_absolute_error") -> float:
        """
        Evaluate the linear models using x, and y test data.

        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            metric (str): scorer function, used with
                sklearn.metrics.get_scorer
        Returns:
        """
        metric_func = get_scorer(metric)
        lr = LinearRegression(fit_intercept=False)
        lr.coef_ = self.coef_[self.indices]  # type: ignore
        lr.intercept_ = 0
        return metric_func(lr, x[:, self.indices], y)

    def get_coef(self) -> np.ndarray | None:
        """
        Get coefficients
        Returns: the coefficients array.
        """
        return self.coef_

    def get_feature_indices(self) -> np.ndarray | None:
        """
        Get selected feature indices.

        Returns: ndarray
        """
        return self.indices

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the results using sparsified coefficients.

        Args:
            x (np.ndarray): design matrix
        Returns: ndarray
        """
        return x[:, self.indices].dot(self.coef_[self.indices])  # type: ignore

    def compute_residual(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute.

        Args:
            x (np.ndarray): design matrix
            y (np.ndarray): target vector
        Returns: residual vector.
        """
        return y - self.predict(x)

    @classmethod
    def _get_param_names(cls):
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            return []
        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values() if p.name != "self" and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_KEYWORD:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    f" {cls} with constructor {init_signature} doesn't "
                    " follow this convention."
                )
        return sorted(p.name for p in parameters)

    def get_params(self):
        """
        Get params for this selector.

        Returns: mapping of string to any
                parameter names mapped to their values

        """
        out = {}
        for key in self._get_param_names():
            value = getattr(self, key, None)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this selector
        Args:
            **params: dict
            Selector parameters.

        Returns:
            self: selector instance

        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params()

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key} for selector {self}. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self


class DantzigSelector(BaseSelector):
    """
    Equation 11 in
    https://orfe.princeton.edu/~jqfan/papers/06/SIS.pdf
    and reference in https://projecteuclid.org/download/pdfview_1/euclid.aos/1201012958.
    """

    def __init__(self, lambd, sigma=1.0, **kwargs):
        """
        Dantzig selector.

        Args:
            lambd: tunable parameter
            sigma: standard deviation of the error.
            **kwargs: passthrough.
        """
        self.lambd = lambd
        self.sigma = sigma
        super().__init__(**kwargs)

    def construct_loss(self, x, y, beta) -> float:
        """
        Get loss function from data and tentative coefficients beta.

        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            beta (np.ndarray): N coefficients
        Returns: loss value.
        """
        return np.sum(np.abs(beta)).item()

    def construct_jac(self, x: np.ndarray, y: np.ndarray) -> Callable:
        """
        Jacobian of cost functions.

        Args:
            x: ndarray
            y: ndarray
        Returns: callable
        """

        def _jac(beta):
            sign = np.sign(beta)
            sign[np.abs(sign) < 0.1] = 1.0
            sign *= 30.0  # multiply the gradients to get better convergence
            return sign

        return _jac

    def construct_constraints(
        self, x: np.ndarray, y: np.ndarray, beta: np.ndarray | None = None
    ) -> NonlinearConstraint:
        """
        Get constraints dictionary from data, e.g.,
        {"func": lambda beta: fun(x, y, beta), "type": "ineq"}.

        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            beta (np.ndarray): placeholder
        Returns: dict of constraints.
        """

        def _constraint(beta):
            return np.linalg.norm(x.T @ (y - x @ beta), np.inf)

        def _jac(beta):
            vec = x.T @ (y - x @ beta)
            max_ind = np.argmax(np.abs(vec))
            der = np.zeros_like(vec.ravel())
            der[max_ind] = np.sign(vec[max_ind])
            return -x.T.dot(x).dot(der)

        return NonlinearConstraint(_constraint, -np.inf, self.lambd * self.sigma, jac=_jac)


class PenalizedLeastSquares(BaseSelector):
    """
    Penalized least squares. In addition to minimizing the sum of squares loss,
    it adds an additional penalty to the coefficients.
    """

    def construct_loss(self, x: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
        """
        Construct the loss function. An extra penalty term is added
        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            beta (np.ndarray): N coefficients
        Returns: sum of errors.
        """
        n = x.shape[0]
        return 1.0 / (2 * n) * np.sum((y - x.dot(beta)) ** 2) + self.penalty(beta, x=x, y=y)

    def _sse_jac(self, x, y, beta):
        n = x.shape[0]
        return 1.0 / n * (y - x.dot(beta)).T.dot(-x)

    def _penalty_jac(self, x, y, beta):
        return 0.0

    def construct_jac(self, x: np.ndarray, y: np.ndarray):
        """
        Construct the jacobian of loss function
        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
        Returns: jacobian vector.
        """

        def _jac(beta):
            return self._sse_jac(x, y, beta) + self._penalty_jac(x, y, beta)

        return _jac

    def construct_constraints(self, x: np.ndarray, y: np.ndarray, beta: np.ndarray | None = None) -> list[dict | None]:
        """
        No constraints
        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            beta (np.ndarray): placeholder only
        Returns: a list of dictionary constraints.
        """
        return []

    def penalty(self, beta: np.ndarray, x: np.ndarray | None = None, y: np.ndarray | None = None) -> float:
        """
        Calculate the penalty from input x, output y and coefficient beta
        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            beta (np.ndarray): N coefficients
        Returns: penalty value.
        """
        return 0.0


class SCAD(PenalizedLeastSquares):
    """
    Smoothly clipped absolute deviation (SCAD),
    equation 12 and 13 in https://orfe.princeton.edu/~jqfan/papers/06/SIS.pdf.
    """

    def __init__(self, lambd: float | np.ndarray, a: float = 3.7, **kwargs):
        """
        Smoothly clipped absolute deviation.

        Args:
            lambd (float or list of floats): The weights for the penalty
            a (float): hyperparameter in SCAD penalty.
            **kwargs: passthrough
        """
        self.lambd = lambd
        self.a = a
        super().__init__(**kwargs)

    def penalty(self, beta: np.ndarray, x: np.ndarray | None = None, y: np.ndarray | None = None) -> float:
        """
        Calculate the SCAD penalty from input x, output y
            and coefficient beta
        Args:
            beta (np.ndarray): N coefficients
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
        Returns: penalty value.
        """
        beta_abs = np.abs(beta)
        penalty = (
            self.lambd * beta_abs * (beta_abs <= self.lambd)
            + -(beta_abs**2 - 2 * self.a * self.lambd * beta_abs + self.lambd**2)
            / (2 * (self.a - 1))
            * (beta_abs > self.lambd)
            * (beta_abs <= self.a * self.lambd)
            + (self.a + 1) * self.lambd**2 / 2.0 * (beta_abs > self.a * self.lambd)
        )
        return np.sum(penalty).item()

    def _penalty_jac(self, x, y, beta):
        beta = np.abs(beta)
        z = self.a * self.lambd - beta
        z[z < 0] = 0
        return self.lambd * (beta <= self.lambd + z / ((self.a - 1) * self.lambd) * (beta > self.lambd))


class Lasso(PenalizedLeastSquares):
    """Simple Lasso regression."""

    def __init__(self, lambd, **kwargs):
        """
        Lasso regression with lambda * norm_1(beta) as penalty.

        Args:
            lambd (float): weights for the penalty
            **kwargs: passthrough
        """
        self.lambd = lambd
        super().__init__(**kwargs)

    def penalty(self, beta: np.ndarray, x: np.ndarray | None = None, y: np.ndarray | None = None) -> float:
        """
        Calculate the penalty from input x, output y and coefficient beta.

        Args:
            beta (np.ndarray): N coefficients
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
        Returns: penalty value.
        """
        beta_abs = np.abs(beta)
        return np.sum(self.lambd * beta_abs).item()

    def _penalty_jac(self, x, y, beta):
        sign = np.sign(beta)
        sign[np.abs(sign) < 0.2] = 1
        return self.lambd * sign


class AdaptiveLasso(PenalizedLeastSquares):
    """Adaptive lasso regression using OLS coefficients as the root-n estimator coefficients."""

    def __init__(self, lambd, gamma, **kwargs):
        """
        Adaptive lasso regression.

        Args:
            lambd (float or list of floats): lambda values
            gamma (float): exponential for hat(beta)
            **kwargs: passthrough
        """
        self.lambd = lambd
        self.gamma = gamma
        self.w = 1
        super().__init__(**kwargs)

    def select(self, x, y, options=None) -> np.ndarray | None:
        """
        Select feature indices from x.

        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            options (dict): options in the optimizations provided
                to scipy.optimize.minimize
        Returns: list of int indices.
        """
        self.w = self.get_w(x, y)
        return super().select(x, y, options)

    def get_w(self, x, y) -> np.ndarray:
        """
        Get adaptive weights from data.

        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
        Returns: coefficients array.
        """
        beta_hat = lstsq(x, y)[0]
        return 1.0 / np.abs(beta_hat) ** self.gamma

    def penalty(self, beta: np.ndarray, x: np.ndarray | None = None, y: np.ndarray | None = None) -> float:
        """
        Calculate the penalty from input x, output y and coefficient beta.

        Args:
            beta (np.ndarray): N coefficients
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
        Returns: penalty value.
        """
        return np.sum(self.lambd * self.w * np.abs(beta)).item()

    def _penalty_jac(self, x, y, beta):
        sign = np.sign(beta)
        sign[np.abs(sign) < 0.2] = 1
        return self.lambd * self.w * sign


class L0BrutalForce(BaseSelector):
    """
    Brutal force combinatorial screening of features.
    This method takes all possible combinations of features
    and optimize the following loss function
        1/2 * mean((y-x @ beta)**2) + lambd * |beta|_0.
    """

    def __init__(self, lambd: float, **kwargs):
        """
        Initialization of L0 optimization.

        Args:
            lambd (float): penalty term
            **kwargs: passthrough
        """
        self.lambd = lambd
        super().__init__(**kwargs)

    def select(self, x: np.ndarray, y: np.ndarray, options: dict | None = None, n_job: int = 1) -> np.ndarray | None:
        """
        L0 combinatorial optimization.

        Args:
            x (np.ndarray): design matrix
            y (np.ndarray): target vector
            options: Dict of options.
            n_job (int): number of cpu
        Returns:
        """
        _n, p = x.shape
        index_array = list(range(p))

        def _lstsq(c):
            x_comb = x[:, c]
            beta = lstsq(x_comb, y)[0]
            res = 1.0 / 2 * np.mean((x_comb.dot(beta) - y) ** 2)
            penalty = self.lambd * len(c)
            res += penalty
            return res

        indices = []
        for p_temp in range(1, p + 1):
            indices.extend(list(combinations(index_array, p_temp)))
        loss = Parallel(n_jobs=n_job)(delayed(_lstsq)(comb) for comb in indices)
        argmin = np.argmin(loss)
        self.indices = np.array(indices[argmin])
        x_temp = x[:, self.indices]
        self.coef_ = np.zeros_like(x[0, :])
        self.coef_[self.indices] = lstsq(x_temp, y)[0]  # type: ignore
        return self.indices
