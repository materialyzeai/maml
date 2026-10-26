"""Module implements the new candidate proposal."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.special import erfc

if TYPE_CHECKING:
    from numpy.random import RandomState
    from sklearn.gaussian_process import GaussianProcessRegressor

EPS = np.finfo(float).eps


def _trunc(values: np.ndarray, decimals: int = 3):
    """
    Truncate values to decimal places
    Args:
        values (np.ndarray): input array
        decimals (int): number of decimals to keep.

    Returns: truncated array

    """
    return np.trunc(values * 10**decimals) / 10**decimals


def ensure_rng(seed: int | None = None) -> RandomState:
    """
    Create a random number generator based on an optional seed.
    This can be an integer for a seeded rng or None for an unseeded rng.
    """
    return np.random.RandomState(seed=seed)  # pylint: disable=E1101


def predict_mean_std(x: list | np.ndarray, gpr: GaussianProcessRegressor, noise: float) -> tuple:
    """
    Speed up the gpr.predict method by manually computing the kernel operations.

    Args:
        x (list/ndarray): Query point need to be evaluated.
        gpr (GaussianProcessRegressor): A Gaussian process regressor fitted to
            known data.
        noise (float): Noise added to standard deviation if test target
            instead of GP posterior is sampled. 0 otherwise.
    """
    # cache result of K_inv computation.
    X = np.array(x)

    # todo: @yunxing the cache mechanism is causing inconsistent shapes for
    # the dot product line. Please double check. I am disabling the cache for now
    if True:  # getattr(gpr, "_K_inv", None) is None:
        L_inv = solve_triangular(gpr.L_.T, np.eye(gpr.L_.shape[0]))
        gpr._K_inv = L_inv.dot(L_inv.T)

    K_trans = gpr.kernel_(X, gpr.X_train_)
    y_mean = K_trans.dot(gpr.alpha_) + gpr._y_train_mean
    y_var = gpr.kernel_.diag(X)
    y_var = y_var - np.einsum("ij,ij->i", np.dot(K_trans, gpr._K_inv), K_trans)
    y_var += noise**2
    return (y_mean, np.sqrt(y_var))


def lhs_sample(n_intervals: int, bounds: np.ndarray, random_state: RandomState) -> np.ndarray:
    """
    Latin hypercube sampling.

    Args:
        n_intervals (int): Number of intervals.
        bounds (nd.array): Bounds for each dimension.
        random_state (RandomState): Random state.
    """
    bounds = np.array(bounds)
    dim = len(bounds)
    linspace = np.linspace(bounds[:, 0], bounds[:, 1], n_intervals + 1)
    linspace = _trunc(linspace, decimals=3)
    lower = linspace[:n_intervals, :]
    upper = linspace[1 : n_intervals + 1, :]
    _center = (lower + upper) / 2
    return np.stack([random_state.permutation(_center[:, i]) for i in range(dim)]).T


def propose_query_point(
    acquisition, scaler, gpr, y_max, noise, bounds, random_state, sampler, n_warmup=10000
) -> np.ndarray:
    """
    Strategy used to find the maximum of the acquisition function.
    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method by first sampling `n_warmup` points at random
    and running L-BFGS-B from `n_iter` random starting points.

    Args:
        acquisition: The acquisition function.
        scaler: The Scaler used to transform features.
        gpr (GaussianProcessRegressor): A Gaussian process regressor fitted to
            known data.
        y_max (float): The current maximum target value.
        noise (float): The noise added to the acquisition function if noisy-based
            Bayesian optimization was performed.
        bounds (ndarray): The bounds of candidate points.
        random_state (RandomState): Random number generator.
        sampler (str): Sampler generating warmup points. "uniform" or "lhs".
        n_warmup (int): Number of randomly sampled points to select the initial
            point for minimization.
    """
    dim = bounds.shape[0]
    if sampler == "lhs":
        x_warmup = lhs_sample(n_warmup, bounds, random_state)
    else:
        x_warmup = _trunc(random_state.uniform(bounds[:, 0], bounds[:, 1], size=(n_warmup, dim)), decimals=3)
    x_warmup = scaler.transform(x_warmup)
    acq_warmup = acquisition(x_warmup, gpr=gpr, y_max=y_max, noise=noise)
    x_max = x_warmup[acq_warmup.argmax()]
    acq_max = acq_warmup.max()

    def min_obj(x):
        # Minimize objective is the negative acquisition function
        return -acquisition(x.reshape(-1, dim), gpr=gpr, y_max=y_max, noise=noise)

    # The GPR (and hence the acquisition function) lives in the scaled space,
    # so the local optimization must be bounded in the scaled space as well.
    scaled_bounds = np.sort(np.asarray(scaler.transform(np.asarray(bounds, dtype=float).T)).T, axis=1)

    # make sure that the initial conditions fall into the bounds
    x0 = np.clip(x_max, scaled_bounds[:, 0] + 3 * EPS, scaled_bounds[:, 1] - 3 * EPS)

    res = minimize(min_obj, x0=x0, bounds=scaled_bounds, method="L-BFGS-B")
    if -float(res.fun) >= acq_max:
        x_max = res.x
    return _trunc(scaler.inverse_transform(x_max), decimals=3)


class AcquisitionFunction:
    """An object to compute the acquisition functions."""

    def __init__(self, acq_type: str, kappa: float, xi: float):
        """
        Args:
            acq_type (str): The type of acquisition function. Three choices are given,
                ucb: Upper confidence bound,
                ei: Expected improvement,
                poi: Probability of improvement.
            kappa (float): Tradeoff parameter used in upper confidence bound formulism.
            xi (float): Tradeoff parameter between exploitation and exploration.
        """
        self.kappa = kappa
        self.xi = xi

        if acq_type not in ["ucb", "ei", "poi", "gp-ucb"]:
            err_msg = f"The utility function {acq_type} has not been implemented, please choose one of ucb, ei, or poi."
            raise NotImplementedError(err_msg)
        self.acq_type = acq_type

    def calculate(self, x: list | np.ndarray, gpr: GaussianProcessRegressor, y_max: float, noise: float) -> np.ndarray:
        """
        Calculate the value of acquisition function.

        Args:
            x (ndarray): Query point need to be evaluated.
            gpr (GaussianProcessRegressor): A Gaussian process regressor fitted to
                known data.
            y_max (float): The current maximum target value.
            noise (float): Noise added to acquisition function if noisy-based Bayesian
                optimization is performed, 0 otherwise.
        """
        if self.acq_type not in ["ucb", "ei", "poi", "gp-ucb"]:
            raise ValueError("acq type not recognized")
        if self.acq_type == "ucb":
            return self._ucb(x, gpr, self.kappa, noise)
        if self.acq_type == "ei":
            return self._ei(x, gpr, y_max, self.xi, noise)
        if self.acq_type == "poi":
            return self._poi(x, gpr, y_max, self.xi, noise)
        return self._gpucb(x, gpr, noise)

    @staticmethod
    def _ucb(x: list | np.ndarray, gpr: GaussianProcessRegressor, kappa: float, noise: float) -> np.ndarray:
        mean, std = predict_mean_std(x, gpr, noise)
        return mean + kappa * std

    @staticmethod
    def _ei(x: list | np.ndarray, gpr: GaussianProcessRegressor, y_max: float, xi: float, noise: float) -> np.ndarray:
        mean, std = predict_mean_std(x, gpr, noise)

        imp = mean - y_max - xi
        z = imp / std
        pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        cdf = 0.5 * erfc(-z / np.sqrt(2))
        return imp * cdf + std * pdf

    @staticmethod
    def _poi(x: list | np.ndarray, gpr: GaussianProcessRegressor, y_max: float, xi: float, noise: float) -> np.ndarray:
        mean, std = predict_mean_std(x, gpr, noise)

        z = (mean - y_max - xi) / std
        return 0.5 * erfc(-z / np.sqrt(2))

    @staticmethod
    def _gpucb(x: list | np.ndarray, gpr: GaussianProcessRegressor, noise: float) -> np.ndarray:
        if not hasattr(gpr, "X_train_"):
            raise AttributeError("GP-UCB acquisition function can not be applued.")
        T = gpr.X_train_.shape[0]
        D = gpr.X_train_.shape[1]
        kappa = np.sqrt(4 * (D + 1) * np.log(T))

        mean, std = predict_mean_std(x, gpr, noise)
        return mean + kappa * std
