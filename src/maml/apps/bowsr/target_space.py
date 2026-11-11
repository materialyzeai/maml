"""Module implements the target space."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pymatgen.core.periodic_table import Element

from maml.apps.bowsr.acquisition import lhs_sample

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.random import RandomState

    from maml.apps.bowsr.perturbation import WyckoffPerturbation
    from maml.apps.bowsr.preprocessing import DummyScaler, StandardScaler


def _hashable(x):
    """Ensure that an point is hashable by a python dict."""
    return tuple(map(float, x))


class TargetSpace:
    """
    Holds the perturbations of coordinates/lattice (x_wyckoff/x_lattice)
    -- formation energy (Y). Allows for constant-time appends while
    ensuring no duplicates are added.
    """

    def __init__(
        self,
        target_func: Callable,
        wps: list[WyckoffPerturbation],
        abc_dim: int,
        angles_dim: int,
        relax_coords: bool,
        relax_lattice: bool,
        scaler: StandardScaler | DummyScaler,
        random_state: RandomState,
    ):
        """
        Args:
            target_func: Target function to be maximized.
            wps (list): WyckoffPerturbations for derivation of symmetrically
                unique sites. Used to derive the coordinates of the sites.
            abc_dim (int): Number of independent variables in lattice constants.
            angles_dim (int): Number of independent variables in lattice angles.
            relax_coords (bool): Whether to relax symmetrical coordinates.
            relax_lattice (bool): Whether to relax lattice.
            scaler: Scaler used in scaling params.
            random_state (RandomState): Seeded rng for random number generator. None
                for an unseeded rng.
        """
        self.target_func = target_func
        self.random_state = random_state
        self.wps = wps
        self.abc_dim = abc_dim
        self.angles_dim = angles_dim
        self.relax_coords = relax_coords
        self.relax_lattice = relax_lattice
        self.scaler = scaler
        wyckoff_dims = [wp.dim for wp in wps]

        if not any([relax_coords, relax_lattice]):
            raise ValueError(
                "No degrees of freedom for relaxation are given. "
                "Please check the relax_coords and relax_lattice arguments."
            )

        if all([relax_coords, relax_lattice]):
            self.dim = sum(wyckoff_dims) + abc_dim + angles_dim
        elif relax_coords:
            self.dim = sum(wyckoff_dims)
        else:
            self.dim = abc_dim + angles_dim

        # Preallocated memory for x and y points.
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=0)

    def __len__(self):
        assert len(self._params) == len(self._target)
        return len(self._target)

    @property
    def params(self) -> np.ndarray:
        """Returns the parameters in target space."""
        return self._params

    @property
    def target(self) -> np.ndarray:
        """Returns the target (i.e., formation energy) in target space."""
        return self._target

    @property
    def bounds(self) -> np.ndarray:
        """Returns the search space of parameters."""
        return self._bounds

    def register(self, x, target) -> None:
        """
        Append a point and its target value to the known data.

        Args:
            x (ndarray): A single query point.
            target (float): Target value.
        """
        # if x in self:
        #     raise ValueError("Data point {} is not unique.".format(x))

        # Insert data into unique dictionary.
        x = self.scaler.transform(x)
        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [target]])

    def probe(self, x) -> float:
        """
        Evaluate a single point x, to obtain the value y and then records
        them as observations.

        Args:
            x (ndarray): A single point.
        """
        target = self.target_func(x)
        self.register(x, target)
        return target

    def uniform_sample(self) -> np.ndarray:
        """Creates random points within the bounds of the space."""
        data = np.empty((1, self.dim))
        for col, (lower, upper) in enumerate(self.bounds):
            data.T[col] = np.round(self.random_state.uniform(lower, upper, size=1), decimals=3)
        return data.ravel()

    def lhs_sample(self, n_intervals: int) -> np.ndarray:
        """
        Latin hypercube sampling.

        Args:
            n_intervals (int): Number of intervals.
        """
        return lhs_sample(n_intervals, self.bounds, self.random_state)

    def set_bounds(
        self, abc_bound: float = 1.2, angles_bound: float = 5, element_wise_wyckoff_bounds: dict | None = None
    ) -> None:
        """
        Set the bound value of wyckoff perturbation and
        lattice perturbation/volume perturbation.
        """
        if not element_wise_wyckoff_bounds:
            element_wise_wyckoff_bounds = {}
        included_elements = list(element_wise_wyckoff_bounds.keys())
        element_wise_wyckoff_bounds = {
            frozenset({el: 1}.items()): bound for el, bound in element_wise_wyckoff_bounds.items()
        }
        element_wise_wyckoff_bounds.update(
            {frozenset({el.symbol: 1}): 0.2 for el in Element if el.symbol not in included_elements}
        )

        for wp in self.wps:
            frozen_key = frozenset(wp.site.species.as_dict().items())
            if frozen_key not in element_wise_wyckoff_bounds:
                element_wise_wyckoff_bounds.update({frozen_key: 0.2})

        wyckoff_bounds = np.concatenate(
            [
                np.tile([-1, 1], (wp.dim, 1))
                * element_wise_wyckoff_bounds[frozenset(wp.site.species.as_dict().items())]
                for wp in self.wps
            ]
        )
        abc_bounds = np.tile([-1, 1], (self.abc_dim, 1)) * abc_bound
        angles_bounds = np.tile([-1, 1], (self.angles_dim, 1)) * angles_bound

        if not any([self.relax_coords, self.relax_lattice]):
            raise ValueError(
                "No degrees of freedom for relaxation are given. "
                "Please check the relax_coords and relax_lattice arguments."
            )
        if all([self.relax_coords, self.relax_lattice]):
            self._bounds = np.concatenate((wyckoff_bounds, abc_bounds, angles_bounds))
        elif self.relax_coords:
            self._bounds = wyckoff_bounds
        else:
            self._bounds = np.concatenate((abc_bounds, angles_bounds))

    def set_empty(self) -> None:
        """Empty the param, target of the space."""
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=0)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(relax_coords={self.relax_coords}, relax_lattice={self.relax_lattice},"
            f" dim={self.dim}, length={len(self)})"
        )
