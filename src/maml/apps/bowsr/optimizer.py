"""Module implements the BayesianOptimizer."""

from __future__ import annotations

import warnings
from copy import copy
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic

from maml.apps.bowsr.acquisition import AcquisitionFunction, ensure_rng, propose_query_point
from maml.apps.bowsr.perturbation import LatticePerturbation, WyckoffPerturbation, get_standardized_structure
from maml.apps.bowsr.preprocessing import DummyScaler, StandardScaler
from maml.apps.bowsr.target_space import TargetSpace

if TYPE_CHECKING:
    from maml.apps.bowsr.model import EnergyModel


def struct2perturbation(
    structure: Structure,
    use_symmetry: bool = True,
    wyc_tol: float = 0.3 * 1e-3,
    abc_tol: float = 1e-3,
    angle_tol: float = 2e-1,
    symprec: float = 1e-2,
) -> tuple[list[WyckoffPerturbation], list[int], dict, LatticePerturbation]:
    """
    Get the symmetry-driven perturbation of the structure.

    Args:
        structure (Structure): Pymatgen Structure object.
        use_symmetry (bool): Whether to use constraint of symmetry to reduce
            parameters space.
        wyc_tol (float): Tolerance for wyckoff symbol determined coordinates.
        abc_tol (float): Tolerance for lattice lengths determined by crystal system.
        angle_tol (float): Tolerance for lattice angles determined by crystal system.
        symprec (float): Tolerance for symmetry finding.

    Return:
        wps (list): WyckoffPerturbations for derivation of symmetrically
            unique sites. Used to derive the coordinates of the sites.
        indices (list): Indices of symmetrically unique sites.
        mapping (dict): A mapping dictionary that maps equivalent atoms
            onto each other.
        lp (LatticePerturbation): LatticePerturbation for derivation of lattice
            of the structure.
    """
    sa = SpacegroupAnalyzer(structure, symprec=symprec)
    sd = sa.get_symmetry_dataset()
    spg_int_number = sd["number"]
    wyckoffs = sd["wyckoffs"]
    symm_ops = [
        SymmOp.from_rotation_and_translation(rotation, translation, tol=0.1)
        for rotation, translation in zip(sd["rotations"], sd["translations"], strict=False)
    ]

    if use_symmetry:
        equivalent_atoms = sd["equivalent_atoms"]
        mapping = dict(zip(range(structure.num_sites), equivalent_atoms, strict=False))
        symmetrically_distinct_indices = np.unique(equivalent_atoms)
        indices = symmetrically_distinct_indices.tolist()
    else:
        indices = list(range(structure.num_sites))
        mapping = dict(zip(indices, indices, strict=False))

    wps = []
    for i in indices:
        wyckoff_symbol = wyckoffs[i]
        wp = WyckoffPerturbation(spg_int_number, wyckoff_symbol, symmetry_ops=symm_ops, use_symmetry=use_symmetry)
        wp.sanity_check(structure[i], wyc_tol=wyc_tol)
        wps.append(wp)
    # wps = [wp for wp in wps]

    lp = LatticePerturbation(spg_int_number, use_symmetry=use_symmetry)
    lp.sanity_check(structure.lattice, abc_tol=abc_tol, angle_tol=angle_tol)

    return (wps, indices, mapping, lp)  # type: ignore


def atoms_crowded(structure: Structure, cutoff_distance: float = 1.1) -> bool:
    """
    Identify whether structure is unreasonable because the atoms are "too close".

    Args:
        structure (Structure): Pymatgen Structure object.
        cutoff_distance (float): The minimum allowed atomic distance.
    """
    distance_matrix = copy(structure.distance_matrix)
    distance_matrix[distance_matrix == 0] = np.inf
    return bool(np.min(distance_matrix) < cutoff_distance)


class BayesianOptimizer:
    """Bayesian optimizer used to optimize the structure."""

    def __init__(
        self,
        model: EnergyModel,
        structure: Structure,
        relax_coords: bool = True,
        relax_lattice: bool = True,
        use_symmetry: bool = True,
        use_scaler: bool = True,
        noisy: bool = True,
        seed: int | None = None,
        **kwargs,
    ):
        """
        Args:
            model (EnergyModel): energy model exposes `predict_energy` method.
            structure (Structure): Pymatgen Structure object.
            relax_coords (bool): Whether to relax symmetrical coordinates.
            relax_lattice (bool): Whether to relax lattice.
            use_symmetry (bool): Whether to use constraint of symmetry to reduce
                parameters space.
            use_scaler (bool): Whether to use scaler for Gaussian process regression.
            noisy (bool): Whether to perform noise-based Bayesian optimization
                (predictive noisy test data) or noise-free Bayesian optimization
                (predictive GP posterior).
            seed (int): Seeded rng for random number generator. None
                for an unseeded rng.
            **kwargs: Passthrough.
        """
        random_state = ensure_rng(seed)
        self.model = model
        self.noisy = noisy
        self.use_symmetry = use_symmetry
        scaler = StandardScaler() if use_scaler else DummyScaler()
        self.scaler = scaler

        structure.remove_oxidation_states()
        standardized_structure = get_standardized_structure(structure)
        wps, _, _, lp = struct2perturbation(standardized_structure, **kwargs)

        u = 0
        while not (all(wp.fit_site for wp in wps) and lp.fit_lattice) and u < 3:
            standardized_structure = get_standardized_structure(standardized_structure)
            wps, _, _, lp = struct2perturbation(standardized_structure, use_symmetry, **kwargs)
            u += 1
        if not (all(wp.fit_site for wp in wps) and lp.fit_lattice):
            raise ValueError("Standard structures can not be obtained.")

        self.structure = standardized_structure

        wps, indices, mapping, lp = struct2perturbation(self.structure, use_symmetry, **kwargs)
        wyckoff_dims = [wp.dim for wp in wps]
        abc_dim, angles_dim = lp.dims
        self._space = TargetSpace(
            target_func=self.get_formation_energy,
            wps=wps,
            abc_dim=abc_dim,
            angles_dim=angles_dim,
            relax_coords=relax_coords,
            relax_lattice=relax_lattice,
            scaler=scaler,  # type: ignore
            random_state=random_state,
        )
        self.wps = wps
        self.indices = indices
        self.mapping = mapping
        self.lp = lp
        self.relax_coords = relax_coords
        self.relax_lattice = relax_lattice
        self.wyckoff_dims = wyckoff_dims
        self.abc_dim = abc_dim
        self.angles_dim = angles_dim

        # Initialize internal GaussianProcessRegressor
        self._gpr = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale=1.0), alpha=0.028**2)

    def get_derived_structure(self, x: np.ndarray) -> Structure:
        """
        Get the derived structure.

        Args:
            x (ndarray): The input of getting perturbed structure.
        """
        struct = self.structure.copy()
        lattice = struct.lattice
        species = [dict(site.species.as_dict()) for site in struct]
        frac_coords = struct.frac_coords

        if self.relax_coords:
            x_wyckoff = np.concatenate(
                [
                    self.wps[i].perturbation_mode(x[sum(self.wyckoff_dims[:i]) : sum(self.wyckoff_dims[:i]) + num])
                    for i, num in enumerate(self.wyckoff_dims)
                ]
            )

            for i, (idx, wp) in enumerate(zip(self.indices, self.wps, strict=False)):
                p = wp.standardize(frac_coords[idx]) + x_wyckoff[i * 3 : (i + 1) * 3]
                orbit = wp.get_orbit(p)

                assert len(orbit) == len([k for k, v in self.mapping.items() if v == idx])

                frac_coords[[k for k, v in self.mapping.items() if v == idx]] = orbit

        x_lattice = (
            self.lp.perturbation_mode(x[-(self.abc_dim + self.angles_dim) :]) if self.relax_lattice else np.zeros(6)
        )
        abc = lattice.abc
        angles = lattice.angles
        lattice_parameters = np.array(abc + angles) + x_lattice
        lattice = Lattice.from_parameters(*lattice_parameters)

        return Structure(lattice=lattice, species=species, coords=frac_coords)

    def get_formation_energy(self, x: np.ndarray) -> float:
        """
        Calculate the formation energy of the perturbed structure. Absolute value
        is calculated on practical purpose of maximization of target function in
        Bayesian optimization.

        Args:
            x (ndarray): The input of formation energy calculation.
        """
        derived_struct = self.get_derived_structure(x)
        return -self.model.predict_energy(derived_struct)

    def add_query(self, x: np.ndarray) -> float:
        """
        Add query point into the TargetSpace.

        Args:
            x (ndarray): Query point.
        """
        try:
            target = self.space.probe(x)
        except:  # noqa
            target = 0
        return target

    def propose(self, acquisition_function: AcquisitionFunction, n_warmup: int, sampler: str) -> np.ndarray:
        """
        Suggest the next most promising point.

        Args:
            acquisition_function (AcquisitionFunction): AcquisitionFunction.
            n_warmup (int): Number of randomly sampled points to select the initial
                point for minimization.
            sampler (str): Sampler. Options are Latin Hyperparameter Sampling and uniform sampling.
        """
        # Sklearn's GaussianProcessRegressor throws a large number of warnings at times
        # which are unnecessary to be seen.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gpr.fit(self.space.params, self.space.target)

        if self.noisy:
            y_max = np.max(self.gpr.predict(self.space.params))
            noise = 0.028
        else:
            y_max = np.max(self.space.target)
            noise = 0.0

        # Find the next point that maximize the acquisition function.
        return propose_query_point(
            acquisition=acquisition_function.calculate,
            scaler=self.scaler,
            gpr=self.gpr,
            y_max=y_max,
            noise=noise,
            bounds=self.space.bounds,
            random_state=self.space.random_state,
            sampler=sampler,
            n_warmup=n_warmup,
        )

    def optimize(
        self,
        n_init: int,
        n_iter: int,
        acq_type: str = "ei",
        kappa: float = 2.576,
        xi: float = 0.0,
        n_warmup: int = 1000,
        is_continue: bool = False,
        sampler: str = "lhs",
        **gpr_params,
    ) -> None:
        """
        Optimize the coordinates and/or lattice of the structure by minimizing the
        model predicted formation energy of the structure. Model prediction error
        can be considered as a constant white noise.

        Args:
            n_init (int): The number of initial points.
            n_iter (int): The number of iteration steps.
            acq_type (str): The type of acquisition function. Three choices are given,
                ucb: Upper confidence bound,
                ei: Expected improvement,
                poi: Probability of improvement.
            kappa (float): Tradeoff parameter used in upper confidence bound formulism.
            xi (float): Tradeoff parameter between exploitation and exploration.
            n_warmup (int): Number of randomly sampled points to select the initial
                point for minimization.
            is_continue (bool): whether to continue previous run without resetting GPR
            sampler (str): Sampler generating initial points. "uniform" or "lhs".
            **gpr_params: Passthrough.
        """
        # print('Optimize begins!')
        self.set_gpr_params(**gpr_params)
        if not is_continue:
            self._gpr = clone(self._gpr)
        acq = AcquisitionFunction(acq_type=acq_type, kappa=kappa, xi=xi)

        # Add the initial structure into known data
        if n_init != 0:
            if sampler == "lhs":
                params = np.concatenate([np.zeros((1, self.space.dim)), self.space.lhs_sample(n_init)])
            else:
                params = np.concatenate(
                    ([np.zeros(self.space.dim)], [self.space.uniform_sample() for _ in range(n_init)])
                )
            self.scaler.fit(params)  # type: ignore

            for x in params:
                self.add_query(x)

        iteration = 0
        while iteration < n_iter:
            x_next = self.propose(acquisition_function=acq, n_warmup=n_warmup, sampler=sampler)
            self.add_query(x_next)
            iteration += 1

    def get_optimized_structure_and_energy(self, cutoff_distance: float = 1.1) -> tuple:
        """
        Args:
            cutoff_distance (float): Cutoff distance of the allowed shortest atomic distance in reasonable structures.
                When the cutoff_distance is 0, any structures will be considered reasonable.
        """
        optimized_structure = self.structure.copy()
        idx = 0
        for idx in np.argsort(self.space.target)[::-1]:
            optimized_structure = self.get_derived_structure(
                self.scaler.inverse_transform(self.space.params[idx])  # type: ignore
            )
            if not atoms_crowded(optimized_structure, cutoff_distance=cutoff_distance):
                break
        return (optimized_structure, -self.space.target[idx])

    def set_space_empty(self) -> None:
        """Empty the target space."""
        self.space.set_empty()

    def set_bounds(self, **bounds_parameter) -> None:
        """Set the bound value of wyckoff perturbation and lattice perturbation."""
        if bounds_parameter.get("abc_bound") is not None:
            abc_bound = bounds_parameter.get("abc_bound")
            bounds_parameter.pop("abc_bound")
        elif bounds_parameter.get("abc_ratio") is not None:
            abc_ratio = bounds_parameter.get("abc_ratio")
            bounds_parameter.pop("abc_ratio")
            abc_bound = np.array(self.lp.abc)[:, np.newaxis] * abc_ratio
        else:
            abc_bound = np.array(self.lp.abc)[:, np.newaxis] * 0.2
        self.space.set_bounds(abc_bound=abc_bound, **bounds_parameter)

    def set_gpr_params(self, **gpr_params) -> None:
        """Set the parameters of internal GaussianProcessRegressor."""
        self._gpr.set_params(**gpr_params)

    @property
    def space(self):
        """Returns the target space."""
        return self._space

    @property
    def gpr(self):
        """Returns the Gaussian Process regressor."""
        return self._gpr

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(relax_coords={self.relax_coords}, relax_lattice={self.relax_lattice}, "
            f"use_symmetry={self.use_symmetry}"
            f"\n\t\twyckoff_dims={self.wyckoff_dims}, abc_dim={self.abc_dim}, "
            f"\n\t\tangles_dim={self.angles_dim}, kernel={self.gpr.kernel!r}, "
            f"scaler={self.scaler.__class__.__name__}, noisy={self.noisy})"
        )

    def as_dict(self):
        """Dict representation of BayesianOptimizer."""

        def serialize(t) -> tuple:
            """
            Serialize the object.

            Args:
                t: object
            Returns:

            """
            return tuple([int(num) for num in item] if isinstance(item, np.ndarray) else item for item in t)

        def gpr_as_dict(gpr):
            """Serialize the GaussianProcessRegressor."""
            gpr_dict = gpr.get_params()
            gpr_dict["X_train_"] = gpr.X_train_.tolist()
            gpr_dict["alpha_"] = gpr.alpha_.tolist()
            gpr_dict["_y_train_mean"] = gpr._y_train_mean.tolist()
            gpr_dict["_K_inv"] = gpr._K_inv.tolist()
            gpr_kernel = gpr_dict["kernel"]
            kernel_name = gpr_kernel.__class__.__name__
            kernel_params = gpr_kernel.get_params()
            kernel_opt_params = gpr.kernel_.get_params()

            gpr_dict["kernel"] = {"name": kernel_name, "params": kernel_params, "opt_params": kernel_opt_params}

            return gpr_dict

        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "structure": self.structure.as_dict(),
            "relax_coords": self.relax_coords,
            "relax_lattice": self.relax_lattice,
            "use_symmetry": self.use_symmetry,
            "space": {
                "params": self.space.params.tolist(),
                "target": self.space.target.tolist(),
                "bounds": self.space.bounds.tolist(),
                "random_state": serialize(self.space.random_state.get_state()),
            },
            "gpr": gpr_as_dict(self.gpr),
            "model": self.model.__class__.__name__,
            "scaler": self.scaler.as_dict(),
            "noisy": self.noisy,
        }

    @classmethod
    def from_dict(cls, d):
        """
        Reconstitute a BayesianOptimizer object from a dict representation of
        BayesianOptimizer created using as_dict().

        Args:
            d (dict): Dict representation of BayesianOptimizer.
        """

        def gpr_from_dict(gpr_d):
            """Instantiate GaussianProcessRegressor from serialization."""
            import sklearn.gaussian_process.kernels as sk_kernels

            d = gpr_d.copy()
            X_train_ = d.pop("X_train_")
            alpha_ = d.pop("alpha_")
            _y_train_mean = d.pop("_y_train_mean")
            _K_inv = d.pop("_K_inv")
            kernel_d = d["kernel"]
            kernel = getattr(sk_kernels, kernel_d["name"])(**kernel_d["params"])
            kernel_ = getattr(sk_kernels, kernel_d["name"])(**kernel_d["opt_params"])
            d["kernel"] = kernel

            gpr = GaussianProcessRegressor()
            gpr.kernel = kernel
            gpr.kernel_ = kernel_
            gpr.set_params(**d)
            gpr.X_train_ = X_train_
            gpr.alpha_ = alpha_
            gpr._y_train_mean = _y_train_mean
            gpr._K_inv = _K_inv

            return gpr

        if d["model"] == "MEGNet":
            import maml.apps.bowsr.model.megnet as energy_model

            model = getattr(energy_model, d["model"])()
        elif d["model"] == "CGCNN":
            import maml.apps.bowsr.model.cgcnn as energy_model

            model = getattr(energy_model, d["model"])()
        elif d["model"] == "DFT":
            import maml.apps.bowsr.model.dft as energy_model

            model = getattr(energy_model, d["model"])()
        else:
            raise AttributeError(f"model {d['model']} is not supported.")

        structure = Structure.from_dict(d["structure"])
        use_symmetry = d["use_symmetry"]
        wps, indices, mapping, lp = struct2perturbation(structure, use_symmetry)
        wyckoff_dims = [wp.dim for wp in wps]
        abc_dim, angles_dim = lp.dims
        relax_coords = d["relax_coords"]
        relax_lattice = d["relax_lattice"]
        noisy = d["noisy"]
        optimizer = BayesianOptimizer(
            model=model,
            structure=structure,
            relax_coords=relax_coords,
            relax_lattice=relax_lattice,
            use_symmetry=use_symmetry,
            noisy=noisy,
        )
        optimizer.structure = structure
        optimizer.wps = wps
        optimizer.indices = indices
        optimizer.mapping = mapping
        optimizer.lp = lp
        optimizer.wyckoff_dims = wyckoff_dims
        optimizer.abc_dim = abc_dim
        optimizer.angles_dim = angles_dim
        gpr = gpr_from_dict(d["gpr"])
        optimizer._gpr = gpr

        space_d = d["space"]
        space_params = np.array(space_d["params"])
        space_target = np.array(space_d["target"])
        space_bounds = np.array(space_d["bounds"])
        space_random_state = np.random.RandomState()  # pylint: disable=E1101
        space_random_state.set_state(space_d["random_state"])
        from maml.apps.bowsr import preprocessing

        scaler = getattr(preprocessing, d["scaler"]["@class"])(**d["scaler"]["params"])
        optimizer.scaler = scaler
        optimizer._space.scaler = scaler
        for x, target in zip(space_params, space_target, strict=False):
            optimizer._space._params = np.concatenate([optimizer._space._params, x.reshape(1, -1)])
            optimizer._space._target = np.concatenate([optimizer._space._target, [target]])
        optimizer._space._bounds = space_bounds
        optimizer._space.random_state = space_random_state

        return optimizer
