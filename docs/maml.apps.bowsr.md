---
layout: default
title: maml.apps.bowsr.md
nav_exclude: true
---

# maml.apps.bowsr package

Implementation of BOWSR paper
Zuo, Yunxing, et al.
“Accelerating Materials Discovery with Bayesian Optimization and Graph Deep Learning.”
arXiv preprint arXiv:2104.10242 (2021).

## Subpackages

* [maml.apps.bowsr.model package](maml.apps.bowsr.model.md)
  * `EnergyModel`
    * `EnergyModel.predict_energy()`
  * maml.apps.bowsr.model.base module
    * `EnergyModel`
      * `EnergyModel.predict_energy()`
  * maml.apps.bowsr.model.cgcnn module
  * maml.apps.bowsr.model.dft module
    * `DFT`
      * `DFT.predict_energy()`
  * maml.apps.bowsr.model.megnet module

## maml.apps.bowsr.acquisition module

Module implements the new candidate proposal.

### *class* maml.apps.bowsr.acquisition.AcquisitionFunction(acq_type: str, kappa: float, xi: float)

Bases: `object`

An object to compute the acquisition functions.

#### *static* \_ei(x: list | np.ndarray, gpr: GaussianProcessRegressor, y_max: float, xi: float, noise: float)

#### *static* \_gpucb(x: list | np.ndarray, gpr: GaussianProcessRegressor, noise: float)

#### *static* \_poi(x: list | np.ndarray, gpr: GaussianProcessRegressor, y_max: float, xi: float, noise: float)

#### *static* \_ucb(x: list | np.ndarray, gpr: GaussianProcessRegressor, kappa: float, noise: float)

#### calculate(x: list | np.ndarray, gpr: GaussianProcessRegressor, y_max: float, noise: float)

Calculate the value of acquisition function.

* **Parameters**
  * **x** (*ndarray*) – Query point need to be evaluated.
  * **gpr** (*GaussianProcessRegressor*) – A Gaussian process regressor fitted to
    known data.
  * **y_max** (*float*) – The current maximum target value.
  * **noise** (*float*) – Noise added to acquisition function if noisy-based Bayesian
    optimization is performed, 0 otherwise.

### maml.apps.bowsr.acquisition._trunc(values: ndarray, decimals: int = 3)

Truncate values to decimal places
:param values: input array
:type values: np.ndarray
:param decimals: number of decimals to keep.
:type decimals: int

Returns: truncated array

### maml.apps.bowsr.acquisition.ensure_rng(seed: int | None = None)

Create a random number generator based on an optional seed.
This can be an integer for a seeded rng or None for an unseeded rng.

### maml.apps.bowsr.acquisition.lhs_sample(n_intervals: int, bounds: np.ndarray, random_state: RandomState)

Latin hypercube sampling.

* **Parameters**
  * **n_intervals** (*int*) – Number of intervals.
  * **bounds** (*nd.array*) – Bounds for each dimension.
  * **random_state** (*RandomState*) – Random state.

### maml.apps.bowsr.acquisition.predict_mean_std(x: list | np.ndarray, gpr: GaussianProcessRegressor, noise: float)

Speed up the gpr.predict method by manually computing the kernel operations.

* **Parameters**
  * **x** (*list/ndarray*) – Query point need to be evaluated.
  * **gpr** (*GaussianProcessRegressor*) – A Gaussian process regressor fitted to
    known data.
  * **noise** (*float*) – Noise added to standard deviation if test target
    instead of GP posterior is sampled. 0 otherwise.

### maml.apps.bowsr.acquisition.propose_query_point(acquisition, scaler, gpr, y_max, noise, bounds, random_state, sampler, n_warmup=10000)

Strategy used to find the maximum of the acquisition function.
It uses a combination of random sampling (cheap) and the ‘L-BFGS-B’
optimization method by first sampling n_warmup points at random
and running L-BFGS-B from n_iter random starting points.

* **Parameters**
  * **acquisition** – The acquisition function.
  * **scaler** – The Scaler used to transform features.
  * **gpr** (*GaussianProcessRegressor*) – A Gaussian process regressor fitted to
    known data.
  * **y_max** (*float*) – The current maximum target value.
  * **noise** (*float*) – The noise added to the acquisition function if noisy-based
    Bayesian optimization was performed.
  * **bounds** (*ndarray*) – The bounds of candidate points.
  * **random_state** (*RandomState*) – Random number generator.
  * **sampler** (*str*) – Sampler generating warmup points. “uniform” or “lhs”.
  * **n_warmup** (*int*) – Number of randomly sampled points to select the initial
    point for minimization.

## maml.apps.bowsr.optimizer module

Module implements the BayesianOptimizer.

### *class* maml.apps.bowsr.optimizer.BayesianOptimizer(model: EnergyModel, structure: Structure, relax_coords: bool = True, relax_lattice: bool = True, use_symmetry: bool = True, use_scaler: bool = True, noisy: bool = True, seed: int | None = None, \*\*kwargs)

Bases: `object`

Bayesian optimizer used to optimize the structure.

#### add_query(x: ndarray)

Add query point into the TargetSpace.

* **Parameters**
  **x** (*ndarray*) – Query point.

#### as_dict()

Dict representation of BayesianOptimizer.

#### *classmethod* from_dict(d)

Reconstitute a BayesianOptimizer object from a dict representation of
BayesianOptimizer created using as_dict().

* **Parameters**
  **d** (*dict*) – Dict representation of BayesianOptimizer.

#### get_derived_structure(x: ndarray)

Get the derived structure.

* **Parameters**
  **x** (*ndarray*) – The input of getting perturbed structure.

#### get_formation_energy(x: ndarray)

Calculate the formation energy of the perturbed structure. Absolute value
is calculated on practical purpose of maximization of target function in
Bayesian optimization.

* **Parameters**
  **x** (*ndarray*) – The input of formation energy calculation.

#### get_optimized_structure_and_energy(cutoff_distance: float = 1.1)

* **Parameters**
  **cutoff_distance** (*float*) – Cutoff distance of the allowed shortest atomic distance in reasonable structures.
  When the cutoff_distance is 0, any structures will be considered reasonable.

#### *property* gpr()

Returns the Gaussian Process regressor.

#### optimize(n_init: int, n_iter: int, acq_type: str = ‘ei’, kappa: float = 2.576, xi: float = 0.0, n_warmup: int = 1000, is_continue: bool = False, sampler: str = ‘lhs’, \*\*gpr_params)

Optimize the coordinates and/or lattice of the structure by minimizing the
model predicted formation energy of the structure. Model prediction error
can be considered as a constant white noise.

* **Parameters**
  * **n_init** (*int*) – The number of initial points.
  * **n_iter** (*int*) – The number of iteration steps.
  * **acq_type** (*str*) – The type of acquisition function. Three choices are given,
    ucb: Upper confidence bound,
    ei: Expected improvement,
    poi: Probability of improvement.
  * **kappa** (*float*) – Tradeoff parameter used in upper confidence bound formulism.
  * **xi** (*float*) – Tradeoff parameter between exploitation and exploration.
  * **n_warmup** (*int*) – Number of randomly sampled points to select the initial
    point for minimization.
  * **is_continue** (*bool*) – whether to continue previous run without resetting GPR
  * **sampler** (*str*) – Sampler generating initial points. “uniform” or “lhs”.
  * **\*\*gpr_params** – Passthrough.

#### propose(acquisition_function: AcquisitionFunction, n_warmup: int, sampler: str)

Suggest the next most promising point.

* **Parameters**
  * **acquisition_function** (*AcquisitionFunction*) – AcquisitionFunction.
  * **n_warmup** (*int*) – Number of randomly sampled points to select the initial
    point for minimization.
  * **sampler** (*str*) – Sampler. Options are Latin Hyperparameter Sampling and uniform sampling.

#### set_bounds(\*\*bounds_parameter)

Set the bound value of wyckoff perturbation and lattice perturbation.

#### set_gpr_params(\*\*gpr_params)

Set the parameters of internal GaussianProcessRegressor.

#### set_space_empty()

Empty the target space.

#### *property* space()

Returns the target space.

### maml.apps.bowsr.optimizer.atoms_crowded(structure: Structure, cutoff_distance: float = 1.1)

Identify whether structure is unreasonable because the atoms are “too close”.

* **Parameters**
  * **structure** (*Structure*) – Pymatgen Structure object.
  * **cutoff_distance** (*float*) – The minimum allowed atomic distance.

### maml.apps.bowsr.optimizer.struct2perturbation(structure: Structure, use_symmetry: bool = True, wyc_tol: float = 0.0003, abc_tol: float = 0.001, angle_tol: float = 0.2, symprec: float = 0.01)

Get the symmetry-driven perturbation of the structure.

* **Parameters**
  * **structure** (*Structure*) – Pymatgen Structure object.
  * **use_symmetry** (*bool*) – Whether to use constraint of symmetry to reduce
    parameters space.
  * **wyc_tol** (*float*) – Tolerance for wyckoff symbol determined coordinates.
  * **abc_tol** (*float*) – Tolerance for lattice lengths determined by crystal system.
  * **angle_tol** (*float*) – Tolerance for lattice angles determined by crystal system.
  * **symprec** (*float*) – Tolerance for symmetry finding.
* **Returns**
  WyckoffPerturbations for derivation of symmetrically

```none
  unique sites. Used to derive the coordinates of the sites.
```

indices (list): Indices of symmetrically unique sites.
mapping (dict): A mapping dictionary that maps equivalent atoms

> onto each other.

lp (LatticePerturbation): LatticePerturbation for derivation of lattice

```none
  of the structure.
```

* **Return type**
  wps (list)

## maml.apps.bowsr.perturbation module

Module implements the perturbation class for atomic and lattice relaxation.

### *class* maml.apps.bowsr.perturbation.LatticePerturbation(spg_int_symbol: int, use_symmetry: bool = True)

Bases: `object`

Perturbation class for determining the standard lattice.

#### *property* abc(    *: list[float* )

Returns the lattice lengths.

#### *property* fit_lattice(    *: boo* )

Returns whether the lattice fits any crystal system.

#### *property* lattice(    *: Lattic* )

Returns the lattice.

#### sanity_check(lattice: Lattice, abc_tol: float = 0.001, angle_tol: float = 0.3)

Check whether the perturbation mode exists.

* **Parameters**
  * **lattice** (*Lattice*) – Lattice in Structure.
  * **abc_tol** (*float*) – Tolerance for lattice lengths determined by crystal system.
  * **angle_tol** (*float*) – Tolerance for lattice angles determined by crystal system.

### *class* maml.apps.bowsr.perturbation.WyckoffPerturbation(int_symbol: int, wyckoff_symbol: str, symmetry_ops: list[SymmOp] | None = None, use_symmetry: bool = True)

Bases: `object`

Perturbation class for determining the standard wyckoff position
and generating corresponding equivalent fractional coordinates.

#### *property* fit_site()

Returns whether the site fits any standard wyckoff position.

#### get_orbit(p: list | np.ndarray, tol: float = 0.001)

Returns the orbit for a point.

* **Parameters**
  * **p** (*list/numpy.array*) – Fractional coordinated point.
  * **tol** (*float*) – Tolerance for determining if sites are the same.

#### sanity_check(site: Site | PeriodicSite, wyc_tol: float = 0.0003)

Check whether the perturbation mode exists.

* **Parameters**
  * **site** (*PeriodicSite*) – PeriodicSite in Structure.
  * **wyc_tol** (*float*) – Tolerance for wyckoff symbol determined coordinates.

#### *property* site()

Returns the site.

#### standardize(p: list | np.ndarray, tol: float = 0.001)

Get the standardized position of p.

* **Parameters**
  * **p** (*list/numpy.array*) – Fractional coordinated point.
  * **tol** (*float*) – Tolerance for determining if sites are the same.

### maml.apps.bowsr.perturbation.crystal_system(int_number: int)

Method for crystal system determination.

* **Parameters**
  **int_number** (*int*) – International number of space group.

### maml.apps.bowsr.perturbation.get_standardized_structure(structure: Structure)

Get standardized structure.

* **Parameters**
  **structure** (*Structure*) – Pymatgen Structure object.

### maml.apps.bowsr.perturbation.perturbation_mapping(x, fixed_indices)

Perturbation mapping.

* **Parameters**
  * **x** –
  * **fixed_indices** –

Returns:

## maml.apps.bowsr.preprocessing module

Module implements the scaler.

### *class* maml.apps.bowsr.preprocessing.DummyScaler()

Bases: `MSONable`

Dummy scaler does nothing.

#### as_dict()

Serialize the instance into dictionary
Returns:

#### fit(target: list | np.ndarray)

Fit the DummyScaler to the target.

* **Parameters**
  **target** (*ndarray*) – The (mxn) ndarray. m is the number of samples,
  n is the number of feature dimensions.

#### *classmethod* from_dict(d)

Deserialize from a dictionary
:param d: Dict, dictionary contain class initialization parameters.

Returns:

#### inverse_transform(transformed_target: list | np.ndarray)

Inversely transform the target.

* **Parameters**
  **transformed_target** (*ndarray*) – The (mxn) ndarray. m is the number of samples,
  n is the number of feature dimensions.

#### transform(target: list | np.ndarray)

Transform target.

* **Parameters**
  **target** (*ndarray*) – The (mxn) ndarray. m is the number of samples,
  n is the number of feature dimensions.

### *class* maml.apps.bowsr.preprocessing.StandardScaler(mean: list | np.ndarray | None = None, std: list | np.ndarray | None = None)

Bases: `MSONable`

StandardScaler follows the sklean manner with addition of
dictionary representation.

#### as_dict()

Dict representation of StandardScaler.

#### fit(target: list | np.ndarray)

Fit the StandardScaler to the target.

* **Parameters**
  **target** (*ndarray*) – The (mxn) ndarray. m is the number of samples,
  n is the number of feature dimensions.

#### *classmethod* from_dict(d)

Reconstitute a StandardScaler object from a dict representation of
StandardScaler created using as_dict().

* **Parameters**
  **d** (*dict*) – Dict representation of StandardScaler.

#### inverse_transform(transformed_target: ndarray)

Inversely transform the target.

* **Parameters**
  **transformed_target** (*ndarray*) – The (mxn) ndarray. m is the number of samples,
  n is the number of feature dimensions.

#### transform(target: ndarray)

Transform target according to the mean and std.

* **Parameters**
  **target** (*ndarray*) – The (mxn) ndarray. m is the number of samples,
  n is the number of feature dimensions.

## maml.apps.bowsr.target_space module

Module implements the target space.

### *class* maml.apps.bowsr.target_space.TargetSpace(target_func: Callable, wps: list[WyckoffPerturbation], abc_dim: int, angles_dim: int, relax_coords: bool, relax_lattice: bool, scaler: StandardScaler | DummyScaler, random_state: RandomState)

Bases: `object`

Holds the perturbations of coordinates/lattice (x_wyckoff/x_lattice)
– formation energy (Y). Allows for constant-time appends while
ensuring no duplicates are added.

#### *property* bounds(    *: ndarra* )

Returns the search space of parameters.

#### lhs_sample(n_intervals: int)

Latin hypercube sampling.

* **Parameters**
  **n_intervals** (*int*) – Number of intervals.

#### *property* params(    *: ndarra* )

Returns the parameters in target space.

#### probe(x)

Evaluate a single point x, to obtain the value y and then records
them as observations.

* **Parameters**
  **x** (*ndarray*) – A single point.

#### register(x, target)

Append a point and its target value to the known data.

* **Parameters**
  * **x** (*ndarray*) – A single query point.
  * **target** (*float*) – Target value.

#### set_bounds(abc_bound: float = 1.2, angles_bound: float = 5, element_wise_wyckoff_bounds: dict | None = None)

Set the bound value of wyckoff perturbation and
lattice perturbation/volume perturbation.

#### set_empty()

Empty the param, target of the space.

#### *property* target(    *: ndarra* )

Returns the target (i.e., formation energy) in target space.

#### uniform_sample()

Creates random points within the bounds of the space.

### maml.apps.bowsr.target_space._hashable(x)

Ensure that an point is hashable by a python dict.