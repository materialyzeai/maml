---
layout: default
title: maml.apps.symbolic.md
nav_exclude: true
---

# maml.apps.symbolic package

Symbolic learning.

## *class* maml.apps.symbolic.AdaptiveLasso(lambd, gamma, \*\*kwargs)

Bases: `PenalizedLeastSquares`

Adaptive lasso regression using OLS coefficients as the root-n estimator coefficients.

### \_penalty_jac(x, y, beta)

### get_w(x, y)

Get adaptive weights from data.

* **Parameters**
  * **x** (*np.ndarray*) – MxN input data array
  * **y** (*np.ndarray*) – M output targets

Returns: coefficients array.

### penalty(beta: np.ndarray, x: np.ndarray | None = None, y: np.ndarray | None = None)

Calculate the penalty from input x, output y and coefficient beta.

* **Parameters**
  * **beta** (*np.ndarray*) – N coefficients
  * **x** (*np.ndarray*) – MxN input data array
  * **y** (*np.ndarray*) – M output targets

Returns: penalty value.

### select(x, y, options=None)

Select feature indices from x.

* **Parameters**
  * **x** (*np.ndarray*) – MxN input data array
  * **y** (*np.ndarray*) – M output targets
  * **options** (*dict*) – options in the optimizations provided
    to scipy.optimize.minimize

Returns: list of int indices.

## *class* maml.apps.symbolic.DantzigSelector(lambd, sigma=1.0, \*\*kwargs)

Bases: `BaseSelector`

Equation 11 in
[https://orfe.princeton.edu/~jqfan/papers/06/SIS.pdf](https://orfe.princeton.edu/~jqfan/papers/06/SIS.pdf)
and reference in [https://projecteuclid.org/download/pdfview_1/euclid.aos/1201012958](https://projecteuclid.org/download/pdfview_1/euclid.aos/1201012958).

### construct_constraints(x: np.ndarray, y: np.ndarray, beta: np.ndarray | None = None)

Get constraints dictionary from data, e.g.,
{“func”: lambda beta: fun(x, y, beta), “type”: “ineq”}.

* **Parameters**
  * **x** (*np.ndarray*) – MxN input data array
  * **y** (*np.ndarray*) – M output targets
  * **beta** (*np.ndarray*) – placeholder

Returns: dict of constraints.

### construct_jac(x: ndarray, y: ndarray)

Jacobian of cost functions.

* **Parameters**
  * **x** – ndarray
  * **y** – ndarray

Returns: callable

### construct_loss(x, y, beta)

Get loss function from data and tentative coefficients beta.

* **Parameters**
  * **x** (*np.ndarray*) – MxN input data array
  * **y** (*np.ndarray*) – M output targets
  * **beta** (*np.ndarray*) – N coefficients

Returns: loss value.

## *class* maml.apps.symbolic.FeatureGenerator(feature_df: pd.DataFrame, operators: list)

Bases: `object`

FeatureGenerator class for feature augmentation before selection.

### augment(n: int = 1)

Augment features
:param n: number of rounds of iteration.
:type n: int

Returns: augmented dataframe

## *class* maml.apps.symbolic.ISIS(sis: SIS | None = None, l0_regulate: bool = True)

Bases: `object`

Iterative SIS.

### evaluate(x: ndarray, y: ndarray, metric: str = ‘neg_mean_absolute_error’)

Evaluate the linear models using x, and y test data.

* **Parameters**
  * **x** (*np.ndarray*) – MxN input data array
  * **y** (*np.ndarray*) – M output targets
  * **metric** (*str*) – scorer function, used with
    sklearn.metrics.get_scorer

Returns: float.

### run(x: np.ndarray, y: np.ndarray, max_p: int = 10, metric: str = ‘neg_mean_absolute_error’, options: dict | None = None, step: float = 0.5)

Run the ISIS
:param x: input array
:type x: np.ndarray
:param y: target array
:type y: np.ndarray
:param max_p: Number of feature desired
:type max_p: int
:param metric: scorer function, used with

> sklearn.metrics.get_scorer
* **Parameters**
  * **options** –
  * **step** (*float*) – step to update gamma with.
* **Returns**
  np.array of index of selected features
  coeff(np.array): np.array of coeff of selected features
* **Return type**
  find_sel(np.array)

## *class* maml.apps.symbolic.L0BrutalForce(lambd: float, \*\*kwargs)

Bases: `BaseSelector`

Brutal force combinatorial screening of features.
This method takes all possible combinations of features
and optimize the following loss function

> 1/2 \* mean((y-x @ beta)\*\*2) + lambd \*
> ```default
> |
> ```
> beta|_0.

### select(x: np.ndarray, y: np.ndarray, options: dict | None = None, n_job: int = 1)

L0 combinatorial optimization.

* **Parameters**
  * **x** (*np.ndarray*) – design matrix
  * **y** (*np.ndarray*) – target vector
  * **options** – Dict of options.
  * **n_job** (*int*) – number of cpu

Returns:

## *class* maml.apps.symbolic.Lasso(lambd, \*\*kwargs)

Bases: `PenalizedLeastSquares`

Simple Lasso regression.

### \_penalty_jac(x, y, beta)

### penalty(beta: np.ndarray, x: np.ndarray | None = None, y: np.ndarray | None = None)

Calculate the penalty from input x, output y and coefficient beta.

* **Parameters**
  * **beta** (*np.ndarray*) – N coefficients
  * **x** (*np.ndarray*) – MxN input data array
  * **y** (*np.ndarray*) – M output targets

Returns: penalty value.

## *class* maml.apps.symbolic.Operator(operation: Callable[[…], Any], rep: str, unary: bool, commutative: bool)

Bases: `object`

Operator class. Wrap math operators with more attributes including check
is_unary, is_binary, and is_commutative, and generate name string
for the output.

### compute(i1: np.ndarray, i2: np.ndarray | None = None)

Compute the results
:param i1: first input array
:type i1: np.ndarray
:param i2: second input array (for binary operators).
:type i2: np.ndarray

Returns: array of computed results

### *classmethod* from_str(op_name: str)

Operator from name of the operator
:param op_name: string representation of the operator,
:type op_name: str
:param check Operator.support_op_rep for reference.:

Returns: Operator

### gen_name(f1: str, f2: str | None = None)

Generate string representation for output
:param f1: name of the first input array
:type f1: str
:param f2: name of the second input array.
:type f2: str

Returns: name of the output

### *property* is_binary(    *: boo* )

True if the operator takes two arguments else False.

* **Type**
  Returns

### *property* is_commutative(    *: boo* )

True if the operator is commutative else False.

* **Type**
  Returns

### *property* is_unary(    *: boo* )

True if the operator takes one argument else False.

* **Type**
  Returns

### support_op_rep(_ = [‘^2’, ‘^3’, ‘sqrt’, ‘abssqrt’, ‘cbrt’, ‘exp’, ‘abs’, ‘log10’, ‘abslog10’, ‘+’, ‘-’, ‘\*’, ‘/’, ‘|+|’, ‘|-|’, ‘sum_power_2’, ‘sum_exp’_ )

## *class* maml.apps.symbolic.SCAD(lambd: float | np.ndarray, a: float = 3.7, \*\*kwargs)

Bases: `PenalizedLeastSquares`

Smoothly clipped absolute deviation (SCAD),
equation 12 and 13 in [https://orfe.princeton.edu/~jqfan/papers/06/SIS.pdf](https://orfe.princeton.edu/~jqfan/papers/06/SIS.pdf).

### \_penalty_jac(x, y, beta)

### penalty(beta: np.ndarray, x: np.ndarray | None = None, y: np.ndarray | None = None)

Calculate the SCAD penalty from input x, output y

```none
and coefficient beta
```

* **Parameters**
  * **beta** (*np.ndarray*) – N coefficients
  * **x** (*np.ndarray*) – MxN input data array
  * **y** (*np.ndarray*) – M output targets

Returns: penalty value.

## *class* maml.apps.symbolic.SIS(gamma=0.1, selector: BaseSelector | None = None, verbose: bool = True)

Bases: `object`

Sure independence screening method.
The method consists of two steps:

> 1. Screen
> 2. Select.

### compute_residual(x, y)

Compute residual
:param x: input array
:type x: np.ndarray
:param y: target array.
:type y: np.ndarray

Returns: residual vector

### run(x, y, select_options=None)

Run the SIS with selector
:param x: MxN input data array
:type x: np.ndarray
:param y: M output targets
:type y: np.ndarray
:param select_options: options in the optimizations provided

> to scipy.optimize.minimize. If the selector is using cvxpy
> optimization package, this option is fed into cp.Problem.solve.

Returns: selected feature indices

### screen(x, y)

Simple screening method by comparing the correlation between features
and the target.

* **Parameters**
  * **x** (*np.ndarray*) – input array
  * **y** (*np.ndarray*) – target array

Returns: top indices

### select(x, y, options=None)

Select features using selectors
:param x: input array
:type x: np.ndarray
:param y: target array
:type y: np.ndarray
:param options: options for the optimization.
:type options: dict

### set_gamma(gamma)

Set gamma.

* **Parameters**
  **gamma** (*float*) – new gamma value

### set_selector(selector: BaseSelector)

Set new selector
:param selector: a feature selector.
:type selector: BaseSelector

### update_gamma(ratio: float = 0.5)

Update the sis object so that sis.select
return at least one feature.

* **Parameters**
  **ratio** (*float*) – ratio to update the parameters

## maml.apps.symbolic._feature_generator module

Feature Generator.

### *class* maml.apps.symbolic._feature_generator.FeatureGenerator(feature_df: pd.DataFrame, operators: list)

Bases: `object`

FeatureGenerator class for feature augmentation before selection.

#### augment(n: int = 1)

Augment features
:param n: number of rounds of iteration.
:type n: int

Returns: augmented dataframe

### *class* maml.apps.symbolic._feature_generator.Operator(operation: Callable[[…], Any], rep: str, unary: bool, commutative: bool)

Bases: `object`

Operator class. Wrap math operators with more attributes including check
is_unary, is_binary, and is_commutative, and generate name string
for the output.

#### compute(i1: np.ndarray, i2: np.ndarray | None = None)

Compute the results
:param i1: first input array
:type i1: np.ndarray
:param i2: second input array (for binary operators).
:type i2: np.ndarray

Returns: array of computed results

#### *classmethod* from_str(op_name: str)

Operator from name of the operator
:param op_name: string representation of the operator,
:type op_name: str
:param check Operator.support_op_rep for reference.:

Returns: Operator

#### gen_name(f1: str, f2: str | None = None)

Generate string representation for output
:param f1: name of the first input array
:type f1: str
:param f2: name of the second input array.
:type f2: str

Returns: name of the output

#### *property* is_binary(    *: boo* )

True if the operator takes two arguments else False.

* **Type**
  Returns

#### *property* is_commutative(    *: boo* )

True if the operator is commutative else False.

* **Type**
  Returns

#### *property* is_unary(    *: boo* )

True if the operator takes one argument else False.

* **Type**
  Returns

#### support_op_rep(_ = [‘^2’, ‘^3’, ‘sqrt’, ‘abssqrt’, ‘cbrt’, ‘exp’, ‘abs’, ‘log10’, ‘abslog10’, ‘+’, ‘-’, ‘\*’, ‘/’, ‘|+|’, ‘|-|’, ‘sum_power_2’, ‘sum_exp’_ )

### maml.apps.symbolic._feature_generator._my_abs_diff(x, y)

### maml.apps.symbolic._feature_generator._my_abs_log10(x)

### maml.apps.symbolic._feature_generator._my_abs_sqrt(x)

### maml.apps.symbolic._feature_generator._my_abs_sum(x, y)

### maml.apps.symbolic._feature_generator._my_diff(x, y)

### maml.apps.symbolic._feature_generator._my_div(x, y)

### maml.apps.symbolic._feature_generator._my_exp(x)

### maml.apps.symbolic._feature_generator._my_exp_power_2(x)

### maml.apps.symbolic._feature_generator._my_exp_power_3(x)

### maml.apps.symbolic._feature_generator._my_mul(x, y)

### maml.apps.symbolic._feature_generator._my_power(x: float, n: int)

### maml.apps.symbolic._feature_generator._my_sum(x, y)

### maml.apps.symbolic._feature_generator._my_sum_exp(x, y)

### maml.apps.symbolic._feature_generator._my_sum_power_2(x, y)

### maml.apps.symbolic._feature_generator._my_sum_power_3(x, y)

### maml.apps.symbolic._feature_generator._update_df(df, op, fn1, fn2=None)

Helper function to update the dataframe with new generated feature array.

### maml.apps.symbolic._feature_generator.generate_feature(feature_df: pd.DataFrame, operators: list)

Generate new features by applying operators to columns in feature_df.

* **Parameters**
  * **feature_df** (*pd.DataFrame*) – dataframe of original features
  * **operators** (*list*) – list of str of operators (check Operator.support_op_rep for reference)

Returns: dataframe of augmented features

## maml.apps.symbolic._selectors module

Selectors.

### *class* maml.apps.symbolic._selectors.AdaptiveLasso(lambd, gamma, \*\*kwargs)

Bases: `PenalizedLeastSquares`

Adaptive lasso regression using OLS coefficients as the root-n estimator coefficients.

#### \_penalty_jac(x, y, beta)

#### get_w(x, y)

Get adaptive weights from data.

* **Parameters**
  * **x** (*np.ndarray*) – MxN input data array
  * **y** (*np.ndarray*) – M output targets

Returns: coefficients array.

#### penalty(beta: np.ndarray, x: np.ndarray | None = None, y: np.ndarray | None = None)

Calculate the penalty from input x, output y and coefficient beta.

* **Parameters**
  * **beta** (*np.ndarray*) – N coefficients
  * **x** (*np.ndarray*) – MxN input data array
  * **y** (*np.ndarray*) – M output targets

Returns: penalty value.

#### select(x, y, options=None)

Select feature indices from x.

* **Parameters**
  * **x** (*np.ndarray*) – MxN input data array
  * **y** (*np.ndarray*) – M output targets
  * **options** (*dict*) – options in the optimizations provided
    to scipy.optimize.minimize

Returns: list of int indices.

### *class* maml.apps.symbolic._selectors.BaseSelector(coef_thres: float = 1e-06, method: str = ‘SLSQP’)

Bases: `object`

Feature selector. This is meant to work on relatively smaller
number of features.

#### *classmethod* \_get_param_names()

#### compute_residual(x: ndarray, y: ndarray)

Compute.

* **Parameters**
  * **x** (*np.ndarray*) – design matrix
  * **y** (*np.ndarray*) – target vector

Returns: residual vector.

#### construct_constraints(x: np.ndarray, y: np.ndarray, beta: np.ndarray | None = None)

Get constraints dictionary from data, e.g.,
{“func”: lambda beta: fun(x, y, beta), “type”: “ineq”}.

* **Parameters**
  * **x** (*np.ndarray*) – MxN input data array
  * **y** (*np.ndarray*) – M output targets
  * **beta** (*np.ndarray*) – parameter to optimize

Returns: dict of constraints.

#### construct_jac(x: np.ndarray, y: np.ndarray)

Jacobian of cost function
:param x: MxN input data array
:type x: np.ndarray
:param y: M output targets
:type y: np.ndarray

Returns: Jacobian function.

#### construct_loss(x: ndarray, y: ndarray, beta: ndarray)

Get loss function from data and tentative coefficients beta
:param x: MxN input data array
:type x: np.ndarray
:param y: M output targets
:type y: np.ndarray
:param beta: N coefficients
:type beta: np.ndarray

Returns: loss value.

#### evaluate(x: ndarray, y: ndarray, metric: str = ‘neg_mean_absolute_error’)

Evaluate the linear models using x, and y test data.

* **Parameters**
  * **x** (*np.ndarray*) – MxN input data array
  * **y** (*np.ndarray*) – M output targets
  * **metric** (*str*) – scorer function, used with
    sklearn.metrics.get_scorer

Returns:

#### get_coef()

Get coefficients
Returns: the coefficients array.

#### get_feature_indices()

Get selected feature indices.

Returns: ndarray

#### get_params()

Get params for this selector.

Returns: mapping of string to any

```none
parameter names mapped to their values
```

#### predict(x: ndarray)

Predict the results using sparsified coefficients.

* **Parameters**
  **x** (*np.ndarray*) – design matrix

Returns: ndarray

#### select(x: np.ndarray, y: np.ndarray, options: dict | None = None)

Select feature indices from x
:param x: MxN input data array
:type x: np.ndarray
:param y: M output targets
:type y: np.ndarray
:param options: options in the optimizations provided

> to scipy.optimize.minimize

Returns: list of int indices.

#### set_params(\*\*params)

Set the parameters of this selector
:param \*\*params: dict
:param Selector parameters.:

* **Returns**
  selector instance
* **Return type**
  self

### *class* maml.apps.symbolic._selectors.DantzigSelector(lambd, sigma=1.0, \*\*kwargs)

Bases: `BaseSelector`

Equation 11 in
[https://orfe.princeton.edu/~jqfan/papers/06/SIS.pdf](https://orfe.princeton.edu/~jqfan/papers/06/SIS.pdf)
and reference in [https://projecteuclid.org/download/pdfview_1/euclid.aos/1201012958](https://projecteuclid.org/download/pdfview_1/euclid.aos/1201012958).

#### construct_constraints(x: np.ndarray, y: np.ndarray, beta: np.ndarray | None = None)

Get constraints dictionary from data, e.g.,
{“func”: lambda beta: fun(x, y, beta), “type”: “ineq”}.

* **Parameters**
  * **x** (*np.ndarray*) – MxN input data array
  * **y** (*np.ndarray*) – M output targets
  * **beta** (*np.ndarray*) – placeholder

Returns: dict of constraints.

#### construct_jac(x: ndarray, y: ndarray)

Jacobian of cost functions.

* **Parameters**
  * **x** – ndarray
  * **y** – ndarray

Returns: callable

#### construct_loss(x, y, beta)

Get loss function from data and tentative coefficients beta.

* **Parameters**
  * **x** (*np.ndarray*) – MxN input data array
  * **y** (*np.ndarray*) – M output targets
  * **beta** (*np.ndarray*) – N coefficients

Returns: loss value.

### *class* maml.apps.symbolic._selectors.L0BrutalForce(lambd: float, \*\*kwargs)

Bases: `BaseSelector`

Brutal force combinatorial screening of features.
This method takes all possible combinations of features
and optimize the following loss function

> 1/2 \* mean((y-x @ beta)\*\*2) + lambd \*
> ```default
> |
> ```
> beta|_0.

#### select(x: np.ndarray, y: np.ndarray, options: dict | None = None, n_job: int = 1)

L0 combinatorial optimization.

* **Parameters**
  * **x** (*np.ndarray*) – design matrix
  * **y** (*np.ndarray*) – target vector
  * **options** – Dict of options.
  * **n_job** (*int*) – number of cpu

Returns:

### *class* maml.apps.symbolic._selectors.Lasso(lambd, \*\*kwargs)

Bases: `PenalizedLeastSquares`

Simple Lasso regression.

#### \_penalty_jac(x, y, beta)

#### penalty(beta: np.ndarray, x: np.ndarray | None = None, y: np.ndarray | None = None)

Calculate the penalty from input x, output y and coefficient beta.

* **Parameters**
  * **beta** (*np.ndarray*) – N coefficients
  * **x** (*np.ndarray*) – MxN input data array
  * **y** (*np.ndarray*) – M output targets

Returns: penalty value.

### *class* maml.apps.symbolic._selectors.PenalizedLeastSquares(coef_thres: float = 1e-06, method: str = ‘SLSQP’)

Bases: `BaseSelector`

Penalized least squares. In addition to minimizing the sum of squares loss,
it adds an additional penalty to the coefficients.

#### \_penalty_jac(x, y, beta)

#### \_sse_jac(x, y, beta)

#### construct_constraints(x: np.ndarray, y: np.ndarray, beta: np.ndarray | None = None)

No constraints
:param x: MxN input data array
:type x: np.ndarray
:param y: M output targets
:type y: np.ndarray
:param beta: placeholder only
:type beta: np.ndarray

Returns: a list of dictionary constraints.

#### construct_jac(x: ndarray, y: ndarray)

Construct the jacobian of loss function
:param x: MxN input data array
:type x: np.ndarray
:param y: M output targets
:type y: np.ndarray

Returns: jacobian vector.

#### construct_loss(x: ndarray, y: ndarray, beta: ndarray)

Construct the loss function. An extra penalty term is added
:param x: MxN input data array
:type x: np.ndarray
:param y: M output targets
:type y: np.ndarray
:param beta: N coefficients
:type beta: np.ndarray

Returns: sum of errors.

#### penalty(beta: np.ndarray, x: np.ndarray | None = None, y: np.ndarray | None = None)

Calculate the penalty from input x, output y and coefficient beta
:param x: MxN input data array
:type x: np.ndarray
:param y: M output targets
:type y: np.ndarray
:param beta: N coefficients
:type beta: np.ndarray

Returns: penalty value.

### *class* maml.apps.symbolic._selectors.SCAD(lambd: float | np.ndarray, a: float = 3.7, \*\*kwargs)

Bases: `PenalizedLeastSquares`

Smoothly clipped absolute deviation (SCAD),
equation 12 and 13 in [https://orfe.princeton.edu/~jqfan/papers/06/SIS.pdf](https://orfe.princeton.edu/~jqfan/papers/06/SIS.pdf).

#### \_penalty_jac(x, y, beta)

#### penalty(beta: np.ndarray, x: np.ndarray | None = None, y: np.ndarray | None = None)

Calculate the SCAD penalty from input x, output y

```none
and coefficient beta
```

* **Parameters**
  * **beta** (*np.ndarray*) – N coefficients
  * **x** (*np.ndarray*) – MxN input data array
  * **y** (*np.ndarray*) – M output targets

Returns: penalty value.

## maml.apps.symbolic._selectors_cvxpy module

This module implements more robust optimization
using the cvxpy package.

### *class* maml.apps.symbolic._selectors_cvxpy.AdaptiveLassoCP(lambd, gamma, \*\*kwargs)

Bases: `PenalizedLeastSquaresCP`

Adaptive lasso regression using OLS coefficients
as the root-n estimator coefficients.

#### get_w(x: ndarray, y: ndarray)

Get adaptive weights from data
:param x: MxN input data array
:type x: np.ndarray
:param y: M output targets
:type y: np.ndarray

Returns: coefficients array.

#### penalty(beta: cp.Variable, x: np.ndarray | None = None, y: np.ndarray | None = None)

Calculate the penalty from input x, output y and coefficient beta
:param x: MxN input data array
:type x: np.ndarray
:param y: M output targets
:type y: np.ndarray
:param beta: N coefficients
:type beta: np.ndarray

Returns: penalty value.

#### select(x: np.ndarray, y: np.ndarray, options: dict | None = None)

Select feature indices from x
:param x: MxN input data array
:type x: np.ndarray
:param y: M output targets
:type y: np.ndarray
:param options: options in the cp.Problem.solve
:type options: dict

Returns: array int indices.

### *class* maml.apps.symbolic._selectors_cvxpy.BaseSelectorCP(coef_thres: float = 1e-06, method: str = ‘ECOS’)

Bases: `BaseSelector`

Base selector using cvxpy (CP).

#### construct_constraints(x: np.ndarray, y: np.ndarray, beta: cp.Variable | None = None)

Get constraints dictionary from data, e.g.,
{“func”: lambda beta: fun(x, y, beta), “type”: “ineq”}.

* **Parameters**
  * **x** (*np.ndarray*) – MxN input data array
  * **y** (*np.ndarray*) – M output targets
  * **beta** – (np.ndarray): target variable for optimization

Returns: dict of constraints.

#### construct_loss(x: np.ndarray, y: np.ndarray, beta: cp.Variable)

Get loss function from data and tentative coefficients beta
:param x: MxN input data array
:type x: np.ndarray
:param y: M output targets
:type y: np.ndarray
:param beta: N coefficients
:type beta: np.ndarray

Returns: loss value.

#### select(x: np.ndarray, y: np.ndarray, options: dict | None = None)

Select feature indices from x
:param x: MxN input data array
:type x: np.ndarray
:param y: M output targets
:type y: np.ndarray
:param options: kwargs for cp.Problem.solve
:type options: dict

Returns: list of int indices.

### *class* maml.apps.symbolic._selectors_cvxpy.DantzigSelectorCP(lambd, sigma=1.0, \*\*kwargs)

Bases: `BaseSelectorCP`

Equation 11 in
[https://orfe.princeton.edu/~jqfan/papers/06/SIS.pdf](https://orfe.princeton.edu/~jqfan/papers/06/SIS.pdf)
and reference in [https://projecteuclid.org/download/pdfview_1/euclid.aos/1201012958](https://projecteuclid.org/download/pdfview_1/euclid.aos/1201012958).

#### construct_constraints(x: np.ndarray, y: np.ndarray, beta: cp.Variable | None = None)

Dantzig selector constraints
:param x: MxN input data array
:type x: np.ndarray
:param y: M output targets
:type y: np.ndarray
:param beta: dimension N vector for optimization
:type beta: cp.Variable

Returns: List of constraints.

#### construct_loss(x: np.ndarray, y: np.ndarray, beta: cp.Variable)

L1 loss
:param x: MxN input data array
:type x: np.ndarray
:param y: M output targets
:type y: np.ndarray
:param beta: dimension N vector for optimization
:type beta: cp.Variable

Returns: loss expression.

### *class* maml.apps.symbolic._selectors_cvxpy.LassoCP(lambd, \*\*kwargs)

Bases: `PenalizedLeastSquaresCP`

Simple Lasso regression.

#### penalty(beta: cp.Variable, x: np.ndarray | None = None, y: np.ndarray | None = None)

Calculate the penalty from input x, output y and coefficient beta
:param x: MxN input data array
:type x: np.ndarray
:param y: M output targets
:type y: np.ndarray
:param beta: N coefficients
:type beta: np.ndarray

Returns: penalty value.

### *class* maml.apps.symbolic._selectors_cvxpy.PenalizedLeastSquaresCP(coef_thres: float = 1e-06, method: str = ‘ECOS’)

Bases: `BaseSelectorCP`

Penalized least squares. In addition to minimizing the sum of squares loss,
it adds an additional penalty to the coefficients.

#### construct_loss(x: np.ndarray, y: np.ndarray, beta: cp.Variable)

L1 loss
:param x: MxN input data array
:type x: np.ndarray
:param y: M output targets
:type y: np.ndarray
:param beta: dimension N vector for optimization
:type beta: cp.Variable

Returns: loss expression.

#### penalty(beta: cp.Variable, x: np.ndarray | None = None, y: np.ndarray | None = None)

Calculate the penalty from input x, output y and coefficient beta
:param x: MxN input data array
:type x: np.ndarray
:param y: M output targets
:type y: np.ndarray
:param beta: N coefficients
:type beta: np.ndarray

Returns: penalty value.

## maml.apps.symbolic._sis module

Sure Independence Screening.

[https://orfe.princeton.edu/~jqfan/papers/06/SIS.pdf](https://orfe.princeton.edu/~jqfan/papers/06/SIS.pdf)

### *class* maml.apps.symbolic._sis.ISIS(sis: SIS | None = None, l0_regulate: bool = True)

Bases: `object`

Iterative SIS.

#### evaluate(x: ndarray, y: ndarray, metric: str = ‘neg_mean_absolute_error’)

Evaluate the linear models using x, and y test data.

* **Parameters**
  * **x** (*np.ndarray*) – MxN input data array
  * **y** (*np.ndarray*) – M output targets
  * **metric** (*str*) – scorer function, used with
    sklearn.metrics.get_scorer

Returns: float.

#### run(x: np.ndarray, y: np.ndarray, max_p: int = 10, metric: str = ‘neg_mean_absolute_error’, options: dict | None = None, step: float = 0.5)

Run the ISIS
:param x: input array
:type x: np.ndarray
:param y: target array
:type y: np.ndarray
:param max_p: Number of feature desired
:type max_p: int
:param metric: scorer function, used with

> sklearn.metrics.get_scorer
* **Parameters**
  * **options** –
  * **step** (*float*) – step to update gamma with.
* **Returns**
  np.array of index of selected features
  coeff(np.array): np.array of coeff of selected features
* **Return type**
  find_sel(np.array)

### *class* maml.apps.symbolic._sis.SIS(gamma=0.1, selector: BaseSelector | None = None, verbose: bool = True)

Bases: `object`

Sure independence screening method.
The method consists of two steps:

> 1. Screen
> 2. Select.

#### compute_residual(x, y)

Compute residual
:param x: input array
:type x: np.ndarray
:param y: target array.
:type y: np.ndarray

Returns: residual vector

#### run(x, y, select_options=None)

Run the SIS with selector
:param x: MxN input data array
:type x: np.ndarray
:param y: M output targets
:type y: np.ndarray
:param select_options: options in the optimizations provided

> to scipy.optimize.minimize. If the selector is using cvxpy
> optimization package, this option is fed into cp.Problem.solve.

Returns: selected feature indices

#### screen(x, y)

Simple screening method by comparing the correlation between features
and the target.

* **Parameters**
  * **x** (*np.ndarray*) – input array
  * **y** (*np.ndarray*) – target array

Returns: top indices

#### select(x, y, options=None)

Select features using selectors
:param x: input array
:type x: np.ndarray
:param y: target array
:type y: np.ndarray
:param options: options for the optimization.
:type options: dict

#### set_gamma(gamma)

Set gamma.

* **Parameters**
  **gamma** (*float*) – new gamma value

#### set_selector(selector: BaseSelector)

Set new selector
:param selector: a feature selector.
:type selector: BaseSelector

#### update_gamma(ratio: float = 0.5)

Update the sis object so that sis.select
return at least one feature.

* **Parameters**
  **ratio** (*float*) – ratio to update the parameters

### maml.apps.symbolic._sis._best_combination(x, y, find_sel, find_sel_new, metric: str = ‘neg_mean_absolute_error’)

### maml.apps.symbolic._sis._eval(x, y, coeff, metric)

### maml.apps.symbolic._sis._get_coeff(x, y)