---
layout: default
title: maml.base.md
nav_exclude: true
---

# maml.base package

Define abstract base classes.

## *class* maml.base.BaseDataSource()

Bases: `object`

Abstract base class for a data source.

### *abstract* get(\*args, \*\*kwargs)

Get data from sources.

## *class* maml.base.BaseDescriber(\*\*kwargs)

Bases: `BaseEstimator`, `TransformerMixin`, `MSONable`

Base class for a Describer. A describers converts an object to a descriptor,
typically a numerical representation useful for machine learning.
The output for the describers can be a single DataFrame/numpy.ndarray or
a list of DataFrame/numpy.ndarray.

### *abc_impl(* = <_abc.*abc_data object* )

### \_is_multi_output()

### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

### clear_cache()

Clear cache.

### *property* feature_dim()

Feature dimension, useful when certain models need to specify
the feature dimension, e.g., MLP models.

### fit(x: Any, y: Any | None = None)

Place holder for fit API.

* **Parameters**
  * **x** – Any inputs
  * **y** – Any outputs

Returns: self

### transform(objs: list[Any])

Transform a list of objs. If the return data is DataFrame,
use df.xs(index, level=’input_index’) to get the result for the i-th object.

* **Parameters**
  **objs** (*list*) – A list of objects.
* **Returns**
  One or a list of pandas data frame/numpy ndarray

### transform_one(obj: Any)

Transform an object.

## *class* maml.base.BaseModel(model, describer: BaseDescriber | None = None, \*\*kwargs)

Bases: `object`

Abstract Base class for a Model. Basically, it usually wraps around a deep
learning package, e.g., the Sequential Model in Keras, but provides for
transparent conversion of arbitrary input and outputs.

### \_predict(features: ndarray, \*\*kwargs)

Predict the values given a set of inputs based on fitted models.

* **Parameters**
  **features** (*np.ndarray*) – array-like input features.
* **Returns**
  List of output objects.

### fit(features: list | np.ndarray, targets: list | np.ndarray | None = None, val_features: list | np.ndarray | None = None, val_targets: list | np.ndarray | None = None, \*\*kwargs)

* **Parameters**
  * **features** (*list*\* or \**np.ndarray*) – Numerical input feature list or
    numpy array with dim (m, n) where m is the number of data and
    n is the feature dimension.
  * **targets** (*list*\* or \**np.ndarray*) – Numerical output target list, or
    numpy array with dim (m, ).
  * **val_features** (*list*\* or \**np.ndarray*) – validation features
  * **val_targets** (*list*\* or \**np.ndarray*) – validation targets.
* **Returns**
  self

### predict_objs(objs: list | np.ndarray)

Predict the values given a set of objects. Usually Pymatgen

```none
Structure objects.
```

### train(objs: list | np.ndarray, targets: list | np.ndarray | None = None, val_objs: list | np.ndarray | None = None, val_targets: list | np.ndarray | None = None, \*\*kwargs)

Train the models from object, target pairs.

* **Parameters**
  * **objs** (*list*\* of \**objects*) – List of objects
  * **targets** (*list*) – list of float or np.ndarray
  * **val_objs** (*list*\* of \**objects*) – list of validation objects
  * **val_targets** (*list*) – list of validation targets
  * **\*\*kwargs** –

Returns: self

## *class* maml.base.DummyDescriber(\*\*kwargs)

Bases: `BaseDescriber`

Dummy Describer that does nothing.

### *abc_impl(* = <_abc.*abc_data object* )

### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

### transform_one(obj: Any)

Does nothing but return the original features.

* **Parameters**
  **obj** – Any inputs

Returns: Any outputs

## *class* maml.base.KerasModel(model, describer: BaseDescriber | None = None, \*\*kwargs)

Bases: `BaseModel`, `KerasMixin`

MAML models with keras models as estimators.

### *static* \_get_validation_data(val_features, val_targets, \*\*val_kwargs)

construct validation data, the default is just returning a list of
val_features and val_targets.

### fit(features: list | np.ndarray, targets: list | np.ndarray | None = None, val_features: list | np.ndarray | None = None, val_targets: list | np.ndarray | None = None, \*\*kwargs)

* **Parameters**
  * **features** (*list*\* or \**np.ndarray*) – Numerical input feature list or
    numpy array with dim (m, n) where m is the number of data and
    n is the feature dimension.
  * **targets** (*list*\* or \**np.ndarray*) – Numerical output target list, or
    numpy array with dim (m, ).
  * **val_features** (*list*\* or \**np.ndarray*) – validation features
  * **val_targets** (*list*\* or \**np.ndarray*) – validation targets.
* **Returns**
  self

## *class* maml.base.SKLModel(model, describer: BaseDescriber | None = None, \*\*kwargs)

Bases: `BaseModel`, `SklearnMixin`

MAML models with sklearn models as estimator.

## *class* maml.base.SequentialDescriber(describers: list, \*\*kwargs)

Bases: `Pipeline`

A thin wrapper of sklearn Pipeline.

### *abc_impl(* = <_abc.*abc_data object* )

### steps(    *: List[Any* )

## *class* maml.base.TargetScalerMixin(\*args, \*\*kwargs)

Bases: `object`

Mixin class for models with target scaler.

### predict_objs(objs: list | np.ndarray)

Predict the values given a set of objects. Usually Pymatgen

```none
Structure objects.
```

### train(objs: list | np.ndarray, targets: list | np.ndarray | None = None, val_objs: list | np.ndarray | None = None, val_targets: list | np.ndarray | None = None, \*\*kwargs)

Train the models from object, target pairs.

* **Parameters**
  * **objs** (*list*\* of \**objects*) – List of objects
  * **targets** (*list*) – list of float or np.ndarray
  * **val_objs** (*list*\* of \**objects*) – list of validation objects
  * **val_targets** (*list*) – list of validation targets
  * **\*\*kwargs** –

Returns: self

## maml.base.describer_type(dtype: str)

Decorate to set describers class type.

* **Parameters**
  **dtype** (*str*) – describers type, e.g., site, composition, structure etc.
* **Returns**
  wrapped class

## maml.base.get_feature_batch(fb_name: str | Callable | None = None)

Providing a feature batch name, returning the function callable
:param fb_name: name of the feature batch function
:type fb_name: str

Returns: callable feature batch function.

## maml.base.is_keras_model(model: BaseModel)

Check whether the model is keras
:param model: model
:type model: BaseModel

Returns: bool.

## maml.base.is_sklearn_model(model: BaseModel)

Check whether the model is sklearn
:param model: model
:type model: BaseModel

Returns: bool.

## maml.base._data module

MAML data base classes.

### *class* maml.base._data.BaseDataSource()

Bases: `object`

Abstract base class for a data source.

#### *abstract* get(\*args, \*\*kwargs)

Get data from sources.

## maml.base._describer module

MAML describers base classes.

### *class* maml.base._describer.BaseDescriber(\*\*kwargs)

Bases: `BaseEstimator`, `TransformerMixin`, `MSONable`

Base class for a Describer. A describers converts an object to a descriptor,
typically a numerical representation useful for machine learning.
The output for the describers can be a single DataFrame/numpy.ndarray or
a list of DataFrame/numpy.ndarray.

#### *abc_impl(* = <_abc.*abc_data object* )

#### \_is_multi_output()

#### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

#### clear_cache()

Clear cache.

#### *property* feature_dim()

Feature dimension, useful when certain models need to specify
the feature dimension, e.g., MLP models.

#### fit(x: Any, y: Any | None = None)

Place holder for fit API.

* **Parameters**
  * **x** – Any inputs
  * **y** – Any outputs

Returns: self

#### transform(objs: list[Any])

Transform a list of objs. If the return data is DataFrame,
use df.xs(index, level=’input_index’) to get the result for the i-th object.

* **Parameters**
  **objs** (*list*) – A list of objects.
* **Returns**
  One or a list of pandas data frame/numpy ndarray

#### transform_one(obj: Any)

Transform an object.

### *class* maml.base._describer.DummyDescriber(\*\*kwargs)

Bases: `BaseDescriber`

Dummy Describer that does nothing.

#### *abc_impl(* = <_abc.*abc_data object* )

#### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

#### transform_one(obj: Any)

Does nothing but return the original features.

* **Parameters**
  **obj** – Any inputs

Returns: Any outputs

### *class* maml.base._describer.SequentialDescriber(describers: list, \*\*kwargs)

Bases: `Pipeline`

A thin wrapper of sklearn Pipeline.

#### *abc_impl(* = <_abc.*abc_data object* )

#### steps(    *: List[Any* )

### maml.base._describer._transform_one(describer: BaseDescriber, obj: Any)

A wrapper to make a pure function.

* **Parameters**
  **describer** (*BaseDescriber*) – a describers
* **Returns**
  np.ndarray

### maml.base._describer.describer_type(dtype: str)

Decorate to set describers class type.

* **Parameters**
  **dtype** (*str*) – describers type, e.g., site, composition, structure etc.
* **Returns**
  wrapped class

## maml.base._feature_batch module

Batch a list of features output by describers.transform method.

### maml.base._feature_batch.get_feature_batch(fb_name: str | Callable | None = None)

Providing a feature batch name, returning the function callable
:param fb_name: name of the feature batch function
:type fb_name: str

Returns: callable feature batch function.

### maml.base._feature_batch.no_action(features: list[Any])

Return original feature lists.

### maml.base._feature_batch.pandas_concat(features: list[pandas.core.frame.DataFrame])

Concatenate a list of pandas dataframe into a single one
:param features: list of pandas dataframe.
:type features: list

Returns: concatenated pandas dataframe

### maml.base._feature_batch.stack_first_dim(features: list[numpy.ndarray])

Stack the first dimension. If the original features
are a list of nxm array, the stacked features will be lxnxm,
where l is the number of entries in the list
:param features: list of numpy array features.
:type features: list

Returns: stacked features

### maml.base._feature_batch.stack_padded(features: list[numpy.ndarray])

Stack the first dimension. If the original features
are a list of nxm array, the stacked features will be lxnxm,
where l is the number of entries in the list
:param features: list of numpy array features.
:type features: list

Returns: stacked features

## maml.base._mixin module

Model mixins.

### *class* maml.base._mixin.TargetScalerMixin(\*args, \*\*kwargs)

Bases: `object`

Mixin class for models with target scaler.

#### predict_objs(objs: list | np.ndarray)

Predict the values given a set of objects. Usually Pymatgen

```none
Structure objects.
```

#### train(objs: list | np.ndarray, targets: list | np.ndarray | None = None, val_objs: list | np.ndarray | None = None, val_targets: list | np.ndarray | None = None, \*\*kwargs)

Train the models from object, target pairs.

* **Parameters**
  * **objs** (*list*\* of \**objects*) – List of objects
  * **targets** (*list*) – list of float or np.ndarray
  * **val_objs** (*list*\* of \**objects*) – list of validation objects
  * **val_targets** (*list*) – list of validation targets
  * **\*\*kwargs** –

Returns: self

## maml.base._model module

MAML models base classes.

### *class* maml.base._model.BaseModel(model, describer: BaseDescriber | None = None, \*\*kwargs)

Bases: `object`

Abstract Base class for a Model. Basically, it usually wraps around a deep
learning package, e.g., the Sequential Model in Keras, but provides for
transparent conversion of arbitrary input and outputs.

#### \_predict(features: ndarray, \*\*kwargs)

Predict the values given a set of inputs based on fitted models.

* **Parameters**
  **features** (*np.ndarray*) – array-like input features.
* **Returns**
  List of output objects.

#### fit(features: list | np.ndarray, targets: list | np.ndarray | None = None, val_features: list | np.ndarray | None = None, val_targets: list | np.ndarray | None = None, \*\*kwargs)

* **Parameters**
  * **features** (*list*\* or \**np.ndarray*) – Numerical input feature list or
    numpy array with dim (m, n) where m is the number of data and
    n is the feature dimension.
  * **targets** (*list*\* or \**np.ndarray*) – Numerical output target list, or
    numpy array with dim (m, ).
  * **val_features** (*list*\* or \**np.ndarray*) – validation features
  * **val_targets** (*list*\* or \**np.ndarray*) – validation targets.
* **Returns**
  self

#### predict_objs(objs: list | np.ndarray)

Predict the values given a set of objects. Usually Pymatgen

```none
Structure objects.
```

#### train(objs: list | np.ndarray, targets: list | np.ndarray | None = None, val_objs: list | np.ndarray | None = None, val_targets: list | np.ndarray | None = None, \*\*kwargs)

Train the models from object, target pairs.

* **Parameters**
  * **objs** (*list*\* of \**objects*) – List of objects
  * **targets** (*list*) – list of float or np.ndarray
  * **val_objs** (*list*\* of \**objects*) – list of validation objects
  * **val_targets** (*list*) – list of validation targets
  * **\*\*kwargs** –

Returns: self

### *class* maml.base._model.KerasMixin()

Bases: `object`

keras models mixin with save and load functionality.

#### evaluate(eval_objs: list | np.ndarray, eval_targets: list | np.ndarray, is_feature: bool = False)

Evaluate objs, targets.

* **Parameters**
  * **eval_objs** (*list*) – objs for evaluation
  * **eval_targets** (*list*) – target list for the corresponding objects
  * **is_feature** (*bool*) – whether the input x is feature matrix
  * **metric** (*callable*) – metric for evaluation

#### *classmethod* from_file(filename: str, \*\*kwargs)

Load the models from file
:param filename: filename
:type filename: str
:param \*\*kwargs:

Returns: object instance

#### *static* get_input_dim(describer: BaseDescriber | None = None, input_dim: int | None = None)

Get feature dimension/input_dim from describers or input_dim.

* **Parameters**
  * **describer** (*Describer*) – describers
  * **input_dim** (*int*) – optional input dim int

#### load(filename: str, custom_objects: list | None = None)

Load models parameters from filename
:param filename: models file name.
:type filename: str

Returns: None

#### save(filename: str)

Save the models and describers.

* **Parameters**
  **filename** (*str*) – filename for save

### *class* maml.base._model.KerasModel(model, describer: BaseDescriber | None = None, \*\*kwargs)

Bases: `BaseModel`, `KerasMixin`

MAML models with keras models as estimators.

#### *static* \_get_validation_data(val_features, val_targets, \*\*val_kwargs)

construct validation data, the default is just returning a list of
val_features and val_targets.

#### fit(features: list | np.ndarray, targets: list | np.ndarray | None = None, val_features: list | np.ndarray | None = None, val_targets: list | np.ndarray | None = None, \*\*kwargs)

* **Parameters**
  * **features** (*list*\* or \**np.ndarray*) – Numerical input feature list or
    numpy array with dim (m, n) where m is the number of data and
    n is the feature dimension.
  * **targets** (*list*\* or \**np.ndarray*) – Numerical output target list, or
    numpy array with dim (m, ).
  * **val_features** (*list*\* or \**np.ndarray*) – validation features
  * **val_targets** (*list*\* or \**np.ndarray*) – validation targets.
* **Returns**
  self

### *class* maml.base._model.SKLModel(model, describer: BaseDescriber | None = None, \*\*kwargs)

Bases: `BaseModel`, `SklearnMixin`

MAML models with sklearn models as estimator.

### *class* maml.base._model.SklearnMixin()

Bases: `object`

Sklearn models save and load functionality.

#### evaluate(eval_objs: list | np.ndarray, eval_targets: list | np.ndarray, is_feature: bool = False, metric: str | Callable | None = None)

Evaluate objs, targets.

* **Parameters**
  * **eval_objs** (*list*) – objs for evaluation
  * **eval_targets** (*list*) – target list for the corresponding objects
  * **is_feature** (*bool*) – whether the input x is feature matrix
  * **metric** (*callable*) – metric for evaluation

#### *classmethod* from_file(filename: str, \*\*kwargs)

Load the models from file
:param filename: filename
:type filename: str
:param \*\*kwargs:

Returns: object instance

#### load(filename: str)

Load models parameters from filename
:param filename: models file name.
:type filename: str

Returns: None

#### save(filename: str)

Save the models and describers.

* **Parameters**
  **filename** (*str*) – filename for save

### maml.base._model.is_keras_model(model: BaseModel)

Check whether the model is keras
:param model: model
:type model: BaseModel

Returns: bool.

### maml.base._model.is_sklearn_model(model: BaseModel)

Check whether the model is sklearn
:param model: model
:type model: BaseModel

Returns: bool.