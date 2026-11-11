"""MAML models base classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import joblib
import numpy as np

from maml.base import BaseDescriber, DummyDescriber
from maml.utils import get_full_args, to_array

if TYPE_CHECKING:
    from collections.abc import Callable


class BaseModel:
    """
    Abstract Base class for a Model. Basically, it usually wraps around a deep
    learning package, e.g., the Sequential Model in Keras, but provides for
    transparent conversion of arbitrary input and outputs.
    """

    def __init__(self, model, describer: BaseDescriber | None = None):
        """
        Args:
            model (Any): ML models, for example, sklearn models or keras models
            describer (BaseDescriber): Describer that converts object into features.
        """
        if describer is None:
            describer = DummyDescriber()
        self.describer = describer
        self.model = model

    def fit(
        self,
        features: list | np.ndarray,
        targets: list | np.ndarray | None = None,
        val_features: list | np.ndarray | None = None,
        val_targets: list | np.ndarray | None = None,
        **kwargs,
    ) -> BaseModel:
        """
        Args:
            features (list or np.ndarray): Numerical input feature list or
                numpy array with dim (m, n) where m is the number of data and
                n is the feature dimension.
            targets (list or np.ndarray): Numerical output target list, or
                numpy array with dim (m, ).
            val_features (list or np.ndarray): validation features
            val_targets (list or np.ndarray): validation targets.
            **kwargs: pass to the fit method of the respective ML model.

        Returns:
            self
        """
        self.model.fit(features, targets, **kwargs)  # type: ignore
        return self

    def train(
        self,
        objs: list | np.ndarray,
        targets: list | np.ndarray | None = None,
        val_objs: list | np.ndarray | None = None,
        val_targets: list | np.ndarray | None = None,
        **kwargs,
    ) -> BaseModel:
        """
        Train the models from object, target pairs.

        Args:
            objs (list of objects): List of objects
            targets (list): list of float or np.ndarray
            val_objs (list of objects): list of validation objects
            val_targets (list): list of validation targets
            **kwargs: pass to the fit method of this BaseModel.

        Returns: self

        """
        features = to_array(self.describer.fit_transform(objs))
        targets = to_array(targets)

        if (val_objs is None) and (val_targets is not None):
            raise ValueError("training objects are none, but the targets are not")
        if val_objs is not None:
            val_features = to_array(self.describer.transform(val_objs))  # type: ignore
            val_targets = to_array(val_targets)
        else:
            val_features = None
            val_targets = None
        return self.fit(
            features=features,
            targets=targets,
            val_features=val_features,
            val_targets=val_targets,
            **kwargs,
        )

    def _predict(self, features: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict the values given a set of inputs based on fitted models.

        Args:
            features (np.ndarray): array-like input features.
            **kwargs: pass to the predict method of the respective ML model.

        Returns:
            List of output objects.
        """
        return self.model.predict(np.array(features), **kwargs)  # type: ignore

    def predict_objs(self, objs: list | np.ndarray) -> np.ndarray:
        """
        Predict the values given a set of objects. Usually Pymatgen
            Structure objects.
        """
        return self._predict(self.describer.transform(objs))  # type: ignore


class SklearnMixin:
    """Sklearn models save and load functionality."""

    def save(self, filename: str):
        """Save the models and describers.

        Arguments:
            filename (str): filename for save
        """
        joblib.dump({"models": self.model, "describers": self.describer}, filename)

    def load(self, filename: str):
        """
        Load models parameters from filename
        Args:
            filename (str): models file name.

        Returns: None

        """
        m = joblib.load(filename)
        self.model = m["models"]
        self.describer = m["describers"]

    @classmethod
    def from_file(cls, filename: str, **kwargs):
        """
        Load the models from file.

        Args:
            filename (str): filename
            **kwargs: pass to the SklearnMixin

        Returns: object instance

        """
        instance = cls(model=None, **kwargs)  # type: ignore
        instance.load(filename)
        return instance

    def evaluate(
        self,
        eval_objs: list | np.ndarray,
        eval_targets: list | np.ndarray,
        is_feature: bool = False,
        metric: str | Callable | None = None,
    ) -> np.ndarray:
        """
        Evaluate objs, targets.

        Args:
            eval_objs (list): objs for evaluation
            eval_targets (list): target list for the corresponding objects
            is_feature (bool): whether the input x is feature matrix
            metric (callable): metric for evaluation
        """
        from sklearn.metrics import check_scoring

        eval_features = eval_objs if is_feature else self.describer.transform(eval_objs)
        scorer = check_scoring(self.model, scoring=metric)
        return scorer(self.model, eval_features, eval_targets)


class KerasMixin:
    """keras models mixin with save and load functionality."""

    def save(self, filename: str):
        """Save the models and describers.

        Arguments:
            filename (str): filename for save
        """
        joblib.dump(self.describer, filename)
        self.model.save(filename + ".hdf5")

    def load(self, filename: str, custom_objects: list | None = None):
        """
        Load models parameters from filename.

        Args:
            filename (str): models file name.
            custom_objects: Optional dictionary mapping names (strings) to custom classes or functions to be
                considered during deserialization.

        Returns: None

        """
        import tensorflow as tf

        self.describer = joblib.load(filename)
        self.model = tf.keras.models.load_model(filename + ".hdf5", custom_objects=custom_objects)

    @classmethod
    def from_file(cls, filename: str, **kwargs):
        """
        Load the models from file.

        Args:
            filename (str): filename
            **kwargs: pass to the KerasMixin

        Returns: object instance

        """
        instance = cls(model=None, **kwargs)  # type: ignore
        instance.load(filename)
        return instance

    def evaluate(
        self,
        eval_objs: list | np.ndarray,
        eval_targets: list | np.ndarray,
        is_feature: bool = False,
    ) -> np.ndarray:
        """
        Evaluate objs, targets.

        Args:
            eval_objs (list): objs for evaluation
            eval_targets (list): target list for the corresponding objects
            is_feature (bool): whether the input x is feature matrix
            metric (callable): metric for evaluation
        """
        eval_features = eval_objs if is_feature else self.describer.transform(eval_objs)
        return self.model.evaluate(to_array(eval_features), to_array(eval_targets))  # type: ignore

    @staticmethod
    def get_input_dim(describer: BaseDescriber | None = None, input_dim: int | None = None) -> int | None:
        """
        Get feature dimension/input_dim from describers or input_dim.

        Args:
            describer (Describer): describers
            input_dim (int): optional input dim int
        """
        if input_dim is not None:
            feature_size: int | None = input_dim
        elif describer is not None:
            feature_size = describer.feature_dim
        else:
            feature_size = None
        return feature_size


class SKLModel(BaseModel, SklearnMixin):
    """MAML models with sklearn models as estimator."""

    def __init__(self, model, describer: BaseDescriber | None = None, **kwargs):
        """
        Args:
            model (Any): ML models, for example, sklearn models or keras models
            describer (BaseDescriber): Describer that converts object into features.
            **kwargs: pass to super class (BaseModel) construct the model.
        """
        super().__init__(model=model, describer=describer, **kwargs)


class KerasModel(BaseModel, KerasMixin):
    """MAML models with keras models as estimators."""

    def __init__(self, model, describer: BaseDescriber | None = None, **kwargs):
        """
        Args:
            model (Any): ML models, for example, sklearn models or keras models.
            describer (BaseDescriber): Describer that converts object into features.
            **kwargs: pass to BaseModel to construct KerasModel.
        """
        super().__init__(model=model, describer=describer, **kwargs)

    def fit(
        self,
        features: list | np.ndarray,
        targets: list | np.ndarray | None = None,
        val_features: list | np.ndarray | None = None,
        val_targets: list | np.ndarray | None = None,
        **kwargs,
    ) -> BaseModel:
        """
        Args:
            features (list or np.ndarray): Numerical input feature list or
                numpy array with dim (m, n) where m is the number of data and
                n is the feature dimension.
            targets (list or np.ndarray): Numerical output target list, or
                numpy array with dim (m, ).
            val_features (list or np.ndarray): validation features
            val_targets (list or np.ndarray): validation targets.
            **kwargs: pass to the fit method of the ML model in KerasModel.

        Returns:
            self
        """
        if (val_features is None) and (val_targets is not None):
            raise ValueError("Validation data incorrect")
        targets = np.array(targets)
        val_targets = np.array(val_targets)
        fit_args = get_full_args(self.model.fit)  # type: ignore

        # construct validation data for keras models
        val_kwargs = {}
        kw_remove = []
        for kw in kwargs:
            if kw not in fit_args:
                val_kwargs.update({kw: kwargs.get(kw)})
                kw_remove.append(kw)

        for kw in kw_remove:
            kwargs.pop(kw)

        if val_features is not None:
            validation_data = self._get_validation_data(val_features, val_targets, **val_kwargs)
        else:
            validation_data = None
        self.model.fit(features, targets, validation_data=validation_data, **kwargs)  # type: ignore
        return self

    @staticmethod
    def _get_validation_data(val_features, val_targets, **val_kwargs):
        """
        Construct validation data, the default is just returning a list of
        val_features and val_targets.
        """
        return val_features, val_targets


def is_sklearn_model(model: BaseModel) -> bool:
    """
    Check whether the model is sklearn
    Args:
        model (BaseModel): model
    Returns: bool.
    """
    from sklearn.base import BaseEstimator

    return isinstance(model.model, BaseEstimator)


def is_keras_model(model: BaseModel) -> bool:
    """
    Check whether the model is keras
    Args:
        model (BaseModel): model
    Returns: bool.
    """
    from tensorflow.keras.models import Model

    return isinstance(model.model, Model)
