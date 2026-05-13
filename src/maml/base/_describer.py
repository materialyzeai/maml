"""MAML describers base classes."""

from __future__ import annotations

import abc
import logging
import tempfile
from typing import TYPE_CHECKING, Any

from joblib import Parallel, cpu_count, delayed
from monty.json import MSONable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_memory
from tqdm import tqdm

from maml.utils import feature_dim_from_test_system

from ._feature_batch import get_feature_batch

if TYPE_CHECKING:
    import numpy as np

_ALLOWED_DATA = ("number", "structure", "molecule", "spectrum")

_DESCRIBER_TYPES = [
    "composition",
    "site",
    "structure",
    "general",
    "band_structure",
    "spectrum",
]

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseDescriber(BaseEstimator, TransformerMixin, MSONable, metaclass=abc.ABCMeta):
    """
    Base class for a Describer. A describers converts an object to a descriptor,
    typically a numerical representation useful for machine learning.
    The output for the describers can be a single DataFrame/numpy.ndarray or
    a list of DataFrame/numpy.ndarray.
    """

    def __init__(self, **kwargs):
        """
        Base estimator with the following allowed keyword args.

            memory (bool/str/joblib.Memory): The path or Memory for caching the computational
                results, default None means no cache.
            verbose (bool): Whether to show the progress of feature calculations.
            n_jobs (int): The number of parallel jobs. 0 means no parallel computations.
                If this value is set to negative or greater than the total cpu
                then n_jobs is set to the number of cpu on system.
            feature_batch (str): method to batch a list of features into one

        Args:
            **kwargs: keyword args that contain possibly memory (str/joblib.Memory),
                verbose (bool), n_jobs (int)
        """
        allowed_kwargs = ["memory", "verbose", "n_jobs", "feature_batch"]

        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError(f"{k} not allowed as kwargs")
        memory = kwargs.get("memory")
        if isinstance(memory, bool):
            memory = tempfile.mkdtemp()
            logger.info(f"Created temporary directory {memory}")
        verbose = kwargs.get("verbose", False)
        n_jobs = kwargs.get("n_jobs", 0)

        self.memory = check_memory(memory)
        self.verbose = verbose
        # find out the number of parallel jobs
        if (n_jobs < 0) or (n_jobs > cpu_count()):
            n_jobs = cpu_count()
            logger.info(f"Using {n_jobs} jobs for computation")
        self.n_jobs = n_jobs
        self.feature_batch = get_feature_batch(kwargs.get("feature_batch"))

    def fit(self, x: Any, y: Any = None) -> BaseDescriber:
        """
        Place holder for fit API.

        Args:
            x: Any inputs
            y: Any outputs

        Returns: self

        """
        return self

    def transform_one(self, obj: Any) -> list[np.ndarray] | np.ndarray:
        """Transform an object."""
        raise NotImplementedError

    def transform(self, objs: list[Any]) -> Any:
        """
        Transform a list of objs. If the return data is DataFrame,
        use df.xs(index, level='input_index') to get the result for the i-th object.

        Args:
            objs (list): A list of objects.

        Returns:
            One or a list of pandas data frame/numpy ndarray
        """
        cached_transform_one = self.memory.cache(_transform_one)

        if self.verbose:
            objs = tqdm(objs)

        # run the featurizer
        if self.n_jobs == 0:
            features = [cached_transform_one(self, obj) for obj in objs]
        else:
            features = Parallel(n_jobs=self.n_jobs)(delayed(cached_transform_one)(self, obj) for obj in objs)

        multi_output = self._is_multi_output()
        if not multi_output:
            features = [features]
        batched_features = [self.feature_batch(i) for i in list(*zip(features, strict=False))]  # type: ignore
        return batched_features if multi_output else batched_features[0]

    def _is_multi_output(self) -> bool:
        # sklearn >= 1.6 replaced the private ``_get_tags`` dict API with
        # ``__sklearn_tags__()``, which returns a ``Tags`` dataclass whose
        # ``target_tags.multi_output`` flag is the modern equivalent of the
        # old ``tags["multioutput"]`` lookup.
        return self.__sklearn_tags__().target_tags.multi_output

    def clear_cache(self):
        """Clear cache."""
        if self.memory.location is not None:
            self.memory.clear()

    @property
    def feature_dim(self):
        """
        Feature dimension, useful when certain models need to specify
        the feature dimension, e.g., MLP models.
        """
        return feature_dim_from_test_system(self)


def _transform_one(describer: BaseDescriber, obj: Any) -> list[np.ndarray] | np.ndarray:
    """
    A wrapper to make a pure function.

    Args:
        describer (BaseDescriber): a describers
        obj: an object (structure/molecule) to transform.

    Returns:
        np.ndarray
    """
    return describer.transform_one(obj)


class DummyDescriber(BaseDescriber):
    """Dummy Describer that does nothing."""

    def transform_one(self, obj: Any):
        """
        Does nothing but return the original features.

        Args:
            obj: Any inputs

        Returns: Any outputs.

        """
        return obj


class SequentialDescriber(Pipeline):
    """A thin wrapper of sklearn Pipeline."""

    def __init__(self, describers: list, **kwargs):
        """
        Put a list of describers into one pipeline.

        Args:
            describers (list): a list of describers that will be applied
                consecutively
            **kwargs: pass to pipeline.
        """
        steps = [(i.__class__.__name__, i) for i in describers]
        super().__init__(steps, **kwargs)


def describer_type(dtype: str):
    """
    Decorate to set describers class type.

    Args:
        dtype (str): describers type, e.g., site, composition, structure etc.

    Return:
        wrapped class
    """

    def wrapped_describer(klass):
        klass.describer_type = dtype
        return klass

    return wrapped_describer
