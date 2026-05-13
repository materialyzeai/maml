"""Wrapper for matminer_wrapper featurizers."""

from __future__ import annotations

import logging
from inspect import signature
from typing import TYPE_CHECKING, Any

import pandas as pd

from maml.base import BaseDescriber

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def wrap_matminer_describer(
    cls_name: str, wrapped_class: Any, obj_conversion: Callable, describer_type: Any | None = None
):
    """
    Wrapper of matminer_wrapper describers.

    Args:
        cls_name (str): new class name
        wrapped_class (class object): matminer_wrapper BaseFeaturizer
        obj_conversion (callable): function to convert objects into desired
            object type within transform_one
        describer_type (object): object type.

    Returns: maml describers class
    """

    def constructor(self, *args, **kwargs):
        """Wrapped __init__ constructor."""
        n_jobs = kwargs.pop("n_jobs", 0)
        memory = kwargs.pop("memory", None)
        verbose = kwargs.pop("verbose", False)
        feature_batch = kwargs.pop("feature_concat", "pandas_concat")
        wrapped_class(*args, **kwargs)
        logger.info("Using matminer_wrapper %s class", wrapped_class.__name__)
        base_kwargs = {"n_jobs": n_jobs, "memory": memory, "verbose": verbose, "feature_batch": feature_batch}
        BaseDescriber.__init__(self, **base_kwargs)

    @classmethod  # type: ignore
    def _get_param_names(cls):  # type: ignore
        return wrapped_class._get_param_names()

    def get_params(self, deep=False):
        return wrapped_class.get_params(self, deep=deep)

    def transform_one(self, obj: Any):
        """Featurize to transform_one."""
        obj = obj_conversion(obj)
        results = wrapped_class.featurize(self, obj)
        labels = wrapped_class.feature_labels(self)
        return pd.DataFrame({label: [value] for label, value in zip(labels, results, strict=True)})

    @classmethod  # type: ignore
    def from_preset(cls, name: str, **kwargs):  # type: ignore
        """Wrap matminer_wrapper's from_preset function."""
        instance = wrapped_class.from_preset(name)
        sig = signature(wrapped_class.__init__)
        args = list(sig.parameters.keys())[1:]
        params = dict.fromkeys(args)
        params.update(**kwargs)
        instance_new = cls(**params)
        instance_new.__dict__.update(instance.__dict__)
        return instance_new

    return type(
        cls_name,
        (BaseDescriber,),
        {
            "__doc__": wrapped_class.__doc__,
            "__init__": constructor,
            "__str__": wrapped_class.__str__,
            "__repr__": wrapped_class.__repr__,
            "__getstate__": wrapped_class.__getstate__,
            "__setstate__": wrapped_class.__setstate__,
            "_get_param_names": _get_param_names,
            "transform_one": transform_one,
            "from_preset": from_preset,
            "get_params": get_params,
            "__module__": "maml.describers",
            "describer_type": describer_type,
        },
    )
