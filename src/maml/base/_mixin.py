"""Model mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from maml.utils import DummyScaler

if TYPE_CHECKING:
    from ._model import BaseModel


class TargetScalerMixin:
    """Mixin class for models with target scaler."""

    def __init__(self, *args, **kwargs):
        """
        Mixin with target scaler, required kwarg target_scaler.

        Args:
            *args: input args for model constructor
            **kwargs: kwargs for model constructor
        """
        target_scaler = kwargs.pop("target_scaler", DummyScaler())
        self.target_scaler = target_scaler
        super().__init__(*args, **kwargs)

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
            **kwargs: kwargs for the train method.

        Returns: self

        """
        if targets is None:
            return super().train(objs, targets, val_objs, val_targets, **kwargs)  # type: ignore

        num_atoms = [len(i) for i in objs]
        targets = [self.target_scaler.transform(i, j) for i, j in zip(targets, num_atoms, strict=False)]
        if val_objs is not None and val_targets is not None:
            num_val_atoms = [len(i) for i in val_objs]
            val_targets = [self.target_scaler.transform(i, j) for i, j in zip(val_targets, num_val_atoms, strict=False)]
        super().train(objs, targets, val_objs, val_targets, **kwargs)  # type: ignore
        return self  # type: ignore

    def predict_objs(self, objs: list | np.ndarray) -> np.ndarray:
        """
        Predict the values given a set of objects. Usually Pymatgen
            Structure objects.
        """
        preds = super().predict_objs(objs)  # type: ignore
        ns = [len(i) for i in objs]
        return np.array([self.target_scaler.inverse_transform(i, j) for i, j in zip(preds, ns, strict=False)])
