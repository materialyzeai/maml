"""Module implements the scaler."""

from __future__ import annotations

import numpy as np
from monty.json import MSONable


class StandardScaler(MSONable):
    """
    StandardScaler follows the sklean manner with addition of
    dictionary representation.
    """

    def __init__(
        self,
        mean: list | np.ndarray | None = None,
        std: list | np.ndarray | None = None,
    ):
        """
        Args:
            mean: np.ndarray, mean values
            std: np.ndnarray, standard deviations.
        """
        self.mean = mean
        self.std = std

    def fit(self, target: list | np.ndarray) -> None:
        """
        Fit the StandardScaler to the target.

        Args:
            target (ndarray): The (mxn) ndarray. m is the number of samples,
                n is the number of feature dimensions.
        """
        mean = np.mean(target, axis=0)
        std = np.std(target, axis=0)
        self.mean = mean
        self.std = std

    def transform(self, target: np.ndarray) -> np.ndarray:
        """
        Transform target according to the mean and std.

        Args:
            target (ndarray): The (mxn) ndarray. m is the number of samples,
                n is the number of feature dimensions.
        """
        if self.mean is None or self.std is None:
            raise ValueError("No parameters is given.")
        return (target - self.mean) / self.std

    def inverse_transform(self, transformed_target: np.ndarray) -> np.ndarray:
        """
        Inversely transform the target.

        Args:
            transformed_target (ndarray): The (mxn) ndarray. m is the number of samples,
                n is the number of feature dimensions.
        """
        if self.mean is None or self.std is None:
            raise ValueError("No parameters is given.")
        return transformed_target * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

    def as_dict(self):
        """Dict representation of StandardScaler."""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "params": {
                "mean": None if self.mean is None else np.asarray(self.mean).tolist(),
                "std": None if self.std is None else np.asarray(self.std).tolist(),
            },
        }

    @classmethod
    def from_dict(cls, d):
        """
        Reconstitute a StandardScaler object from a dict representation of
        StandardScaler created using as_dict().

        Args:
            d (dict): Dict representation of StandardScaler.
        """
        return cls(**d["params"])


class DummyScaler(MSONable):
    """Dummy scaler does nothing."""

    def fit(self, target: list | np.ndarray) -> None:
        """
        Fit the DummyScaler to the target.

        Args:
            target (ndarray): The (mxn) ndarray. m is the number of samples,
                n is the number of feature dimensions.
        """
        return

    def transform(self, target: list | np.ndarray) -> list | np.ndarray:
        """
        Transform target.

        Args:
            target (ndarray): The (mxn) ndarray. m is the number of samples,
                n is the number of feature dimensions.
        """
        return target

    def inverse_transform(self, transformed_target: list | np.ndarray) -> list | np.ndarray:
        """
        Inversely transform the target.

        Args:
            transformed_target (ndarray): The (mxn) ndarray. m is the number of samples,
                n is the number of feature dimensions.
        """
        return transformed_target

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def as_dict(self):
        """
        Serialize the instance into dictionary.

        Returns: dict.
        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "params": {},
        }

    @classmethod
    def from_dict(cls, d):
        """
        Deserialize from a dictionary
        Args:
            d: Dict, dictionary contain class initialization parameters.

        Returns: DummyScaler.

        """
        return cls()
