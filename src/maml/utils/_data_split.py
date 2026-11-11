"""Data split."""

from __future__ import annotations

import itertools

import numpy as np
from monty.json import MSONable
from sklearn.utils import shuffle


class DataSplitter(MSONable):
    """Data splitter base class."""

    def split(self, mat_ids):
        """
        Split the mat_ids, optionally one can provide
        targets. This is useful in stratified split.

        Args:
            mat_ids (list): list of material ids

        Returns: (train_ids, val_ids, test_ids) or
            (train_ids, test_ids)
        """


class ShuffleSplitter(DataSplitter):
    """
    Randomly shuffe the material ids and split the ids
    into given ratios.
    """

    def __init__(self, ratios: str = "80/10/10", delim: str = "/", random_seed: int | None = None):
        """
        Random shuffle split of material ids.

        Args:
            ratios (str): ratios
            delim (str): deliminators for separate ratios
            random_seed (int): random seed.
        """
        self.random_seed = random_seed
        int_ratios = [float(i) for i in ratios.strip().split(delim)]
        self.ratios = [i / sum(int_ratios) for i in int_ratios]

    def split(self, mat_ids) -> tuple:
        """
        Randomly split the mat_ids.

        Args:
            mat_ids (list): material ids
        Returns: split ids.

        """
        mat_id_copy = shuffle(mat_ids, random_state=self.random_seed)
        n = len(mat_ids)
        end_points = [0, *np.cumsum([int(n * i) for i in self.ratios[:-1]]).tolist(), n]
        return tuple(mat_id_copy[i:j] for i, j in itertools.pairwise(end_points))
