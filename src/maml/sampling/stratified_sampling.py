"""Implementation of stratefied sampling approaches."""

from __future__ import annotations

import logging
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SelectKFromClusters(BaseEstimator, TransformerMixin):
    """Wrapper around selection of k data from each cluster."""

    def __init__(
        self,
        k: int = 1,
        allow_duplicate=False,
        selection_criteria="center",
        n_sites=None,
    ):
        """
        Args:
            k: Select k structures from each cluster.
            allow_duplicate: Whether structures are allowed to be selected over once.
            selection_criteria: The criteria to do stratified sampling from each cluster. Supported criterion
                include "random", "smallest", and "center" (default). By default, structures are ranked with respect to
                their distances to the centroid of the cluster they locate, then up to k structures with fixed ranking
                intervals are selected from each cluster, and when k=1, the structure with the smallest Euclidean
                distance to the centroid of each cluster is sampled. For "random", k structures are randomly sampled
                with replacement. For "smallest", it is ensured to select the k structures with the least number of
                atoms in each cluster.
            n_sites: The number of sites in all the structures to sample from. Only needed when
                selection_criteria="smallest".
        """
        self.k = k
        self.allow_duplicate = allow_duplicate
        allowed_selection_criterion = ["random", "smallest", "center"]
        if selection_criteria not in allowed_selection_criterion:
            raise ValueError(f"Invalid selection_criteria, it must be one of {allowed_selection_criterion}.")
        if selection_criteria == "smallest" and not n_sites:
            raise ValueError('n_sites must be provided when selection_criteria="smallest."')
        self.selection_criteria = selection_criteria
        self.n_sites = n_sites

    def fit(self, X, y=None):
        """
        Fit the model.

        Args:
            X: Input features
            y: Target.
        """
        return self

    def transform(self, clustering_data: dict):
        """
        Perform stratified sampling of data from each cluster
        based on clustering results.

        Args:
            clustering_data: Results from clustering in a dict. The dict
                should at least contain "PCAfeatures" and their respective
                "labels" of belonged clusters. The positions of centroid
                for each cluster should also be provided with "label_centers",
                with which data in each cluster can be ranked according to
                their Euclidean distance to centroid and then selected by
                interval for optimal coverage.

        Returns:
            A dict with "PCAfeatures" used in clustering and "selected_indexes"
                as the indexes of DIRECT sampled structures.
        """
        if any(key not in clustering_data for key in ["labels", "PCAfeatures"]):
            raise Exception(
                "The data returned by clustering step should at least provide label and feature information."
            )
        if self.selection_criteria == "center" and "label_centers" not in clustering_data:
            warnings.warn(
                "Centroid location is not provided, so random selection from each cluster will be performed, "
                "which likely will still outperform manual sampling in terms of feature coverage. "
            )
        if self.selection_criteria == "smallest":
            try:
                assert len(self.n_sites) == len(clustering_data["PCAfeatures"])
            except Exception:
                raise ValueError("n_sites must have same length as features processed in clustering.")

        selected_indexes = []
        for label in set(clustering_data["labels"]):
            indexes_same_label = np.where(label == clustering_data["labels"])[0]
            features_same_label = clustering_data["PCAfeatures"][indexes_same_label]
            n_same_label = len(features_same_label)
            if "label_centers" in clustering_data and self.selection_criteria == "center":
                center_same_label = clustering_data["label_centers"][label]
                distance_to_center = np.linalg.norm(features_same_label - center_same_label, axis=1).reshape(
                    len(indexes_same_label)
                )
                select_k_indexes = np.array([int(i) for i in np.linspace(0, n_same_label - 1, self.k)])
                selected_indexes.extend(
                    indexes_same_label[np.argpartition(distance_to_center, select_k_indexes)[select_k_indexes]]
                )
            elif self.selection_criteria == "smallest":
                if self.k >= n_same_label:
                    selected_indexes.extend(indexes_same_label)
                else:
                    select_k_indexes = np.arange(self.k)
                    selected_indexes.extend(
                        indexes_same_label[
                            np.argpartition(
                                np.array(self.n_sites)[indexes_same_label],
                                select_k_indexes,
                            )[select_k_indexes]
                        ]
                    )
            else:
                selected_indexes.extend(indexes_same_label[np.random.randint(n_same_label, size=self.k)])
        n_duplicate = len(selected_indexes) - len(set(selected_indexes))
        if not self.allow_duplicate and n_duplicate > 0:
            selected_indexes = list(set(selected_indexes))
        elif self.allow_duplicate and n_duplicate > 0:
            warnings.warn(f"There are {n_duplicate} duplicated selections.")
        logger.info(f"Finally selected {len(selected_indexes)} configurations.")
        return {
            "PCAfeatures": clustering_data["PCAfeatures"],
            "selected_indexes": selected_indexes,
        }
