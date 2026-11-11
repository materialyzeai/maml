"""Clustering methods."""

from __future__ import annotations

import heapq
import logging

import numpy as np
from fastcluster import linkage_vector
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import Birch

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FastclusterWard(ClusterMixin, BaseEstimator):
    """Ward agglomerative clustering with fastcluster.linkage_vector.

    Drop-in replacement for the AgglomerativeClustering(ward) global step of
    sklearn's Birch. sklearn builds the condensed distance matrix over all M
    subcluster centres (O(M^2) memory); linkage_vector needs only O(M * d).
    """

    def __init__(self, n_clusters: int = 2):
        """
        Args:
            n_clusters: Number of clusters to cut the Ward tree into.
        """
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        """
        Build the Ward tree and cut it into n_clusters flat clusters.

        Args:
            X: Array of shape (n_samples, n_features).
            y: Ignored.

        Returns: self
        """
        X = np.asarray(X, dtype=np.float64)
        if len(X) < max(self.n_clusters, 2):
            # Too few subclusters to merge: one label each, as sklearn's Birch does.
            self.labels_ = np.arange(len(X))
            return self
        children = linkage_vector(X, method="ward")[:, :2].astype(np.intp)
        self.labels_ = _cut_ward_tree(children, self.n_clusters, len(X))
        return self


def _cut_ward_tree(children: np.ndarray, n_clusters: int, n_leaves: int) -> np.ndarray:
    """Cut a merge tree into n_clusters, numbering clusters as sklearn's AgglomerativeClustering does.

    Undoes the last n_clusters - 1 merges with the same heap traversal as
    sklearn.cluster._agglomerative._hc_cut, so labels match sklearn, not just the partition.
    """
    nodes = [-(int(children[-1].max()) + 1)]
    for _ in range(n_clusters - 1):
        left, right = children[-nodes[0] - n_leaves]
        heapq.heappush(nodes, -int(left))
        heapq.heappushpop(nodes, -int(right))
    labels = np.empty(n_leaves, dtype=np.intp)
    for label, node in enumerate(nodes):
        stack = [-node]
        while stack:
            current = stack.pop()
            if current < n_leaves:
                labels[current] = label
            else:
                stack.extend(children[current - n_leaves])
    return labels


class BirchClustering(BaseEstimator, TransformerMixin):
    """Birch Clustering as one step of the DIRECT pipeline."""

    def __init__(self, n=None, threshold_init=0.5, use_fastcluster=True, **kwargs):
        """
        Args:
            n: Clustering the PCs into n clusters. When n is None, the number of clusters
                is dependent on threshold_init and other kwargs, and the final
                (global) clustering step is skipped. Default to None.
            threshold_init: The initial radius of the subcluster obtained by merging
                a new sample and the closest subcluster should be lesser than
                the threshold. Otherwise, a new subcluster is started. See details in:
                https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html.
                Users may tune this value for desired performance of birch, while 0.5
                is generally a good starting point, and some automatic tuning is done
                with our built-in codes to achieve n clusters if given.
            use_fastcluster: Whether to run the global (Ward) clustering step with
                fastcluster.linkage_vector, which uses O(M * d) instead of O(M^2) memory
                for M Birch subclusters and gives identical labels. False uses sklearn's
                AgglomerativeClustering instead. Only relevant when n is given. Default to True.
            **kwargs: Pass to BIRCH.
        """
        self.n = n
        self.threshold_init = threshold_init
        self.use_fastcluster = use_fastcluster
        self.kwargs = kwargs

    def fit(self, X, y=None):
        """
        Place holder for fit API.

        Args:
            X: Any inputs
            y: Any outputs

        Returns: self
        """
        return self

    def transform(self, PCAfeatures):
        """
        Perform Birch Clustering to an array of input PCA features.

        Args:
            PCAfeatures: An array of PCA features.

        Returns:
            A dict of Birch Clustering results, including labels of each
            PCA feature, centroid positions of each cluster in PCA feature s
            pace, and the array of input PCA features.
        """
        n_clusters = self.n
        if self.n is not None and self.use_fastcluster:
            n_clusters = FastclusterWard(n_clusters=self.n)
        model = Birch(n_clusters=n_clusters, threshold=self.threshold_init, **self.kwargs).fit(PCAfeatures)
        if self.n is not None:
            while (
                len(set(model.subcluster_labels_)) < self.n
            ):  # decrease threshold until desired n clusters is achieved
                logger.info(
                    f"BirchClustering with threshold_init={self.threshold_init} and n={self.n} "
                    f"gives {len(set(model.subcluster_labels_))} clusters.",
                )
                self.threshold_init = self.threshold_init / self.n * len(set(model.subcluster_labels_))
                model = Birch(n_clusters=n_clusters, threshold=self.threshold_init, **self.kwargs).fit(PCAfeatures)

        labels = model.predict(PCAfeatures)
        self.model = model
        logger.info(
            f"BirchClustering with threshold_init={self.threshold_init} and n={self.n} "
            f"gives {len(set(model.subcluster_labels_))} clusters.",
        )
        label_centers = dict(zip(model.subcluster_labels_, model.subcluster_centers_, strict=False))
        return {
            "labels": labels,
            "label_centers": label_centers,
            "PCAfeatures": PCAfeatures,
        }
