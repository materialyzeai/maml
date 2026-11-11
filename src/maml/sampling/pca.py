"""This file contains the PCA implementation."""

from __future__ import annotations

import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PrincipalComponentAnalysis(BaseEstimator, TransformerMixin):
    """Wrap around PCA in scikit-learn to support weighting PCs."""

    def __init__(self, weighting_PCs=True):
        """
        Args:
            weighting_PCs: Whether to weight PCs with explained variances.
        """
        self.pca = PCA()
        self.weighting_PCs = weighting_PCs

    def fit(self, normalized_features):
        """
        Fit PCA with normalized features as input.

        Args:
            normalized_features: An array of normalized features with fixed
                dimensionality. The features have to be normalized first
                beforehand.

        Returns: self
        """
        self.pca.fit(normalized_features)
        return self

    def transform(self, normalized_features):
        """
        Transform normalized features into weighted or unweighted PCA features
        and select the first m PCs according to Kaiser's rule.

        Args:
            normalized_features: An array of normalized features with fixed
                dimensionality. The features have to be normalized first
                beforehand.

        Returns:
            An array of weighted or unweighted PCA feature with dimension of
            N x m, where N and m are the numbers of input normalized features
            and selected PCs according to Kaiser's rule, respectively.
        """
        m = len([e for e in self.pca.explained_variance_ if e > 1])
        explained_variance = self.pca.explained_variance_ratio_
        logger.info(f"Selected first {m} PCs, explaining {100 * sum(explained_variance[:m]):.2f}% variance")
        if not self.weighting_PCs:
            return self.pca.transform(normalized_features)[:, :m]
        return self.pca.transform(normalized_features)[:, :m] * explained_variance[:m]
