from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering

from maml.sampling.clustering import BirchClustering, FastclusterWard


class TestBirchClustering:
    def setup_method(self):
        self.Birch = BirchClustering(n=2, threshold_init=0.01)
        self.Birch_t002 = BirchClustering(n=10, threshold_init=0.1)

    def test_fit(self, PC_weighted):
        assert self.Birch == self.Birch.fit(PC_weighted)

    def test_transform(self, Birch_results, PC_weighted):
        clustering_results = self.Birch.transform(PC_weighted)
        assert "labels" in clustering_results
        assert "label_centers" in clustering_results
        assert "PCAfeatures" in clustering_results
        assert (clustering_results["labels"] == Birch_results["labels"]).all()
        assert all(
            ((clustering_results["label_centers"].get(k) == v).all() for k, v in Birch_results["label_centers"].items())
        )
        assert (clustering_results["PCAfeatures"] == Birch_results["PCAfeatures"]).all()
        clustering_results_t002 = self.Birch_t002.transform(PC_weighted)
        assert len(set(clustering_results_t002["labels"])) == 10


@pytest.mark.parametrize(
    ("n_samples", "n_features", "n_clusters"), [(2, 1, 1), (50, 3, 5), (300, 4, 40), (120, 2, 120)]
)
def test_fastcluster_ward_matches_sklearn(n_samples, n_features, n_clusters):
    points = np.random.default_rng(1).normal(size=(n_samples, n_features))
    expected = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(points)
    np.testing.assert_array_equal(FastclusterWard(n_clusters=n_clusters).fit_predict(points), expected)


def test_birch_fastcluster_matches_sklearn():
    points = np.random.default_rng(2).normal(size=(2000, 4))
    reference = BirchClustering(n=50, threshold_init=0.1, use_fastcluster=False).transform(points)
    fast = BirchClustering(n=50, threshold_init=0.1, use_fastcluster=True).transform(points)
    np.testing.assert_array_equal(fast["labels"], reference["labels"])
    assert fast["label_centers"].keys() == reference["label_centers"].keys()
    for label, center in reference["label_centers"].items():
        np.testing.assert_allclose(fast["label_centers"][label], center)
