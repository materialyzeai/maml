from __future__ import annotations

from maml.sampling.pca import PrincipalComponentAnalysis


class TestPrincipalComponentAnalysis:
    def setup_method(self):
        self.pca_weightedPC = PrincipalComponentAnalysis(weighting_PCs=True)
        self.pca_unweightedPC = PrincipalComponentAnalysis(weighting_PCs=False)

    def test_fit(self, MPF_2021_2_8_first10_features_test):
        assert self.pca_weightedPC == self.pca_weightedPC.fit(
            MPF_2021_2_8_first10_features_test["M3GNet_features_normalized"]
        )
        assert self.pca_unweightedPC == self.pca_unweightedPC.fit(
            MPF_2021_2_8_first10_features_test["M3GNet_features_normalized"]
        )

    def test_transform(self, MPF_2021_2_8_first10_features_test, PC_unweighted, PC_weighted):
        assert (
            self.pca_weightedPC.fit_transform(MPF_2021_2_8_first10_features_test["M3GNet_features_normalized"])
            == PC_weighted
        ).all
        assert (
            self.pca_unweightedPC.fit_transform(MPF_2021_2_8_first10_features_test["M3GNet_features_normalized"])
            == PC_unweighted
        ).all
