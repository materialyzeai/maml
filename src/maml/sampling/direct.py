"""DIRECT sampling."""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from maml.describers import M3GNetStructure, MatGLStructure

from .clustering import BirchClustering
from .pca import PrincipalComponentAnalysis
from .stratified_sampling import SelectKFromClusters


class DIRECTSampler(Pipeline):
    """
    DImensionality REduction-Clustering-sTratified (DIRECT)
    sampling Pipeline. For more details, please refer to our
    manuscript: https://arxiv.org/abs/2307.13710.
    """

    def __init__(
        self,
        structure_encoder="MatGL",
        scaler="StandardScaler",
        pca="PrincipalComponentAnalysis",
        weighting_PCs=True,
        clustering="Birch",
        select_k_from_clusters="select_k_from_clusters",
    ):
        """
        Args:
        structure_encoder: Structure featurizer. It can be any encoder
            that takes in a list of N structures and returns a 2-D array
            of N*D features, where D is the fixed dimensionality of the
            feature vector. By default, the M3GNet formation energy model
            is used. Set this to False to skip the encoding step when needed.
        scaler: StandardScaler to perform normalization before PCA.
        pca: PCA for dimensionality reduction.
        weighting_PCs: Whether to weight PC with their explained variance.
        clustering: Clustering method to clustering based on PCs.
        select_k_from_clusters: Straitified sampling of k structures from
            each cluster.
        """
        if structure_encoder == "MatGL":
            self.structure_encoder = MatGLStructure()
        elif structure_encoder == "M3GNet":
            self.structure_encoder = M3GNetStructure()
        else:
            self.structure_encoder = structure_encoder

        self.scaler = StandardScaler() if scaler == "StandardScaler" else scaler
        self.pca = (
            PrincipalComponentAnalysis(weighting_PCs=weighting_PCs) if pca == "PrincipalComponentAnalysis" else pca
        )
        self.weighting_PCs = weighting_PCs
        self.clustering = BirchClustering() if clustering == "Birch" else clustering
        self.select_k_from_clusters = (
            SelectKFromClusters() if select_k_from_clusters == "select_k_from_clusters" else select_k_from_clusters
        )
        steps = [
            (i.__class__.__name__, i)
            for i in [
                self.structure_encoder,
                self.scaler,
                self.pca,
                self.clustering,
                self.select_k_from_clusters,
            ]
            if i
        ]
        super().__init__(steps)
