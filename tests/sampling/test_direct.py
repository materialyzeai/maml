from __future__ import annotations

import pytest

import maml.sampling.direct as direct_module
from maml.sampling.clustering import BirchClustering
from maml.sampling.direct import DIRECTSampler


class TestDIRECTSampler:
    def setup_method(self):
        self.direct_fixed_n = DIRECTSampler(structure_encoder=None, clustering=BirchClustering(n=1))
        self.direct_fixed_t = DIRECTSampler(
            structure_encoder=None,
            clustering=BirchClustering(n=None, threshold_init=0.5),
        )

    def test_fit_transform(self, MPF_2021_2_8_first10_features_test):
        result_fixed_n = self.direct_fixed_n.fit_transform(MPF_2021_2_8_first10_features_test["M3GNet_features"])
        result_fixed_t = self.direct_fixed_t.fit_transform(MPF_2021_2_8_first10_features_test["M3GNet_features"])
        assert result_fixed_n["selected_indexes"] == [9]
        assert result_fixed_t["selected_indexes"] == [0, 6, 7, 8, 9]


class _FakeEncoder:
    def __init__(self, tag):
        self.tag = tag

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x


@pytest.mark.parametrize(("name", "tag"), [("MatGL", "matgl"), ("M3GNet", "m3gnet")])
def test_named_structure_encoders(monkeypatch, name, tag):
    # avoid loading the pretrained models; only the dispatch on the encoder name is tested here
    monkeypatch.setattr(direct_module, "MatGLStructure", lambda: _FakeEncoder("matgl"))
    monkeypatch.setattr(direct_module, "M3GNetStructure", lambda: _FakeEncoder("m3gnet"))
    sampler = DIRECTSampler(structure_encoder=name)
    assert sampler.structure_encoder.tag == tag
    assert [step for step, _ in sampler.steps] == [
        "_FakeEncoder",
        "StandardScaler",
        "PrincipalComponentAnalysis",
        "BirchClustering",
        "SelectKFromClusters",
    ]


def test_custom_encoder_is_used(MPF_2021_2_8_first10_features_test):
    features = MPF_2021_2_8_first10_features_test["M3GNet_features"]
    sampler = DIRECTSampler(structure_encoder=_FakeEncoder("custom"), clustering=direct_module.BirchClustering(n=1))
    assert sampler.steps[0][1].tag == "custom"
    assert sampler.fit_transform(features)["selected_indexes"] == [9]
