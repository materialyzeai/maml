"""Offline tests for the MaterialsProject data source, with MPRester mocked out."""

from __future__ import annotations

from types import SimpleNamespace

import maml.data._mp as mp_module
from maml.data import MaterialsProject


class _FakeMPRester:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.calls = []
        self.materials = SimpleNamespace(summary=SimpleNamespace(search=self._search))

    def _search(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {"material_id": "mp-1", "formula_pretty": "Fe", "band_gap": 0.0, "extra": 1},
            {"material_id": "mp-2", "formula_pretty": "FeO", "band_gap": 2.1, "extra": 2},
        ]


def test_get_filters_properties(monkeypatch):
    monkeypatch.setattr(mp_module, "MPRester", _FakeMPRester)
    source = MaterialsProject(api_key="dummy")
    assert source.mpr.api_key == "dummy"

    df = source.get({"elements": ["Fe"]}, ["material_id", "band_gap"])

    assert source.mpr.calls == [{"elements": ["Fe"], "fields": ["material_id", "band_gap"]}]
    assert list(df.columns) == ["material_id", "band_gap"]
    assert df["material_id"].tolist() == ["mp-1", "mp-2"]
    assert df["band_gap"].tolist() == [0.0, 2.1]
