"""Tests for the MEGNet describers.

``megnet`` is TF1-era and not installed in the test environment, so ``megnet.utils.*`` is replaced
by a fake ``MEGNetDescriptor`` to exercise maml's own logic.
"""

from __future__ import annotations

import pickle
import sys
import types

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure

from maml.describers import MEGNetSite, MEGNetStructure, _megnet
from maml.describers._megnet import MEGNetNotFound
from maml.utils import get_full_stats_and_funcs

ATOM_FEATURES = np.array([[1.0, 10.0], [3.0, 30.0], [2.0, 20.0]])


class FakeMEGNetDescriptor:
    instances: list = []

    def __init__(self, name_or_model):
        self.name_or_model = name_or_model
        self.valid_names = [f"meg_net_layer_{i}_{t}" for i in (1, 2) for t in ("atom", "bond", "state")] + [
            "set2set_atom",
            "dense_1",
        ]
        self.calls: list = []
        FakeMEGNetDescriptor.instances.append(self)

    def get_atom_features(self, obj, level):
        self.calls.append(("atom", level))
        return ATOM_FEATURES

    def get_set2set(self, obj, ftype):
        self.calls.append(("set2set", ftype))
        return np.ones((1, 4))

    def get_global_features(self, obj, level):
        self.calls.append(("global", level))
        return np.ones((1, 2)) * level

    def get_structure_features(self, obj):
        self.calls.append(("structure", None))
        return np.ones((1, 10))


@pytest.fixture
def fake_megnet(monkeypatch):
    FakeMEGNetDescriptor.instances = []
    loaded = []

    def load_model(name):
        loaded.append(name)
        return f"model<{name}>"

    descriptor = types.ModuleType("megnet.utils.descriptor")
    descriptor.MEGNetDescriptor = FakeMEGNetDescriptor
    models = types.ModuleType("megnet.utils.models")
    models.MODEL_MAPPING = {"Eform_MP_2019": "path"}
    models.load_model = load_model
    for name, mod in {
        "megnet": types.ModuleType("megnet"),
        "megnet.utils": types.ModuleType("megnet.utils"),
        "megnet.utils.descriptor": descriptor,
        "megnet.utils.models": models,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return loaded


@pytest.fixture(scope="module")
def nacl() -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.69169), ["Na", "Cl"], [[0, 0, 0], [0, 0, 0.5]])


@pytest.mark.parametrize("cls", [MEGNetSite, MEGNetStructure])
def test_missing_megnet(monkeypatch, cls):
    monkeypatch.setitem(sys.modules, "megnet", None)
    monkeypatch.setitem(sys.modules, "megnet.utils", None)
    monkeypatch.setitem(sys.modules, "megnet.utils.descriptor", None)
    with pytest.raises(MEGNetNotFound, match="requires installation of megnet"):
        cls()


@pytest.mark.parametrize(
    ("name", "expected", "n_loaded"),
    [
        (None, str(_megnet.DEFAULT_MODEL), 0),
        ("Eform_MP_2019", "model<Eform_MP_2019>", 1),
        ("/path/to/model.hdf5", "/path/to/model.hdf5", 0),
    ],
)
def test_load_model(fake_megnet, name, expected, n_loaded):
    descriptor = _megnet._load_model(name)
    assert descriptor.name_or_model == expected
    assert len(fake_megnet) == n_loaded


def test_load_model_object(fake_megnet):
    model = object()
    assert _megnet._load_model(model).name_or_model is model


@pytest.mark.parametrize("cls", [MEGNetSite, MEGNetStructure])
@pytest.mark.parametrize("prefix", ["meg_net", "megnet"])
def test_default_level(fake_megnet, monkeypatch, cls, prefix):
    names = [f"{prefix}_layer_{i}_{t}" for i in (1, 2, 3) for t in ("atom", "bond", "state")] + ["dense_1"]

    def init(self, name_or_model):
        self.name_or_model = name_or_model
        self.valid_names = names

    monkeypatch.setattr(FakeMEGNetDescriptor, "__init__", init)
    assert cls().level == 3
    assert cls(level=1).level == 1


def test_site_transform(fake_megnet, nacl):
    describer = MEGNetSite(level=1, feature_batch="pandas_concat")
    assert describer.describer_type == "site"
    df = describer.transform_one(nacl)
    np.testing.assert_array_equal(df.to_numpy(), ATOM_FEATURES)
    assert describer.describer_model.calls == [("atom", 1)]
    batched = describer.transform([nacl, nacl])
    assert batched.shape == (6, 2)


def test_structure_site_stats(fake_megnet, nacl):
    describer = MEGNetStructure(level=2)
    assert describer.describer_type == "structure"
    df = describer.transform_one(nacl)
    full_stats, _ = get_full_stats_and_funcs(["min", "max", "range", "mean", "mean_absolute_error", "mode"])
    assert list(df.columns) == [f"{i}_{s}" for i in range(2) for s in full_stats]
    assert df.shape == (1, 2 * len(full_stats))
    assert df["0_min"].item() == 1.0
    assert df["1_max"].item() == 30.0
    assert df["0_range"].item() == 2.0
    assert df["1_mean"].item() == pytest.approx(20.0)
    assert describer.describer_model.calls == [("atom", 2)]


def test_structure_custom_stats(fake_megnet, nacl):
    df = MEGNetStructure(level=1, stats=["max"]).transform_one(nacl)
    assert list(df.columns) == ["0_max", "1_max"]
    np.testing.assert_array_equal(df.to_numpy(), [[3.0, 30.0]])


@pytest.mark.parametrize(
    ("mode", "shape", "call"),
    [
        ("site_readout", (1, 4), ("set2set", "atom")),
        ("state", (1, 2), ("global", 2)),
        ("final", (1, 10), ("structure", None)),
    ],
)
def test_structure_modes(fake_megnet, nacl, mode, shape, call):
    describer = MEGNetStructure(mode=mode, level=2)
    df = describer.transform_one(nacl)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == shape
    assert describer.describer_model.calls == [call]


def test_structure_invalid_mode(fake_megnet, nacl):
    with pytest.raises(ValueError, match="Mode not allowed"):
        MEGNetStructure(mode="bogus").transform_one(nacl)


@pytest.mark.parametrize("cls", [MEGNetSite, MEGNetStructure])
def test_pickle_reloads_model(fake_megnet, cls):
    describer = cls(name="Eform_MP_2019", level=1)
    state = describer.__getstate__()
    assert state["describer_model"] is None
    assert describer.describer_model is not None  # original untouched

    restored = pickle.loads(pickle.dumps(describer))
    assert isinstance(restored.describer_model, FakeMEGNetDescriptor)
    assert restored.describer_model is not describer.describer_model
    assert restored.level == 1
    assert fake_megnet == ["Eform_MP_2019", "Eform_MP_2019"]
