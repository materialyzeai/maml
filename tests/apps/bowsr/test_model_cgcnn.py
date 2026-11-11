"""Tests for the CGCNN energy-model wrapper with the cgcnn package faked.

torch is a real dependency; only the (uninstallable) ``cgcnn`` package is
replaced by minimal stand-ins mirroring its public API.
"""

from __future__ import annotations

import importlib
import json
import sys
import types

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

import maml.apps.bowsr.model.cgcnn as cgcnn_module


class FakeGaussianDistance:
    """Same expansion as cgcnn.data.GaussianDistance."""

    def __init__(self, dmin, dmax, step, var=None):
        self.filter = np.arange(dmin, dmax + step, step)
        self.var = var if var is not None else step

    def expand(self, distances):
        return np.exp(-((distances[..., np.newaxis] - self.filter) ** 2) / self.var**2)


class FakeAtomCustomJSONInitializer:
    """Same lookup as cgcnn.data.AtomCustomJSONInitializer."""

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        self._embedding = {int(k): np.array(v, dtype=float) for k, v in elem_embedding.items()}

    def get_atom_fea(self, atom_type):
        return self._embedding[atom_type]


class FakeCrystalGraphConvNet(torch.nn.Module):
    instances: list = []

    def __init__(self, orig_atom_fea_len, nbr_fea_len, atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1):
        super().__init__()
        self.kwargs = {
            "orig_atom_fea_len": orig_atom_fea_len,
            "nbr_fea_len": nbr_fea_len,
            "n_conv": n_conv,
            "h_fea_len": h_fea_len,
            "n_h": n_h,
        }
        self.loaded_keys = None
        self.inputs = None
        FakeCrystalGraphConvNet.instances.append(self)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.loaded_keys = sorted(state_dict)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        self.inputs = (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        return torch.tensor([[0.5]])


@pytest.fixture
def cgcnn_wrapper(monkeypatch):
    """Reload the wrapper module with a fake ``cgcnn`` package installed."""
    pkg = types.ModuleType("cgcnn")
    data = types.ModuleType("cgcnn.data")
    model = types.ModuleType("cgcnn.model")
    data.GaussianDistance = FakeGaussianDistance
    data.AtomCustomJSONInitializer = FakeAtomCustomJSONInitializer
    model.CrystalGraphConvNet = FakeCrystalGraphConvNet
    for name, mod in {"cgcnn": pkg, "cgcnn.data": data, "cgcnn.model": model}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    FakeCrystalGraphConvNet.instances = []
    yield importlib.reload(cgcnn_module)
    monkeypatch.undo()
    importlib.reload(cgcnn_module)


@pytest.fixture
def nacl() -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.69), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_requires_cgcnn():
    if cgcnn_module.cgcnn is not None:
        pytest.skip("cgcnn is installed")
    with pytest.raises(RuntimeError, match="cgcnn and torch are needed"):
        cgcnn_module.CGCNNInput()


def test_normalizer_roundtrip(cgcnn_wrapper):
    data = torch.tensor([1.0, 2.0, 3.0, 6.0])
    normalizer = cgcnn_wrapper.CGCNNNormalizer(data)
    assert float(normalizer.mean) == pytest.approx(3.0)
    assert float(normalizer.std) == pytest.approx(float(torch.std(data)))
    normed = normalizer.norm(data)
    assert float(normed.mean()) == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(normalizer.denorm(normed), data)

    other = cgcnn_wrapper.CGCNNNormalizer(torch.zeros(3))
    other.load_state_dict(normalizer.state_dict())
    assert other.state_dict() == {"mean": normalizer.mean, "std": normalizer.std}


def test_generate_input_shapes_and_features(cgcnn_wrapper, nacl):
    gen = cgcnn_wrapper.CGCNNInput()
    atom_fea, nbr_fea, nbr_fea_idx = gen.generate_input(nacl)
    n = nacl.num_sites
    assert atom_fea.shape == (n, 92)
    assert nbr_fea.shape == (n, 12, 41)
    assert nbr_fea_idx.shape == (n, 12)
    assert nbr_fea_idx.dtype == torch.int64
    assert np.allclose(atom_fea[0].numpy(), gen.ari.get_atom_fea(nacl[0].specie.Z))

    # Neighbours are sorted by distance, so the first 6 neighbours of every
    # atom in rock salt are the opposite species at a/2.
    nn_dist = 5.69 / 2
    expected = np.exp(-((nn_dist - gen.gdf.filter) ** 2) / gen.gdf.var**2)
    assert np.allclose(nbr_fea[:, :6, :].numpy(), expected, atol=1e-6)
    species = np.array([site.specie.symbol for site in nacl])
    for i in range(n):
        assert np.all(species[nbr_fea_idx[i, :6].numpy()] != species[i])


def test_generate_input_disordered_site(cgcnn_wrapper):
    structure = Structure(Lattice.cubic(4.0), [{"Fe": 0.5, "Co": 0.5}, "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    gen = cgcnn_wrapper.CGCNNInput()
    atom_fea, _, _ = gen.generate_input(structure)
    expected = 0.5 * gen.ari.get_atom_fea(26) + 0.5 * gen.ari.get_atom_fea(27)
    assert np.allclose(atom_fea[0].numpy(), expected)


def test_generate_input_pads_missing_neighbours(cgcnn_wrapper):
    structure = Structure(Lattice.cubic(3.0), ["Cu"], [[0, 0, 0]])
    gen = cgcnn_wrapper.CGCNNInput(max_num_nbr=12, radius=3.5)
    with pytest.warns(UserWarning, match="7 not find enough neighbors"):
        _, nbr_fea, nbr_fea_idx = gen.generate_input(structure, cif_id=7)
    # Only the 6 nearest images lie within 3.5 A; the rest are padded with radius + 1.
    pad = np.exp(-((3.5 + 1.0 - gen.gdf.filter) ** 2) / gen.gdf.var**2)
    assert np.allclose(nbr_fea[0, 6:].numpy(), pad)
    assert np.all(nbr_fea_idx[0, 6:].numpy() == 0)


def test_generate_inputs(cgcnn_wrapper, nacl):
    gen = cgcnn_wrapper.CGCNNInput(radius=6, step=0.5)
    single = gen.generate_input(nacl)
    batch = gen.generate_inputs([nacl, nacl], cif_ids=[3, 4])
    assert len(batch) == 2
    assert batch[1][1].shape == (nacl.num_sites, 12, len(gen.gdf.filter))
    assert all(torch.allclose(a, b) for a, b in zip(single, batch[0], strict=True))
    assert len(gen.generate_inputs([nacl])) == 1


def test_cgcnn_predict_energy(cgcnn_wrapper, nacl):
    model = cgcnn_wrapper.CGCNN()
    net = FakeCrystalGraphConvNet.instances[-1]
    # Hyperparameters come from the bundled checkpoint.
    assert net.kwargs == {"orig_atom_fea_len": 92, "nbr_fea_len": 41, "n_conv": 4, "h_fea_len": 32, "n_h": 1}
    assert "convs.0.fc_full.weight" in net.loaded_keys
    assert float(model.normalizer.mean) == pytest.approx(-1.699178719088435)
    assert float(model.normalizer.std) == pytest.approx(1.0563029540802564)

    energy = model.predict_energy(nacl)
    assert energy == pytest.approx(0.5 * 1.0563029540802564 - 1.699178719088435, rel=1e-6)
    assert not net.training
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = net.inputs
    assert atom_fea.shape == (nacl.num_sites, 92)
    assert nbr_fea.shape == (nacl.num_sites, 12, 41)
    assert nbr_fea_idx.shape == (nacl.num_sites, 12)
    assert torch.equal(crystal_atom_idx[0], torch.arange(nacl.num_sites))
