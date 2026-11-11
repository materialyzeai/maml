from __future__ import annotations

import os
import unittest
from unittest.mock import create_autospec

import pytest
from monty.serialization import dumpfn, loadfn
from pymatgen.core import Lattice, Structure
from pymatgen.ext.matproj import MPRester, MPRestError
from pymatgen.util.testing import PymatgenTest

import maml.apps.gbe as gbe
import maml.apps.gbe.utils as utils
from maml.apps.gbe.utils import load_b0_dict, load_data, load_mean_delta_bl_dict, pjoin, update_b0_dict

module_dir = os.path.dirname(gbe.__file__)
REFS = pjoin(module_dir, "references")
DATA = pjoin(module_dir, "data")


class TestUtil(PymatgenTest):
    def test_load_data(self):
        data = load_data()
        assert len(data) == 361

    def test_b0(self):
        b0_dict = load_b0_dict()
        assert os.path.isfile(pjoin(REFS, "el2b0.json"))
        assert b0_dict["W"] == 2.7603814963781725
        els = [
            "Ac",
            "Ag",
            "Al",
            "Au",
            "Ba",
            "Be",
            "Ca",
            "Cd",
            "Ce",
            "Co",
            "Cr",
            "Cs",
            "Cu",
            "Dy",
            "Er",
            "Eu",
            "Fe",
            "Gd",
            "Hf",
            "Hg",
            "Ho",
            "Ir",
            "K",
            "La",
            "Li",
            "Lu",
            "Mg",
            "Mo",
            "Na",
            "Nb",
            "Nd",
            "Ni",
            "Os",
            "Pb",
            "Pd",
            "Pm",
            "Pr",
            "Pt",
            "Rb",
            "Re",
            "Rh",
            "Ru",
            "Sc",
            "Sm",
            "Sr",
            "Ta",
            "Tb",
            "Tc",
            "Th",
            "Ti",
            "Tl",
            "Tm",
            "V",
            "W",
            "Y",
            "Yb",
            "Zn",
            "Zr",
        ]
        assert sorted(b0_dict.keys()) == els

    def test_mean_delta_bl(self):
        mean_delta_bl = load_mean_delta_bl_dict()
        assert os.path.isfile(pjoin(REFS, "mean_bl_chg_crystalnn.json"))
        assert mean_delta_bl["5094"] == -0.004265041421307773
        assert len(mean_delta_bl) == 361


W_BULK = Structure.from_file(pjoin(REFS, "mp-91.cif"))


@pytest.fixture
def fake_rester(monkeypatch):
    """Autospec'd MPRester (no network) returning bulk W for any material_id."""
    rester = create_autospec(MPRester, instance=True)
    rester.get_structure_by_material_id.side_effect = lambda material_id: W_BULK.copy()
    monkeypatch.setattr(utils, "MPRester", lambda api_key: rester)
    return rester


def test_load_data_from_file(tmp_path):
    data = [{"task_id": 1, "pretty_formula": "W"}, {"task_id": 2, "pretty_formula": "Mo"}]
    fname = tmp_path / "gb.json"
    dumpfn(data, fname)
    assert load_data(str(fname)) == data


def test_update_b0_dict(fake_rester):
    data = [
        {"pretty_formula": "W", "material_id": "mp-91"},
        {"pretty_formula": "W", "material_id": "mp-91"},
        {"pretty_formula": "Mo", "material_id": "mp-129"},
    ]
    b0_dict = update_b0_dict(data, mp_api="fake-key")
    assert set(b0_dict) == {"W", "Mo"}
    # Primitive bcc W: b0 matches the value shipped in references/el2b0.json.
    assert b0_dict["W"] == pytest.approx(2.7603814963781725)
    # One query per unique formula.
    assert [c.args[0] for c in fake_rester.get_structure_by_material_id.call_args_list] == ["mp-91", "mp-129"]


def test_update_b0_dict_capped_by_lattice_a(monkeypatch):
    # Conventional bcc cell: the [111] self-image distance (sqrt(3) a) exceeds a, so b0 = a.
    conv = Structure(Lattice.cubic(3.17), ["W", "W"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    rester = create_autospec(MPRester, instance=True)
    rester.get_structure_by_material_id.return_value = conv
    monkeypatch.setattr(utils, "MPRester", lambda api_key: rester)
    assert update_b0_dict([{"pretty_formula": "W", "material_id": "x"}], mp_api="k") == {"W": pytest.approx(3.17)}


def test_load_b0_dict_missing_without_key(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "REFS", str(tmp_path))
    monkeypatch.delenv("MAPI_KEY", raising=False)
    with pytest.raises(MPRestError, match="MAPI_KEY"):
        load_b0_dict()


def test_load_b0_dict_downloads_and_caches(monkeypatch, tmp_path, fake_rester):
    monkeypatch.setattr(utils, "REFS", str(tmp_path))
    monkeypatch.setenv("MAPI_KEY", "fake-key")
    monkeypatch.setattr(utils, "load_data", lambda: [{"pretty_formula": "W", "material_id": "mp-91"}])
    b0_dict = load_b0_dict()
    assert b0_dict == {"W": pytest.approx(2.7603814963781725)}
    cached = tmp_path / "el2b0.json"
    assert cached.is_file()
    assert loadfn(cached) == b0_dict
    # Second call reads the cache rather than querying again.
    assert load_b0_dict() == b0_dict
    assert fake_rester.get_structure_by_material_id.call_count == 1


def test_load_mean_delta_bl_dict_missing():
    with pytest.raises(ValueError, match="mean_delta_bond_length"):
        load_mean_delta_bl_dict(loc_algo="voronoinn")


if __name__ == "__main__":
    unittest.main()
