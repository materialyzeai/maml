"""Tests for the DFT energy model with VASP and its I/O mocked out."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pymatgen.core import Lattice, Structure

import maml.apps.bowsr.model.dft as dft_module
from maml.apps.bowsr.model.dft import DFT


@pytest.fixture
def nacl() -> Structure:
    return Structure(Lattice.cubic(5.6), ["Na", "Na", "Cl", "Cl"], [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0], [0, 0.5, 0]])


class FakePopen:
    """Stand-in for subprocess.Popen recording the command it was given."""

    calls: list = []
    returncode_to_use = 0
    stderr_to_use = b""

    def __init__(self, cmd, stdout=None, stderr=None):
        FakePopen.calls.append(cmd)
        self.returncode = FakePopen.returncode_to_use

    def communicate(self):
        return b"", FakePopen.stderr_to_use

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_vasp(monkeypatch, tmp_path):
    """Mock MPStaticSet, VASP execution, Vasprun and MP compatibility."""
    written = []

    class FakeStaticSet:
        def __init__(self, structure):
            self.structure = structure

        def write_input(self, output_dir):
            written.append(self.structure)

    total_energy = -25.0

    class FakeVasprun:
        def __init__(self, filename):
            assert filename == "vasprun.xml"

        def get_computed_entry(self):
            return SimpleNamespace(energy=total_energy)

    class FakeCompat:
        def process_entry(self, entry):
            return SimpleNamespace(energy=entry.energy - 1.0)  # apply a fake correction

    FakePopen.calls = []
    FakePopen.returncode_to_use = 0
    FakePopen.stderr_to_use = b""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dft_module, "MPStaticSet", FakeStaticSet)
    monkeypatch.setattr(dft_module, "Vasprun", FakeVasprun)
    monkeypatch.setattr(dft_module, "MaterialsProjectCompatibility", FakeCompat)
    monkeypatch.setattr(dft_module.subprocess, "Popen", FakePopen)
    return SimpleNamespace(written=written, total_energy=total_energy, correction=-1.0)


def test_init_explicit_exe():
    assert DFT(exe_path="/opt/vasp/bin/vasp_gam").vasp_exe == "/opt/vasp/bin/vasp_gam"


def test_init_finds_vasp_on_path(monkeypatch):
    monkeypatch.setattr(dft_module, "which", lambda name: f"/usr/local/bin/{name}")
    assert DFT().vasp_exe == "/usr/local/bin/vasp_std"


def test_init_without_vasp_raises(monkeypatch):
    monkeypatch.setattr(dft_module, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Vasp executable can not be found"):
        DFT()


def test_predict_energy_formation_energy_per_atom(fake_vasp, nacl):
    model = DFT(exe_path="vasp_std")
    energy = model.predict_energy(nacl)

    refs = dft_module.elements
    expected = (
        fake_vasp.total_energy
        + fake_vasp.correction
        - 2 * refs["Na"]["energy_per_atom"]
        - 2 * refs["Cl"]["energy_per_atom"]
    ) / 4
    assert energy == pytest.approx(expected)
    assert FakePopen.calls == [["vasp_std"]]
    assert fake_vasp.written == [nacl]


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        (
            b"some warning\nERROR: ZBRENT failed\nbracketing\n",
            "vasp exited with return code 1: ERROR: ZBRENT failed, bracketing",
        ),
        (b"line one\nsegfault", "vasp exited with return code 1: segfault"),
        (b"", "vasp exited with return code 1"),
    ],
)
def test_predict_energy_vasp_failure(fake_vasp, nacl, stderr, expected):
    """Regression: an empty stderr raised IndexError and the last line was dropped."""
    FakePopen.returncode_to_use = 1
    FakePopen.stderr_to_use = stderr
    with pytest.raises(RuntimeError) as excinfo:
        DFT(exe_path="vasp_std").predict_energy(nacl)
    assert str(excinfo.value) == expected
