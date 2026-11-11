"""Shared fixtures for testing the PES wrappers without the external binaries.

The potentials shell out to LAMMPS, QUIP, MLIP and n2p2. None of these are installed in CI,
so the fixtures here replace ``which`` and ``subprocess`` inside a module with fakes. A test
supplies a handler that receives the argument list of each "process", writes whatever output
files the real binary would have produced into the current directory, and returns a
``(returncode, stdout, stderr)`` tuple (or ``None`` for a clean exit with no output).
"""

from __future__ import annotations

import subprocess
import types
from pathlib import Path

import pytest
from monty.serialization import loadfn

TEST_DIR = Path(__file__).parent


class CallLog(list):
    """List of launched argument lists. ``files[i]`` maps the names of the text files present in
    the working directory when call ``i`` was launched to their contents (the calculators run
    inside throw-away scratch directories, so inputs cannot be inspected afterwards)."""

    def __init__(self):
        super().__init__()
        self.files: list[dict[str, str]] = []

    def file(self, name):
        """Content of the most recent snapshot of file ``name``."""
        return next(files[name] for files in reversed(self.files) if name in files)


def _snapshot_cwd():
    snapshot = {}
    for path in Path().iterdir():
        if path.is_file() and path.stat().st_size < 1_000_000:
            try:
                snapshot[path.name] = path.read_text()
            except UnicodeDecodeError:
                continue
    return snapshot


def _make_fake_subprocess(handler, calls):
    class FakePopen:
        def __init__(self, args, stdout=None, stderr=None, **kwargs):
            self.args = [str(a) for a in args]
            calls.append(self.args)
            calls.files.append(_snapshot_cwd())
            result = handler(self.args) or (0, "", "")
            self.returncode, out, err = result
            if stdout is subprocess.PIPE:
                self._stdout = out.encode()
            else:
                if stdout is not None:
                    stdout.write(out)
                self._stdout = None
            self._stderr = err.encode() if stderr is subprocess.PIPE else None

        def communicate(self):
            return self._stdout, self._stderr

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return types.SimpleNamespace(Popen=FakePopen, PIPE=subprocess.PIPE)


@pytest.fixture
def mock_binaries(monkeypatch):
    """Install fake ``which``/``subprocess`` in a module.

    Usage: ``calls = mock_binaries(module, handler, available=None)``. ``available`` restricts
    which executable names ``which`` reports as found (default: all). ``calls`` collects the
    argument list of every launched process.
    """

    def install(module, handler=None, available=None):
        calls = CallLog()
        monkeypatch.setattr(module, "subprocess", _make_fake_subprocess(handler or (lambda args: None), calls))

        def fake_which(cmd):
            if not isinstance(cmd, str):  # mirror shutil.which, which rejects None
                raise TypeError(f"expected str, not {type(cmd).__name__}")
            if available is not None and cmd not in available:
                return None
            return f"/fake/bin/{cmd}"

        monkeypatch.setattr(module, "which", fake_which)
        return calls

    return install


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    """Run the test inside a fresh temporary directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(scope="session")
def datapool():
    """The small Mo training set shipped with the tests."""
    return loadfn(TEST_DIR / "datapool.json")


@pytest.fixture
def train_data(datapool):
    """(structures, energies, forces, stresses) for the first three entries of the data pool."""
    pool = datapool[:3]
    return (
        [d["structure"] for d in pool],
        [d["outputs"]["energy"] for d in pool],
        [d["outputs"]["forces"] for d in pool],
        [d["outputs"]["virial_stress"] for d in pool],
    )
