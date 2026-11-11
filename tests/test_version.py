from __future__ import annotations

import importlib
import importlib.metadata

import maml


def test_version_matches_metadata():
    assert maml.__version__ == importlib.metadata.version("maml")


def test_version_missing_when_not_installed(monkeypatch):
    def _not_installed(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.delattr(maml, "__version__")
    monkeypatch.setattr(importlib.metadata, "version", _not_installed)
    try:
        importlib.reload(maml)
        assert not hasattr(maml, "__version__")
    finally:
        monkeypatch.undo()
        importlib.reload(maml)
    assert maml.__version__ == importlib.metadata.version("maml")
