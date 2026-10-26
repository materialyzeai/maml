from __future__ import annotations

import importlib
import sys

import maml.utils._jit as jit_module


def test_njit_fallback_without_numba(monkeypatch):
    monkeypatch.setitem(sys.modules, "numba", None)  # makes `import numba` raise ImportError
    try:
        reloaded = importlib.reload(jit_module)

        def add(a, b):
            return a + b

        assert reloaded.njit(add) is add
    finally:
        monkeypatch.undo()
        importlib.reload(jit_module)
