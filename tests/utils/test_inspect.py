from __future__ import annotations

import pytest
from pymatgen.core import Structure

from maml.utils._inspect import get_full_args, get_param_types


class _Klass:
    def method(self, a: int, b: str | None = None, *, c=1):
        pass


def _real_annotations(x, y, z):
    pass


# Evaluated (non-string) annotations, as produced without `from __future__ import annotations`.
_real_annotations.__annotations__ = {"x": Structure, "y": float | None}


def test_get_full_args_skips_self():
    assert get_full_args(_Klass().method) == ["a", "b", "c"]
    assert get_full_args(_Klass.method) == ["a", "b", "c"]


def test_get_param_types_real_annotations():
    assert get_param_types(_real_annotations) == {"x": "Structure", "y": "float", "z": "_empty"}


def test_get_param_types_string_annotations():
    """Regression: PEP 563 string annotations raised AttributeError ('str' has no __name__)."""
    assert get_param_types(_Klass().method) == {"a": "int", "b": "str", "c": "_empty"}


@pytest.mark.parametrize("annotation", ["Structure", "Structure | None", " Structure|Molecule"])
def test_get_param_types_string_union(annotation):
    def func(obj):
        pass

    func.__annotations__ = {"obj": annotation}
    assert get_param_types(func) == {"obj": "Structure"}
