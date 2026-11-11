"""Tests for the abstract PES base classes."""

from __future__ import annotations

import pytest

from maml.apps.pes import LMPStaticCalculator, PotentialMixin


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("train", ([], [], [], [])),
        ("evaluate", ([], [], [], [])),
        ("write_param", ()),
        ("from_config", ()),
    ],
)
def test_potential_mixin_abstract_methods_raise(method, args):
    with pytest.raises(NotImplementedError):
        getattr(PotentialMixin, method)(object(), *args)


def test_potential_mixin_predict_efs_is_abstract():
    assert PotentialMixin.predict_efs(object(), None) is None


@pytest.mark.parametrize(("method", "args"), [("_setup", ()), ("_sanity_check", (None,)), ("_parse", ())])
def test_lmp_static_calculator_abstract_methods(method, args):
    assert getattr(LMPStaticCalculator, method)(object(), *args) is None
