from __future__ import annotations

import pytest

import maml.apps.gbe.presetfeatures as preset


def test_my_quant_properties():
    quant = preset.my_quant("x", r"$x$", "eV")
    assert (quant.name, quant.latex, quant.unit) == ("x", r"$x$", "eV")
    bare = preset.my_quant("y")
    assert bare.latex is None
    assert bare.unit is None


@pytest.mark.parametrize(
    ("quant", "name", "latex", "unit"),
    [
        (preset.e_coh, "e_coh", r"$E_{coh}$", "J"),
        (preset.a0, "a_0", r"$a_{0}$", r"$\AA$"),
        (preset.cos_theta, "cos(theta)", r"cos($\theta$)", None),
        (preset.mean_delta_bl, "mean_delta_bl", r"$\bar{{\Delta} BL}$", r"$\AA$"),
        (preset.e_gb, "e_gb", r"$E_{GB}$", r"J m$^{-2}$"),
        (preset.bdensity, "Density", r"$Density$", r"g cm$^{-3}$"),
    ],
)
def test_preset_quants(quant, name, latex, unit):
    assert quant.name == name
    assert quant.latex == latex
    assert quant.unit == unit


def test_str2lat_covers_feature_list():
    assert set(preset.features) <= set(preset.str2lat)
