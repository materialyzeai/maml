"""Module defines feature objects for GB energy model."""

from __future__ import annotations


class my_quant:
    """An object to describe GB quantities."""

    def __init__(self, str_name: str, latex_name: str | None = None, latex_unit: str | None = None):
        """

        Args:
            str_name (str): string rep of the quant
            latex_name (): latex rep of the quant
            latex_unit (): latex rep of the quant unit.
        """
        self.str_name = str_name
        self.latex_name = latex_name
        self.latex_unit = latex_unit

    @property
    def name(self):
        """

        Returns:
            string rep of the quant.
        """
        return self.str_name

    @property
    def latex(self):
        """

        Returns:
            latex rep of the quant.
        """
        return self.latex_name

    @property
    def unit(self):
        """

        Returns:
            latex rep of the quant unit.
        """
        return self.latex_unit


initial_fnames = [
    r"$d_{GB}$",
    r"$d_{rot}$",
    r"cos($\theta$)",
    r"$E_{coh}$",
    r"$\widebar{{\Delta} BL}$",
    r"$G$",
    r"$a_{0}$",
    r"$\gamma_{GB}^{DFT}$",
]
str2lat = {
    "d_gb": r"$d_{GB}$",
    "d_rot": r"$d_{rot}$",
    "sin(theta)": r"sin($\theta$)",
    "cos(theta)": r"cos($\theta$)",
    "e_coh": r"$E_{coh}$",
    "G": r"$G$",
    "a_0": r"$a_{0}$",
    "a_r": r"$a_{r}$",
    "mean_delta_bl": r"$\bar{{\Delta} BL}$",
    "mean_bl": r"$\bar{BL}$",
    "Density": "Density",
    "e_gb": r"$\gamma_{GB}$",
}
features = ["d_gb", "d_rot", "cos(theta)", "mean_delta_bl", "G", "mean_bl"]

e_coh = my_quant("e_coh", str2lat["e_coh"], "J")
a0 = my_quant("a_0", str2lat["a_0"], r"$\AA$")
ar = my_quant("a_r", str2lat["a_r"], r"$\AA$")
cos_theta = my_quant("cos(theta)", str2lat["cos(theta)"])
sin_theta = my_quant("sin(theta)", str2lat["sin(theta)"])
d_gb = my_quant("d_gb", str2lat["d_gb"])
d_rot = my_quant("d_rot", str2lat["d_rot"])
mean_delta_bl = my_quant("mean_delta_bl", str2lat["mean_delta_bl"], r"$\AA$")
G = my_quant("G", str2lat["G"], r"J m$^{-3}$")
e_gb = my_quant("e_gb", r"$E_{GB}$", r"J m$^{-2}$")
hb = my_quant("HB", "HB", "")
bdensity = my_quant("Density", r"$Density$", r"g cm$^{-3}$")
CLTE = my_quant("CLTE", r"$CLTE$", "")
mean_bl = my_quant("mean_bl", str2lat["mean_bl"], r"$\AA$")
