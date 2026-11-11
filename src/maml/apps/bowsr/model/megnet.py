"""megnet model wrapper implementation."""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

import numpy as np
from monty.dev import requires
from pymatgen.core.periodic_table import Element

from maml.apps.bowsr.model.base import EnergyModel

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

try:
    import megnet
    from megnet.data.crystal import CrystalGraph
    from megnet.data.graph import GaussianDistance
    from megnet.models import MEGNetModel
except Exception:
    megnet = None
    MEGNetModel = None
    CrystalGraph = None
    GaussianDistance = None
    warnings.warn("megnet module should be installed to use this model type!")

module_dir = os.path.dirname(__file__)
model_filename = os.path.join(module_dir, "model_files", "megnet", "formation_energy.hdf5")


@requires(megnet is not None, "megnet package needs to be installed to use this module")
class MEGNet(EnergyModel):
    """MEGNetModel wrapper."""

    def __init__(self, model=None, reconstruct=False, **kwargs):
        """

        Args:
            model:  MEGNet energy model.
            reconstruct: Whether to reconstruct the model (used in
                disordered model).
            **kwargs: "gaussian_cutoff", "radius_cutoff", and "npass".
        """
        model = model or MEGNetModel.from_file(model_filename)
        gaussian_cutoff = kwargs.get("gaussian_cutoff", 6)
        radius_cutoff = kwargs.get("radius_cutoff", 5)
        npass = kwargs.get("npass", 2)
        self.reconstruct = reconstruct
        weights = model.get_weights()
        self.embedding = weights[0]
        if reconstruct:
            cg = CrystalGraph(
                bond_converter=GaussianDistance(np.linspace(0, gaussian_cutoff, 100), 0.5),
                cutoff=radius_cutoff,
            )
            model_new = MEGNetModel(100, 2, 16, npass=npass, graph_converter=cg)
            model_new.set_weights(weights[1:])
            self.model = model_new
        else:
            self.model = model

    def predict_energy(self, structure: Structure) -> float:
        """
        Predict energy from structure
        Args:
            structure: (pymatgen Structure).

        Returns: float
        """
        if not structure.is_ordered and not self.reconstruct:
            raise ValueError("To predict properties of disordered structure, please set reconstruct=True")

        if self.reconstruct:
            species = [dict(site.species.as_dict()) for site in structure]
            structure_copy = structure.copy()
            structure_copy[:] = ""  # dummy variable
            graph = self.model.graph_converter.convert(structure_copy)

            atom = []
            for d in species:
                f = np.zeros(16)
                for el, amt in d.items():
                    f += self.embedding[Element(el).number] * amt
                atom.append(f)
            graph["atom"] = atom

            energy = self.model.predict_graph(graph)[0]
        else:
            energy = self.model.predict_structure(structure)[0]

        return energy
