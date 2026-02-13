from __future__ import annotations

from typing import TYPE_CHECKING

import matgl
import numpy as np
import pandas as pd
import torch

from maml.base import BaseDescriber, describer_type

if TYPE_CHECKING:
    from pymatgen.core import Molecule, Structure

DEFAULT_MODEL = "M3GNet-MP-2018.6.1-Eform"


@describer_type("structure")
class MatGLStructure(BaseDescriber):
    """Use M3GNet pre-trained models as featurizer to get Structural features."""

    def __init__(
        self,
        model_path: str | None = None,
        **kwargs,
    ):
        """

        Args:
            model_path (str): m3gnet models path. If no path is provided,
                the models will be M3GNet formation energy model on MatGL repo:
                https://github.com/materialsvirtuallab/matgl/tree/main/pretrained_models/M3GNet-MP-2018.6.1-Eform
                Please refer to the M3GNet paper:
                https://doi.org/10.1038/s43588-022-00349-3.
            **kwargs: Pass through to BaseDescriber.
        """
        if model_path:
            self.describer_model = matgl.load_model(model_path).model
            self.model_path = model_path
        else:
            self.describer_model = matgl.load_model(DEFAULT_MODEL).model
            self.model_path = DEFAULT_MODEL
        super().__init__(**kwargs)

    def transform_one(self, structure: Structure | Molecule):
        """
        Transform structure/molecule objects into structural features.

        Args:
            structure (Structure/Molecule): target object structure or molecule
        Returns: M3GNet readout layer output as structural features.

        """
        results = self.describer_model.predict_structure(structure, return_features=True)
        return np.array(torch.squeeze(results["readout"]).detach().cpu().numpy())


@describer_type("site")
class MatGLSite(BaseDescriber):
    """Use MatGL pre-trained models as featurizer to get atomic features."""

    def __init__(
        self,
        model_path: str | None = None,
        output_layers: list | None = None,
        return_type: list | dict = pd.DataFrame,
        **kwargs,
    ):
        """

        Args:
            model_path (str): m3gnet models path. If no path is provided,
                the models will be M3GNet formation energy model on MatGL repo:
                https://github.com/materialsvirtuallab/matgl/tree/main/pretrained_models/M3GNet-MP-2018.6.1-Eform
                Please refer to the M3GNet paper:
                https://doi.org/10.1038/s43588-022-00349-3.
            output_layers: List of names for the layer of GNN as output. Choose from "embedding", "gc_1", "gc_2",...,
                "gc_n", where n is the total number of graph convolutional layers. By default, the node features in
                "gc_1" layer are returned.
            return_type: The data type of the returned the atom features. By default, atom features in different
                output_layers are concatenated to one vector per atom, and a dataframe of vectors are returned.
            **kwargs: Pass through to BaseDescriber. E.g., feature_batch="pandas_concat" is very useful (see test).
        """
        if model_path:
            self.describer_model = matgl.load_model(model_path).model
            self.model_path = model_path
        else:
            self.describer_model = matgl.load_model(DEFAULT_MODEL).model
            self.model_path = DEFAULT_MODEL

        allowed_output_layers = ["embedding"] + [f"gc_{i + 1}" for i in range(self.describer_model.n_blocks)]
        if output_layers is None:
            output_layers = ["gc_1"]
        elif not isinstance(output_layers, list) or set(output_layers).difference(allowed_output_layers):
            raise ValueError(f"Invalid output_layers, it must be a sublist of {allowed_output_layers}.")
        self.output_layers = output_layers
        self.return_type = return_type
        super().__init__(**kwargs)

    def transform_one(self, structure: Structure | Molecule):
        """
        Transform structure/molecule objects into atom features
        Args:
            structure (Structure/Molecule): target object structure or molecule
        Returns: M3GNet node features as atom features.

        """
        result_dict = self.describer_model.predict_structure(
            structure, output_layers=self.output_layers, return_features=True
        )
        atom_fea_dict = {}
        for layer in self.output_layers:
            atom_fea_dict[layer] = result_dict[layer]["node_feat"].detach().cpu().numpy()

        if isinstance(self.return_type, dict):
            return atom_fea_dict
        return pd.DataFrame(np.concatenate(list(atom_fea_dict.values()), axis=1))
