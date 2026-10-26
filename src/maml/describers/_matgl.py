"""MatGL-based describers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matgl
import numpy as np
import pandas as pd
import torch
from huggingface_hub import try_to_load_from_cache
from matgl.config import MATGL_CACHE
from matgl.data.transformer import LogTransformer, Normalizer
from matgl.utils.io import HF_MATGL_ORG

from maml.base import BaseDescriber, describer_type

if TYPE_CHECKING:
    from pymatgen.core import Molecule, Structure

DEFAULT_MODEL = "M3GNet-Eform-MP-2018.6.1"
_MODEL_FILES = ("model.pt", "state.pt", "model.json")

# Property models (e.g. the default M3GNet-Eform) are serialized as a TransformedTargetModel whose
# ``model.pt`` pickles a target transformer. matgl loads checkpoints with ``torch.load(weights_only=True)``
# but does not allowlist these classes, so they are allowlisted here while loading.
_SAFE_GLOBALS: list = [Normalizer, LogTransformer]


def _local_model_dir(path: str) -> Path | None:
    """Return the local directory holding the files of a MatGL model without querying the Hub.

    Args:
        path: Local model directory, or a model name / repo id already in matgl's Hugging Face cache.

    Returns:
        The directory, or None if the model is not available locally.
    """
    if all((Path(path) / fn).exists() for fn in _MODEL_FILES):
        return Path(path)
    repo_id = path if "/" in path else f"{HF_MATGL_ORG}/{path}"
    try:
        cached = [try_to_load_from_cache(repo_id, fn, cache_dir=str(MATGL_CACHE)) for fn in _MODEL_FILES]
    except ValueError:  # not a valid repo id, e.g. a missing local path
        return None
    if all(isinstance(f, str) for f in cached):
        return Path(str(cached[0])).parent
    return None


def _restore_set2set_weights(model: torch.nn.Module, path: str) -> None:
    """Load Set2Set readout LSTM weights that matgl silently skips.

    Checkpoints converted from the DGL backend store the Set2Set LSTM as ``*.readout.set2set.lstm.*``,
    while matgl 4.x's PyG ``Set2SetReadOut`` names it ``*.readout.lstm.*``. matgl loads state dicts with
    ``strict=False``, so these weights are left at their random initialization and the readout (hence
    ``MatGLStructure`` features) changes every time the model is loaded. Remap and load them here.
    This is a no-op for checkpoints without such keys.
    """
    # matgl.load_model has just resolved (and if needed downloaded) the files; avoid a second Hub query.
    model_dir = _local_model_dir(path)
    if model_dir is None:
        return
    state = torch.load(model_dir / "state.pt", map_location="cpu", weights_only=True)
    own_keys = model.state_dict().keys()
    remapped = {}
    for key, value in state.items():
        new_key = key.replace(".set2set.lstm.", ".lstm.")
        if new_key != key and new_key in own_keys:
            remapped[new_key] = value
    if remapped:
        model.load_state_dict(remapped, strict=False)


def _load_matgl_model(model_path: str | None):
    """Load a MatGL model, defaulting to the M3GNet formation energy model.

    Wrapped models (e.g. TransformedTargetModel, Potential) are unwrapped to the underlying graph model.
    """
    path = model_path if model_path else DEFAULT_MODEL
    with torch.serialization.safe_globals(_SAFE_GLOBALS):
        loaded = matgl.load_model(path)
    if isinstance(loaded, torch.nn.Module):
        _restore_set2set_weights(loaded, path)
    return getattr(loaded, "model", loaded), path


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
        self.describer_model, self.model_path = _load_matgl_model(model_path)
        super().__init__(**kwargs)

    def transform_one(self, structure: Structure | Molecule):
        """
        Transform structure/molecule objects into structural features.

        Args:
            structure (Structure/Molecule): target object structure or molecule
        Returns: M3GNet readout layer output as structural features.

        """
        self.describer_model.predict_structure(structure)
        results = self.describer_model.feature_dict
        return np.array(torch.squeeze(results["readout"]).detach().cpu().numpy())


@describer_type("site")
class MatGLSite(BaseDescriber):
    """Use MatGL pre-trained models as featurizer to get atomic features."""

    def __init__(
        self,
        model_path: str | None = None,
        output_layers: list[str] | None = None,
        return_type: type | dict = pd.DataFrame,
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
                Pass an instance of ``dict`` to receive a per-layer dictionary instead.
            **kwargs: Pass through to BaseDescriber. E.g., feature_batch="pandas_concat" is very useful (see test).
        """
        self.describer_model, self.model_path = _load_matgl_model(model_path)

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
        Transform structure/molecule objects into atom features.

        Args:
            structure (Structure/Molecule): target object structure or molecule
        Returns: M3GNet node features as atom features.

        """
        self.describer_model.predict_structure(structure)
        result_dict = self.describer_model.feature_dict
        atom_fea_dict = {layer: result_dict[layer]["node_feat"].detach().cpu().numpy() for layer in self.output_layers}

        if isinstance(self.return_type, dict):
            return atom_fea_dict
        return pd.DataFrame(np.concatenate(list(atom_fea_dict.values()), axis=1))
