"""Compositional describers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element, Species, Structure
from sklearn.decomposition import PCA, KernelPCA

from maml.base import BaseDescriber, describer_type
from maml.utils import Stats, get_full_stats_and_funcs, to_composition

from ._matminer import wrap_matminer_describer

CWD = Path(__file__).parent

DATA_MAPPING: dict[str, str] = {
    "megnet_1": "data/elemental_embedding_1MEGNet_layer.json",
    "megnet_3": "data/elemental_embedding_3MEGNet_layer.json",
}

for _length in (2, 3, 4, 8, 16, 32):
    DATA_MAPPING[f"megnet_l{_length}"] = f"data/elemental_embedding_1MEGNet_layer_length_{_length}.json"
    DATA_MAPPING[f"megnet_ion_l{_length}"] = f"data/ion_embedding_1MEGNet_layer_length_{_length}.json"


try:
    from matminer.featurizers.composition import (  # noqa
        ElementProperty as MatminerElementProperty,
    )

    ElementProperty = wrap_matminer_describer(
        "ElementProperty", MatminerElementProperty, to_composition, describer_type="composition"
    )

except ImportError:
    ElementProperty = None


@describer_type("composition")
class ElementStats(BaseDescriber):
    """
    Element statistics. The allowed stats are accessed via ALLOWED_STATS class
    attributes. If the stats have multiple parameters, the positional arguments
    are separated by ::, e.g., moment::1::None.
    """

    ALLOWED_STATS = Stats.allowed_stats  # type: ignore
    AVAILABLE_DATA = list(DATA_MAPPING.keys())

    def __init__(
        self,
        element_properties: dict,
        stats: list[str] | None = None,
        property_names: list[str] | None = None,
        feature_batch: str = "pandas_concat",
        **kwargs,
    ):
        """
        Elemental stats for composition/str/structure.

        Args:
            element_properties (dict): element properties, e.g.,
                {'H': [0.1, 0.2, 0.3], 'He': [0.12, ...]}
            stats (list): list of stats, check ElementStats.ALLOWED_STATS
                for supported stats. The stats that support additional
                Keyword args, use ':' to separate the args. For example,
                'moment:0:None' will calculate moment stats with order=0,
                and max_order=None.
            property_names (list): list of property names, has to be consistent
                in length with properties in element_properties
            feature_batch (str): way to batch a list of feature outputs into a single
                one
            **kwargs: optional parameters include
                num_dim (int): number of dimension to keep
                reduction_algo (str): dimensional reduction algorithm
                reduction_params (dict): kwargs for dimensional reduction algorithm
        """
        num_dim = kwargs.pop("num_dim", None)
        reduction_algo = kwargs.pop("reduction_algo", "pca")
        reduction_params = kwargs.pop("reduction_params", {})

        element_properties, property_names = self._reduce_dimension(
            element_properties=element_properties,
            property_names=property_names,
            num_dim=num_dim,
            reduction_algo=reduction_algo,
            reduction_params=reduction_params,
        )

        self.element_properties = element_properties
        properties = list(self.element_properties.values())

        property_lengths = {len(p) for p in properties}
        if len(property_lengths) > 1:
            raise ValueError("Property length not consistent")
        n_single_property = next(iter(property_lengths))

        if property_names is None:
            property_names = [f"p{i}" for i in range(n_single_property)]

        if len(property_names) != n_single_property:
            raise ValueError("Property name length is not consistent")

        if stats is None:
            stats = ["mean", "max", "min", "range", "std", "mode"]

        full_stats, stats_func = get_full_stats_and_funcs(stats)
        all_property_names = [f"{p}_{stat}" for stat in full_stats for p in property_names]

        self.stats = full_stats
        self.property_names = property_names
        self.n_features = len(property_names)
        self.all_property_names = all_property_names
        self.stats_func = stats_func
        super().__init__(feature_batch=feature_batch, **kwargs)

    def transform_one(self, obj: Structure | str | Composition) -> pd.DataFrame:
        """
        Transform one object, the object can be string, Compostion or Structure.

        Args:
            obj (str/Composition/Structure): object to transform

        Returns: pd.DataFrame with property names as column names

        """
        comp = to_composition(obj)
        # as_dict() keeps oxidation-state-aware species labels (e.g., 'Ti4+'),
        # matching the legacy behavior of reading Composition._data directly.
        element_n_dict = comp.as_dict()

        data = []
        weights = []
        for el, amount in element_n_dict.items():
            data.append(self.element_properties[el])
            weights.append(amount)

        # Transpose so each row corresponds to one property dimension across elements.
        property_rows = list(zip(*data, strict=True))
        features = [stat(row, weights) for stat in self.stats_func for row in property_rows]
        return pd.DataFrame([features], columns=self.all_property_names)

    @classmethod
    def from_file(cls, filename: str, stats: list[str] | None = None, **kwargs) -> ElementStats:
        """ElementStats from a json file of element property dictionary.

        The keys required are:

            element_properties
            property_names

        Args:
            filename (str): filename
            stats (list): list of stats, check ElementStats.ALLOWED_STATS
                for supported stats. The stats that support additional
                Keyword args, use ':' to separate the args. For example,
                'moment:0:None' will calculate moment stats with order=0,
                and max_order=None.
            **kwargs: Passthrough to class init.

        Returns: ElementStats class
        """
        with open(filename) as f:
            d = json.load(f)

        property_names = d.get("property_names")
        element_properties = d.get("element_properties", d)
        if not _keys_are_elements(element_properties):
            raise ValueError("File is not in correct format")

        if "stats" in d:
            stats = d.get("stats")

        return cls(element_properties=element_properties, property_names=property_names, stats=stats, **kwargs)

    @classmethod
    def from_data(cls, data_name: list[str] | str, stats: list[str] | None = None, **kwargs) -> ElementStats:
        """
        ElementalStats from existing data file(s).

        When a list of data names is provided, the element properties are
        concatenated across the shared element keys, and the resulting
        property names are prefixed with the index of each source.

        Args:
            data_name (str or list of str): data name(s). Current supported data are
                available from ElementStats.AVAILABLE_DATA
            stats (list): list of stats, use ElementStats.ALLOWED_STATS to
                check available stats
            **kwargs: Passthrough to class init.

        Returns: ElementStats instance
        """
        if isinstance(data_name, str):
            if data_name not in cls.AVAILABLE_DATA:
                raise ValueError(f"Data name not found in the list {cls.AVAILABLE_DATA!s}")

            filename = CWD / DATA_MAPPING[data_name]
            return cls.from_file(str(filename), stats=stats, **kwargs)

        if len(data_name) == 1:
            return cls.from_data(data_name[0], stats=stats, **kwargs)

        instances = [cls.from_data(name, stats=stats) for name in data_name]

        # Use elements common to all source instances.
        common_keys = set(instances[0].element_properties.keys())
        for instance in instances[1:]:
            common_keys.intersection_update(instance.element_properties.keys())

        element_properties: dict = {k: [] for k in common_keys}
        property_names: list[str] = []
        for index, instance in enumerate(instances):
            for k in common_keys:
                element_properties[k].extend(instance.element_properties[k])
            property_names.extend(f"{index}_{name}" for name in instance.property_names)

        return cls(element_properties=element_properties, property_names=property_names, stats=stats, **kwargs)

    @staticmethod
    def _reduce_dimension(
        element_properties: dict,
        property_names: list[str] | None,
        num_dim: int | None = None,
        reduction_algo: str = "pca",
        reduction_params: dict | None = None,
    ) -> tuple[dict, list[str] | None]:
        """
        Reduce the feature dimension by reduction_algo.

        Args:
            element_properties (dict): dictionary of elemental/specie propeprties
            property_names (list): list of property names
            num_dim (int): number of dimension to keep
            reduction_algo (str): algorithm for dimensional reduction, currently support
                pca, kpca
            reduction_params (dict): kwargs for reduction algorithm

        Returns: new element_properties and property_names

        """
        if num_dim is None:
            return element_properties, property_names

        reduction_params = reduction_params or {}
        p_keys = list(element_properties.keys())
        value_np_array = np.array([element_properties[k] for k in p_keys])

        if reduction_algo == "pca":
            model = PCA(n_components=num_dim, **reduction_params)
            property_names = [f"pca_{i}" for i in range(num_dim)]
        elif reduction_algo == "kpca":
            model = KernelPCA(n_components=num_dim, **reduction_params)
            property_names = [f"kpca_{i}" for i in range(num_dim)]
        else:
            raise ValueError("Reduction algorithm not available")

        transformed_values = model.fit_transform(value_np_array)
        element_properties = {key: row.tolist() for key, row in zip(p_keys, transformed_values, strict=True)}
        return element_properties, property_names


def _keys_are_elements(dic: dict) -> bool:
    return all(_is_element_or_specie(key) for key in dic)


def _is_element_or_specie(s: str) -> bool:
    if s in {"D", "D+", "D-", "T"}:
        return True
    try:
        Element(s)
    except ValueError:
        try:
            Species.from_str(s)
        except ValueError:
            return False
    return True
