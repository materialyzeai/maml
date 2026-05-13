"""Compositional describers."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element, Species, Structure
from sklearn.decomposition import PCA, KernelPCA

from maml.base import BaseDescriber, describer_type
from maml.utils import Stats, get_full_stats_and_funcs, to_composition

from ._matminer import wrap_matminer_describer

CWD = os.path.abspath(os.path.dirname(__file__))

DATA_MAPPING = {
    "megnet_1": "data/elemental_embedding_1MEGNet_layer.json",
    "megnet_3": "data/elemental_embedding_3MEGNet_layer.json",
}

for length in [2, 3, 4, 8, 16, 32]:
    DATA_MAPPING[f"megnet_l{length}"] = f"data/elemental_embedding_1MEGNet_layer_length_{length}.json"
    DATA_MAPPING[f"megnet_ion_l{length}"] = f"data/ion_embedding_1MEGNet_layer_length_{length}.json"


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

        n_property = list({len(i) for i in properties})
        if len(n_property) > 1:
            raise ValueError("Property length not consistent")
        n_single_property = n_property[0]

        if property_names is None:
            property_names = [f"p{i}" for i in range(n_single_property)]

        if len(property_names) != n_single_property:
            raise ValueError("Property name length is not consistent")

        all_property_names = []

        if stats is None:
            stats = ["mean", "max", "min", "range", "std", "mode"]

        full_stats, stats_func = get_full_stats_and_funcs(stats)
        for stat in full_stats:
            all_property_names.extend([f"{p}_{stat}" for p in property_names])
        self.stats = full_stats
        self.element_properties = element_properties
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
        element_n_dict = {str(i): j for i, j in comp._data.items()}
        # it is more stable when element fraction is extremely small
        # Previously, this was `comp.to_data_dict['unit_cell_composition']`

        data = []
        weights = []
        for i, j in element_n_dict.items():
            data.append(self.element_properties[i])
            weights.append(j)

        data = list(zip(*data, strict=False))
        features = []
        for stat in self.stats_func:
            for d in data:
                f = stat(d, weights)
                features.append(f)
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

        property_names = d.get("property_names", None)
        element_properties = d.get("element_properties") if "element_properties" in d else d
        is_element = _keys_are_elements(element_properties)
        if not is_element:
            raise ValueError("File is not in correct format")

        if "stats" in d:
            stats = d.get("stats")

        return cls(element_properties=element_properties, property_names=property_names, stats=stats, **kwargs)

    @classmethod
    def from_data(cls, data_name: list[str] | str, stats: list[str] | None = None, **kwargs) -> ElementStats:
        """
        ElementalStats from existing data file.

        Args:
            data_name (str of list of str): data name. Current supported data are
                available from ElementStats.AVAILABLE_DATA
            stats (list): list of stats, use ElementStats.ALLOWED_STATS to
                check available stats
            **kwargs: Passthrough to class init.

        Returns: ElementStats instance
        """
        if isinstance(data_name, str):
            if data_name not in ElementStats.AVAILABLE_DATA:
                raise ValueError(f"Data name not found in the list {ElementStats.AVAILABLE_DATA!s}")

            filename = os.path.join(CWD, DATA_MAPPING[data_name])
            return cls.from_file(filename, stats=stats, **kwargs)

        if isinstance(data_name, list) and len(data_name) == 1:
            return cls.from_data(data_name[0], stats=stats, **kwargs)

        property_names = []
        instances = []
        for data_name_ in data_name:
            instance = cls.from_data(data_name_, stats=stats)
            instances.append(instance)

        elements = [set(i.element_properties.keys()) for i in instances]

        common_keys = elements[0]
        for e in elements[1:]:
            common_keys.intersection_update(e)

        element_properties: dict = {i: [] for i in common_keys}

        for index, instance in enumerate(instances):
            for k in common_keys:
                element_properties[k].extend(instance.element_properties[k])

            property_names.extend([f"{index}_{i}" for i in instance.property_names])

        return cls(element_properties=element_properties, property_names=property_names, stats=stats, **kwargs)

    @staticmethod
    def _reduce_dimension(
        element_properties,
        property_names,
        num_dim: int | None = None,
        reduction_algo: str | None = "pca",
        reduction_params: dict | None = None,
    ) -> tuple[dict, list[str]]:
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
        value_array = []
        p_keys = []
        for i, j in element_properties.items():
            value_array.append(j)
            p_keys.append(i)
        value_np_array = np.array(value_array)

        if reduction_algo == "pca":
            m = PCA(n_components=num_dim, **reduction_params)
            property_names = [f"pca_{i}" for i in range(num_dim)]
        elif reduction_algo == "kpca":
            m = KernelPCA(n_components=num_dim, **reduction_params)
            property_names = [f"kpca_{i}" for i in range(num_dim)]
        else:
            raise ValueError("Reduction algorithm not available")

        transformed_values = m.fit_transform(value_np_array)

        for key, value_list in zip(p_keys, transformed_values, strict=False):
            element_properties[key] = value_list.tolist()
        return element_properties, property_names


def _keys_are_elements(dic: dict) -> bool:
    keys = list(dic.keys())

    return all(_is_element_or_specie(key) for key in keys)


def _is_element_or_specie(s: str) -> bool:
    if s in ["D", "D+", "D-", "T"]:
        return True
    try:
        _ = Element(s)
    except ValueError:
        try:
            _ = Species.from_str(s)
        except ValueError:
            print(s)
            return False
    return True
