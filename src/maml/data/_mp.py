"""Materials Project DataSource."""

from __future__ import annotations

import pandas as pd
from mp_api.client import MPRester

from maml.base import BaseDataSource


class MaterialsProject(BaseDataSource):
    """Query the Materials Project for Data."""

    def __init__(self, api_key: str | None = None):
        """
        Args:
            api_key (str or None): Materials Project API key.
        """
        self.mpr = MPRester(api_key=api_key)

    def get(self, criteria: str | dict, properties: list[str]) -> pd.DataFrame:  # type: ignore
        """
        Args:
            criteria (str or dict): Criteria for query
            properties (list): Properties to be queried.

        Returns:
            pandas DataFrame
        """
        data = self.mpr.materials.summary.search(**criteria, fields=properties)  # type: ignore[attr-defined,arg-type]
        data = [{k: v for k, v in dict(d).items() if k in properties} for d in data]
        return pd.DataFrame(data)
