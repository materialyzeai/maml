"""MAML data base classes."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class BaseDataSource:
    """Abstract base class for a data source."""

    @abc.abstractmethod
    def get(self, *args, **kwargs) -> pd.DataFrame:
        """Get data from sources."""
