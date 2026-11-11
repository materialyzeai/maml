"""Get data from url."""

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

from maml.base import BaseDataSource


class URLSource(BaseDataSource):
    """Load raw data from a URL, e.g., figshare."""

    def __init__(self, fmt: str = "csv", read_kwargs=None):
        """

        Args:
            fmt: The format to read the raw data. It should be something supported by pandas.read_<something>.
            Defaults to "csv".
            read_kwargs: kwargs passed to the read_<something> command.
        """
        self.fmt = fmt
        self.read_kwargs = read_kwargs or {}

    def get(self, url: str) -> pd.DataFrame:  # type: ignore
        """
        Get url data source.

        Args:
            url: URL to obtain raw data from.

        Returns:
            pd.DataFrame
        """
        raw = requests.get(url).text
        read_func = getattr(pd, f"read_{self.fmt}")
        return read_func(StringIO(raw), **self.read_kwargs)


class FigshareSource(URLSource):
    """Load data from figshare."""

    def get(self, file_id: str) -> pd.DataFrame:  # type: ignore
        """
        Get data from Figshare
        Args:
            file_id: file id.

        Returns:
            data frame
        """
        url = f"https://ndownloader.figshare.com/files/{file_id}"
        return super().get(url)
