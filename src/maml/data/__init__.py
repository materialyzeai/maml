"""Get data from various sources."""

from __future__ import annotations

from ._mp import MaterialsProject
from ._url import URLSource

__all__ = ["MaterialsProject", "URLSource"]
