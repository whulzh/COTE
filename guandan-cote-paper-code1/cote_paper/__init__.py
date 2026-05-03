# -*- coding: utf-8 -*-
"""Paper-aligned COTE reproduction for the Guandan offline platform."""

from .config import (
    TARGET_WIN_RATE,
    HyperParams,
)
from .policy import PaperCOTEAgent, metrics_reference

__all__ = [
    "HyperParams",
    "PaperCOTEAgent",
    "TARGET_WIN_RATE",
    "metrics_reference",
]
