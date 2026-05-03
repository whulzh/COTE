# -*- coding: utf-8 -*-
"""Compatibility facade for the paper-aligned COTE implementation."""

from __future__ import annotations

from cote_paper import (
    TARGET_WIN_RATE,
    PaperCOTEAgent,
    metrics_reference,
)


CoteAgent = PaperCOTEAgent

__all__ = [
    "CoteAgent",
    "TARGET_WIN_RATE",
    "metrics_reference",
]
