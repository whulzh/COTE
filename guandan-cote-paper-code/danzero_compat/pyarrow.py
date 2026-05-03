# -*- coding: utf-8 -*-
"""Compatibility shim for DanZero_plus' old pyarrow API.

The DanZero_plus win-test clients were written for pyarrow 5, where
``pyarrow.serialize(obj).to_buffer()`` and ``pyarrow.deserialize(bytes)``
were available. New pyarrow releases removed those helpers. The clients and
the local actor only exchange Python objects over localhost ZMQ, so pickle is
enough here and keeps the original client code untouched.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any


@dataclass
class _Serialized:
    payload: bytes

    def to_buffer(self) -> bytes:
        return self.payload


def serialize(obj: Any) -> _Serialized:
    return _Serialized(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def deserialize(data: Any) -> Any:
    if hasattr(data, "to_pybytes"):
        data = data.to_pybytes()
    return pickle.loads(bytes(data))
