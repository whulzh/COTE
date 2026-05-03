# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from .nodes import NODE_KEYS


@dataclass
class StrictModelPool:
    """Registry for heterogeneous paper-style node/edge model backends."""

    shared_backend: Any = None
    node_backends: Dict[str, Any] = field(default_factory=dict)
    edge_backends: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, shared_backend: Any = None) -> "StrictModelPool":
        node_backends: Dict[str, Any] = {}
        raw = os.environ.get("COTE_NODE_MODEL_MAP", "")
        if raw:
            for item in raw.split(","):
                if "=" not in item:
                    continue
                node, value = item.split("=", 1)
                node = node.strip()
                if node in NODE_KEYS:
                    node_backends[node] = value.strip()
        return cls(shared_backend=shared_backend, node_backends=node_backends)

    def node_backend(self, node: str) -> Any:
        return self.node_backends.get(node, self.shared_backend)

    def edge_backend(self, edge_key: str) -> Any:
        return self.edge_backends.get(edge_key, self.shared_backend)

    def runtime_summary(self) -> Mapping[str, object]:
        return {
            "heterogeneous_nodes": sorted(self.node_backends),
            "heterogeneous_edges": sorted(self.edge_backends),
            "shared_backend": self.shared_backend is not None,
        }
