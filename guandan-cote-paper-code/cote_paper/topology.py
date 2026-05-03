# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from .config import NODE_COUNT
from .nodes import NODE_KEYS, OUTPUT_NODE


Edge = Tuple[str, str]


@dataclass
class CoteTopologyState:
    """COTE W in R^(8x8).

    The paper initializes all non-self-loop entries near 1/8 and then lets
    Phase D sparsify them. Self loops are represented in the 8x8 matrix but
    kept at zero because communication edges are j -> i between nodes.
    """

    weights: Dict[Edge, float] = field(default_factory=dict)
    generation: int = 0

    @classmethod
    def dense_initial(cls) -> "CoteTopologyState":
        weights: Dict[Edge, float] = {}
        for src in NODE_KEYS:
            for dst in NODE_KEYS:
                weights[(src, dst)] = 0.0 if src == dst else 1.0 / NODE_COUNT
        return cls(weights=weights)

    @classmethod
    def from_env(cls) -> "CoteTopologyState":
        init = os.environ.get("COTE_TOPOLOGY_INIT", "dense").strip().lower()
        if init != "dense":
            raise ValueError("Only dense topology initialization is available in the release code.")
        return cls.dense_initial()

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "CoteTopologyState":
        raw_weights = payload.get("weights", {})
        weights: Dict[Edge, float] = {(src, dst): 0.0 for src in NODE_KEYS for dst in NODE_KEYS}
        if isinstance(raw_weights, dict):
            for key, value in raw_weights.items():
                if isinstance(key, str) and "->" in key:
                    src, dst = key.split("->", 1)
                    if src in NODE_KEYS and dst in NODE_KEYS:
                        weights[(src, dst)] = float(value)
        return cls(weights=weights, generation=int(payload.get("generation", 0) or 0))

    def to_json(self) -> Dict[str, object]:
        return {
            "generation": self.generation,
            "weights": {f"{src}->{dst}": round(weight, 8) for (src, dst), weight in sorted(self.weights.items())},
        }

    def all_edges(self, include_self: bool = False) -> List[Edge]:
        edges: List[Edge] = []
        for src in NODE_KEYS:
            for dst in NODE_KEYS:
                if include_self or src != dst:
                    edges.append((src, dst))
        return edges

    def active_edges(self, threshold: float = 0.0) -> List[Edge]:
        return [edge for edge in self.all_edges() if self.weight(edge) > threshold]

    def inbound_edges(self, dst: str, threshold: float = 0.0) -> List[Edge]:
        return [(src, dst) for src in NODE_KEYS if src != dst and self.weight((src, dst)) > threshold]

    def outbound_edges(self, src: str, threshold: float = 0.0) -> List[Edge]:
        return [(src, dst) for dst in NODE_KEYS if src != dst and self.weight((src, dst)) > threshold]

    def weight(self, edge: Edge) -> float:
        return float(self.weights.get(edge, 0.0))

    def set_weight(self, edge: Edge, value: float) -> None:
        src, dst = edge
        self.weights[(src, dst)] = 0.0 if src == dst else max(0.0, min(1.0, float(value)))

    @property
    def retained_edge_count(self) -> int:
        return len(self.active_edges(threshold=0.0))

    @property
    def edge_retention(self) -> float:
        return self.retained_edge_count / float(NODE_COUNT * NODE_COUNT)

    def normalize_inbound(self) -> None:
        for dst in NODE_KEYS:
            inbound = [(src, dst) for src in NODE_KEYS if src != dst and self.weight((src, dst)) > 0.0]
            total = sum(self.weight(edge) for edge in inbound)
            if total <= 1e-12:
                continue
            for edge in inbound:
                self.set_weight(edge, self.weight(edge) / total)

    def apply_gradients(self, gradients: Mapping[Edge, float], eta_w: float, theta_grad: float) -> None:
        allow_edge_growth = os.environ.get("COTE_ALLOW_EDGE_GROWTH", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
            "",
        }
        normalize_after_update = os.environ.get("COTE_NORMALIZE_TOPOLOGY", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
            "",
        }
        for edge, grad in gradients.items():
            if self.weight(edge) <= 0.0 and not allow_edge_growth:
                continue
            clipped = max(-theta_grad, min(theta_grad, float(grad)))
            self.set_weight(edge, self.weight(edge) + eta_w * clipped)
        if normalize_after_update:
            self.normalize_inbound()
        self.generation += 1

    def prune(self, threshold: float) -> None:
        normalize_after_update = os.environ.get("COTE_NORMALIZE_TOPOLOGY", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
            "",
        }
        for edge in self.all_edges():
            if self.weight(edge) < threshold:
                self.set_weight(edge, 0.0)
        if normalize_after_update:
            self.normalize_inbound()

    def aggregate_node_scores(self, node_scores: Mapping[str, float], dst: str = OUTPUT_NODE) -> float:
        direct = float(node_scores.get(dst, 0.0))
        incoming = self.inbound_edges(dst)
        if not incoming:
            return direct
        weighted = 0.0
        norm = 0.0
        for src, _ in incoming:
            weight = self.weight((src, dst))
            weighted += weight * float(node_scores.get(src, 0.0))
            norm += weight
        return 0.60 * direct + 0.40 * (weighted / max(norm, 1e-9))

    def top_edges(self, limit: Optional[int] = None) -> List[Tuple[Edge, float]]:
        edges = [item for item in self.weights.items() if item[0][0] != item[0][1] and item[1] > 0.0]
        edges.sort(key=lambda item: item[1], reverse=True)
        return edges[:limit] if limit is not None else edges

    def runtime_summary(self) -> Dict[str, object]:
        return {
            "current_edge_retention": round(self.edge_retention, 4),
            "current_edge_count": self.retained_edge_count,
            "top_edges": [
                {"edge": f"{src}->{dst}", "weight": round(weight, 4)}
                for (src, dst), weight in self.top_edges(limit=8)
            ],
        }


def load_topology(path: str) -> Optional[CoteTopologyState]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        topology_payload = payload.get("topology", payload)
        if isinstance(topology_payload, dict):
            return CoteTopologyState.from_json(topology_payload)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def edge_to_key(edge: Edge) -> str:
    return f"{edge[0]}->{edge[1]}"


def key_to_edge(key: str) -> Edge:
    src, dst = key.split("->", 1)
    return src, dst


def active_edge_keys(edges: Iterable[Edge]) -> List[str]:
    return [edge_to_key(edge) for edge in edges]
