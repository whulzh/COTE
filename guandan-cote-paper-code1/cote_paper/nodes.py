# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class NodeSpec:
    key: str
    short: str
    layer: str
    role: str


NODE_SPECS: List[NodeSpec] = [
    NodeSpec("T1_board_parser", "T1", "perception", "current-board parser"),
    NodeSpec("T2_history_tracker", "T2", "perception", "history-action tracker"),
    NodeSpec("T3_card_counter", "T3", "inference", "card counter and hidden-card belief"),
    NodeSpec("T4_opponent_intent", "T4", "inference", "opponent-intent estimator"),
    NodeSpec("T5_teammate_intent", "T5", "inference", "teammate-intent estimator"),
    NodeSpec("T6_macro_evaluator", "T6", "strategy", "macro evaluator"),
    NodeSpec("T7_hand_value", "T7", "strategy", "hand-value evaluator"),
    NodeSpec("T8_action_decider", "T8", "output", "final action decision maker"),
]

NODE_KEYS = [node.key for node in NODE_SPECS]
NODE_BY_KEY: Dict[str, NodeSpec] = {node.key: node for node in NODE_SPECS}
NODE_INDEX: Dict[str, int] = {node.key: idx for idx, node in enumerate(NODE_SPECS)}

PERCEPTION_NODES = ("T1_board_parser", "T2_history_tracker")
INFERENCE_NODES = ("T3_card_counter", "T4_opponent_intent", "T5_teammate_intent")
STRATEGY_NODES = ("T6_macro_evaluator", "T7_hand_value")
OUTPUT_NODE = "T8_action_decider"


def node_name(index: int) -> str:
    return NODE_KEYS[index]


def node_index(name: str) -> int:
    return NODE_INDEX[name]

