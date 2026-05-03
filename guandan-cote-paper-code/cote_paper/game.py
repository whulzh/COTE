# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PASS_ACTION = ["PASS", "PASS", "PASS"]
RANKS_LOW_TO_HIGH = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2", "B", "R"]
BASE_RANK_VALUE = {rank: idx for idx, rank in enumerate(RANKS_LOW_TO_HIGH, start=3)}
ACTION_TYPE_BONUS = {
    "Single": 0.0,
    "Pair": 7.0,
    "Trips": 12.0,
    "ThreePair": 28.0,
    "ThreeWithTwo": 30.0,
    "TwoTrips": 28.0,
    "Straight": 26.0,
    "StraightFlush": 34.0,
    "Bomb": -22.0,
}


@dataclass(frozen=True)
class Candidate:
    index: int
    action: List[Any]
    base_score: float
    node_scores: Dict[str, float]
    probability: float
    reason: str


@dataclass
class GameContext:
    my_pos: int
    partner_pos: int
    opponents: Tuple[int, int]
    hand_cards: List[str]
    public_info: List[Dict[str, Any]]
    self_rank: str
    oppo_rank: str
    cur_rank: str
    cur_pos: Optional[int]
    cur_action: Any
    greater_pos: Optional[int]
    greater_action: Any
    stage: str

    @classmethod
    def from_message(cls, msg: Dict[str, Any], default_pos: int) -> "GameContext":
        my_pos = int(msg.get("myPos", default_pos))
        partner = (my_pos + 2) % 4
        opponents = tuple(pos for pos in range(4) if pos not in (my_pos, partner))
        return cls(
            my_pos=my_pos,
            partner_pos=partner,
            opponents=(opponents[0], opponents[1]),
            hand_cards=list(msg.get("handCards") or msg.get("handCard") or []),
            public_info=list(msg.get("publicInfo") or [{}, {}, {}, {}]),
            self_rank=str(msg.get("selfRank", "")),
            oppo_rank=str(msg.get("oppoRank", "")),
            cur_rank=str(msg.get("curRank", "")),
            cur_pos=msg.get("curPos"),
            cur_action=msg.get("curAction"),
            greater_pos=msg.get("greaterPos"),
            greater_action=msg.get("greaterAction"),
            stage=str(msg.get("stage", "")),
        )

    @property
    def hand_size(self) -> int:
        return len(self.hand_cards)

    @property
    def leading(self) -> bool:
        if self.stage != "play":
            return False
        return self.greater_pos in (None, -1) or self.greater_action in (None, -1, [None, None, None])

    @property
    def partner_winning(self) -> bool:
        return self.greater_pos == self.partner_pos

    @property
    def opponent_winning(self) -> bool:
        return self.greater_pos in self.opponents

    @property
    def min_opponent_rest(self) -> int:
        rests = [safe_rest(self.public_info, pos) for pos in self.opponents]
        rests = [rest for rest in rests if rest is not None]
        return min(rests) if rests else 27

    @property
    def partner_rest(self) -> int:
        rest = safe_rest(self.public_info, self.partner_pos)
        return rest if rest is not None else 27


def normalize_actions(raw_actions: Any) -> List[List[Any]]:
    if isinstance(raw_actions, list):
        return [list(action) if isinstance(action, (list, tuple)) else [action] for action in raw_actions]
    if isinstance(raw_actions, dict):
        flattened: List[List[Any]] = []
        for key, value in raw_actions.items():
            if isinstance(value, dict):
                for rank, items in value.items():
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, list):
                                flattened.append([key, rank, item])
                            else:
                                flattened.append([key, rank, [item]])
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, list):
                        flattened.append([key, item[0] if item else "", item])
                    else:
                        flattened.append([key, item])
        return flattened
    return []


def parse_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def safe_rest(public_info: Sequence[Dict[str, Any]], pos: int) -> Optional[int]:
    try:
        info = public_info[pos]
        if isinstance(info, dict) and "rest" in info:
            return int(info["rest"])
    except (IndexError, TypeError, ValueError):
        return None
    return None


def is_pass(action: Sequence[Any]) -> bool:
    return bool(action) and action[0] == "PASS"


def action_name(action: Sequence[Any]) -> str:
    return str(action[0]) if action else "UNKNOWN"


def action_rank(action: Sequence[Any]) -> str:
    if len(action) > 1 and isinstance(action[1], str):
        return action[1]
    cards = action_cards(action)
    return card_rank(cards[0]) if cards else ""


def action_cards(action: Any) -> List[str]:
    if isinstance(action, dict):
        cards = action.get("actions") or action.get("cards") or []
        return [str(card) for card in cards]
    if not isinstance(action, (list, tuple)):
        return []
    if len(action) >= 3 and isinstance(action[2], list):
        return [str(card) for card in action[2]]
    if len(action) >= 2 and isinstance(action[1], list):
        return [str(card) for card in action[1]]
    return [str(item) for item in action if isinstance(item, str) and len(item) >= 2 and item[0] in "SHCD"]


def is_bomb(action: Sequence[Any]) -> bool:
    name = action_name(action)
    if name in ("Bomb", "StraightFlush"):
        return True
    cards = action_cards(action)
    ranks = Counter(card_rank(card) for card in cards)
    return len(cards) >= 4 and len(ranks) == 1


def card_rank(card: str) -> str:
    return card[-1] if card else ""


def rank_value(rank: str, cur_rank: str = "") -> float:
    if not rank:
        return 0.0
    value = float(BASE_RANK_VALUE.get(rank, 0))
    if rank == cur_rank and rank not in ("B", "R"):
        value += 2.5
    return value


def card_value(card: str, cur_rank: str = "") -> float:
    return rank_value(card_rank(card), cur_rank)


def high_card_cost(cards: Iterable[str], cur_rank: str = "") -> float:
    cost = 0.0
    for card in cards:
        value = card_value(card, cur_rank)
        if value >= 15:
            cost += (value - 14) * 4.0
        elif value >= 12:
            cost += value - 11
    return cost


def residual_shape_score(hand_cards: Sequence[str], used_cards: Sequence[str], cur_rank: str = "") -> float:
    before = Counter(hand_cards)
    after = before.copy()
    for card in used_cards:
        if after[card] > 0:
            after[card] -= 1
            if after[card] <= 0:
                del after[card]
    return group_shape_score(list(after.elements()), cur_rank) - group_shape_score(list(before.elements()), cur_rank)


def estimated_turns(cards: Sequence[str], cur_rank: str = "") -> float:
    """Small, conservative hand-partition estimate for T7 hand value."""

    rank_counts = Counter(card_rank(card) for card in cards)
    turns = 0.0

    # Prefer consuming long natural sequences before counting isolated ranks.
    remaining = Counter(rank_counts)
    for min_count, min_len in ((3, 2), (2, 3), (1, 5)):
        while True:
            run = best_run(remaining, min_count=min_count, min_len=min_len)
            if not run:
                break
            for rank in run:
                remaining[rank] -= min_count
            turns += 1.0

    for rank, count in remaining.items():
        if count <= 0:
            continue
        if count >= 4:
            turns += 1.0
            count -= 4
        if count == 3:
            turns += 1.0
        elif count == 2:
            turns += 1.0
        elif count == 1:
            turns += 1.0
    high_single_penalty = sum(0.15 for card in cards if rank_value(card_rank(card), cur_rank) >= 15)
    return turns + high_single_penalty


def best_run(rank_counts: Counter, min_count: int, min_len: int) -> List[str]:
    usable = [rank for rank in RANKS_LOW_TO_HIGH[:-3] if rank_counts.get(rank, 0) >= min_count]
    best: List[str] = []
    current: List[str] = []
    previous_idx = None
    for rank in usable:
        idx = RANKS_LOW_TO_HIGH.index(rank)
        if previous_idx is None or idx == previous_idx + 1:
            current.append(rank)
        else:
            if len(current) > len(best):
                best = list(current)
            current = [rank]
        previous_idx = idx
    if len(current) > len(best):
        best = list(current)
    return best if len(best) >= min_len else []


def group_shape_score(cards: Sequence[str], cur_rank: str = "") -> float:
    rank_counts = Counter(card_rank(card) for card in cards)
    score = 0.0
    for rank, count in rank_counts.items():
        value = rank_value(rank, cur_rank)
        if count == 1:
            score -= 1.4
            if value >= 15:
                score += 1.8
        elif count == 2:
            score += 4.0
        elif count == 3:
            score += 9.0
        elif count >= 4:
            score += 18.0 + 3.0 * (count - 4)
    score += consecutive_score(rank_counts, min_count=1, min_len=5, weight=1.0)
    score += consecutive_score(rank_counts, min_count=2, min_len=3, weight=2.2)
    score += consecutive_score(rank_counts, min_count=3, min_len=2, weight=2.2)
    return score


def consecutive_score(rank_counts: Counter, min_count: int, min_len: int, weight: float) -> float:
    usable = [rank for rank in RANKS_LOW_TO_HIGH[:-3] if rank_counts.get(rank, 0) >= min_count]
    best = 0
    current = 0
    previous_idx = None
    for rank in usable:
        idx = RANKS_LOW_TO_HIGH.index(rank)
        if previous_idx is None or idx == previous_idx + 1:
            current += 1
        else:
            best = max(best, current)
            current = 1
        previous_idx = idx
    best = max(best, current)
    return weight * best if best >= min_len else 0.0


def softmax(values: Sequence[float], temperature: float = 1.0) -> List[float]:
    if not values:
        return []
    temp = max(temperature, 1e-6)
    shifted = [(value / temp) for value in values]
    max_value = max(shifted)
    exps = [math.exp(max(-60.0, min(60.0, value - max_value))) for value in shifted]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in exps]


def normalize_distribution(values: Sequence[float]) -> List[float]:
    clipped = [max(0.0, float(value)) for value in values]
    total = sum(clipped)
    if total <= 1e-12:
        return [1.0 / len(values)] * len(values) if values else []
    return [value / total for value in clipped]


def entropy(values: Sequence[float]) -> float:
    dist = normalize_distribution(values)
    return -sum(value * math.log(max(value, 1e-12)) for value in dist)


def kl_divergence(p_values: Sequence[float], q_values: Sequence[float]) -> float:
    p = normalize_distribution(p_values)
    q = normalize_distribution(q_values)
    size = min(len(p), len(q))
    return sum(p[idx] * math.log(max(p[idx], 1e-12) / max(q[idx], 1e-12)) for idx in range(size))


def blend_distribution(base: Sequence[float], signal: Sequence[float], weight: float) -> List[float]:
    base_dist = normalize_distribution(base)
    signal_dist = normalize_distribution(signal)
    size = min(len(base_dist), len(signal_dist))
    alpha = max(0.0, min(1.0, weight))
    return normalize_distribution([(1.0 - alpha) * base_dist[idx] + alpha * signal_dist[idx] for idx in range(size)])


def action_to_text(action: Sequence[Any]) -> str:
    return json.dumps(list(action), ensure_ascii=False, separators=(",", ":"))
