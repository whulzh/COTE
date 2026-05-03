# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Sequence

from .game import GameContext, action_cards, card_rank, entropy, normalize_distribution, safe_rest
from .nodes import NODE_KEYS


TOTAL_RANK_COUNTS = {
    "3": 8,
    "4": 8,
    "5": 8,
    "6": 8,
    "7": 8,
    "8": 8,
    "9": 8,
    "T": 8,
    "J": 8,
    "Q": 8,
    "K": 8,
    "A": 8,
    "2": 8,
    "B": 2,
    "R": 2,
}


@dataclass
class BeliefSnapshot:
    node_vectors: Dict[str, List[float]]
    public_vector: List[float]
    entropy: float
    rest_by_pos: List[int]
    unknown_rank_counts: Dict[str, int]


@dataclass
class BeliefTracker:
    my_pos: int = 0
    history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=256))
    seen_cards: Counter = field(default_factory=Counter)
    last_hand_cards: List[str] = field(default_factory=list)
    rest_by_pos: List[int] = field(default_factory=lambda: [27, 27, 27, 27])
    episode_index: int = 0

    def observe(self, msg: Mapping[str, Any]) -> None:
        if "myPos" in msg:
            self.my_pos = int(msg["myPos"])
        stage = msg.get("stage")
        msg_type = msg.get("type")

        if stage == "beginning" and msg_type == "notify":
            self.episode_index += 1
            self.history.clear()
            self.seen_cards.clear()
            self.rest_by_pos = [27, 27, 27, 27]
            self.last_hand_cards = list(msg.get("handCards") or msg.get("handCard") or [])
            self.seen_cards.update(self.last_hand_cards)
            return

        if "handCards" in msg or "handCard" in msg:
            self.last_hand_cards = list(msg.get("handCards") or msg.get("handCard") or [])

        public_info = msg.get("publicInfo")
        if isinstance(public_info, list):
            for pos in range(min(4, len(public_info))):
                rest = safe_rest(public_info, pos)
                if rest is not None:
                    self.rest_by_pos[pos] = rest

        if stage == "play" and msg_type == "notify":
            cur_pos = msg.get("curPos")
            cur_action = msg.get("curAction")
            cards = action_cards(cur_action)
            if cards:
                self.seen_cards.update(cards)
                try:
                    self.rest_by_pos[int(cur_pos)] = max(0, self.rest_by_pos[int(cur_pos)] - len(cards))
                except (TypeError, ValueError, IndexError):
                    pass
            self.history.append(
                {
                    "pos": cur_pos,
                    "action": cur_action,
                    "cards": cards,
                    "greaterPos": msg.get("greaterPos"),
                    "greaterAction": msg.get("greaterAction"),
                }
            )
        elif stage in {"tribute", "back"} and msg_type == "notify":
            result = msg.get("result") or []
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, list) and len(item) >= 3:
                        self.seen_cards[str(item[2])] += 1
            self.history.append({"stage": stage, "result": result})
        elif stage == "episodeOver" and msg_type == "notify":
            self.history.append(
                {
                    "stage": "episodeOver",
                    "order": msg.get("order"),
                    "curRank": msg.get("curRank"),
                    "restCards": msg.get("restCards"),
                }
            )

    def snapshot(self, context: GameContext) -> BeliefSnapshot:
        rest = list(self.rest_by_pos)
        for pos in range(4):
            observed = safe_rest(context.public_info, pos)
            if observed is not None:
                rest[pos] = observed
        unknown = self.unknown_rank_counts(context.hand_cards)
        unknown_values = list(unknown.values())
        public_vector = self._public_vector(context, rest, unknown)
        node_vectors = {
            node: self._node_vector(node, context, rest, unknown, public_vector)
            for node in NODE_KEYS
        }
        return BeliefSnapshot(
            node_vectors=node_vectors,
            public_vector=public_vector,
            entropy=entropy(unknown_values) if unknown_values else 0.0,
            rest_by_pos=rest,
            unknown_rank_counts=unknown,
        )

    def unknown_rank_counts(self, hand_cards: Sequence[str]) -> Dict[str, int]:
        counts = dict(TOTAL_RANK_COUNTS)
        for card in self.seen_cards:
            rank = card_rank(str(card))
            if rank in counts:
                counts[rank] = max(0, counts[rank] - self.seen_cards[card])
        for card in hand_cards:
            rank = card_rank(str(card))
            if rank in counts:
                counts[rank] = max(0, counts[rank] - 1)
        return counts

    def reachable_belief(self, node: str, context: GameContext) -> List[float]:
        return self.snapshot(context).node_vectors.get(node, [])

    def history_tail(self, limit: int = 12) -> List[Dict[str, Any]]:
        return list(self.history)[-limit:]

    def _public_vector(self, context: GameContext, rest: Sequence[int], unknown: Mapping[str, int]) -> List[float]:
        partner = context.partner_pos
        opponents = context.opponents
        unknown_total = max(1, sum(unknown.values()))
        high_unknown = sum(unknown.get(rank, 0) for rank in ("A", "2", "B", "R")) / unknown_total
        bomb_rank_count = sum(1 for count in unknown.values() if count >= 4)
        pass_streak = 0
        for item in reversed(self.history):
            action = item.get("action")
            if isinstance(action, list) and action and action[0] == "PASS":
                pass_streak += 1
            else:
                break
        return normalize_distribution(
            [
                max(0.0, context.hand_size) / 27.0,
                rest[partner] / 27.0,
                min(rest[pos] for pos in opponents) / 27.0,
                max(rest[pos] for pos in opponents) / 27.0,
                high_unknown,
                bomb_rank_count / 15.0,
                min(pass_streak, 4) / 4.0,
                1.0 if context.partner_winning else 0.0,
                1.0 if context.opponent_winning else 0.0,
            ]
        )

    def _node_vector(
        self,
        node: str,
        context: GameContext,
        rest: Sequence[int],
        unknown: Mapping[str, int],
        public_vector: Sequence[float],
    ) -> List[float]:
        base = list(public_vector)
        if node == "T1_board_parser":
            return normalize_distribution(base[:4] + [1.0 if context.leading else 0.0])
        if node == "T2_history_tracker":
            return normalize_distribution(base[4:] + [min(len(self.history), 64) / 64.0])
        if node == "T3_card_counter":
            high = sum(unknown.get(rank, 0) for rank in ("A", "2", "B", "R"))
            mid = sum(unknown.get(rank, 0) for rank in ("T", "J", "Q", "K"))
            low = sum(unknown.get(rank, 0) for rank in ("3", "4", "5", "6", "7", "8", "9"))
            return normalize_distribution([low, mid, high, sum(1 for count in unknown.values() if count >= 4), 1.0])
        if node == "T4_opponent_intent":
            opp_min = min(rest[pos] for pos in context.opponents)
            return normalize_distribution([27 - opp_min, context.min_opponent_rest <= 5, context.opponent_winning, base[4], 1.0])
        if node == "T5_teammate_intent":
            return normalize_distribution([27 - rest[context.partner_pos], context.partner_winning, context.partner_rest <= 3, base[6], 1.0])
        if node == "T6_macro_evaluator":
            return normalize_distribution([base[0], base[1], base[2], 1.0 if context.hand_size <= 8 else 0.0, 1.0])
        if node == "T7_hand_value":
            pair_pressure = self._pair_pressure(context.hand_cards)
            return normalize_distribution([context.hand_size / 27.0, pair_pressure, base[5], base[4], 1.0])
        return normalize_distribution(base + [1.0])

    def _pair_pressure(self, cards: Sequence[str]) -> float:
        ranks = Counter(card_rank(card) for card in cards)
        singles = sum(1 for count in ranks.values() if count == 1)
        groups = sum(1 for count in ranks.values() if count >= 2)
        return (singles + 1.0) / max(1.0, singles + groups)


def rest_cards_by_pos(rest_cards: Any) -> Dict[int, List[str]]:
    result: Dict[int, List[str]] = {}
    if isinstance(rest_cards, list):
        for item in rest_cards:
            if isinstance(item, list) and len(item) >= 2:
                try:
                    result[int(item[0])] = [str(card) for card in item[1]]
                except (TypeError, ValueError):
                    continue
    return result


def belief_calibration_proxy(snapshot: BeliefSnapshot, rest_cards: Any, my_pos: int) -> float:
    """Offline-only GT calibration proxy from episodeOver restCards.

    The bundled platform exposes only remaining cards for unfinished players,
    not the full hidden hand trajectory. This keeps the L_GT_calib slot and
    uses the available Ground Truth without leaking it into online decisions.
    """

    gt_rest = rest_cards_by_pos(rest_cards)
    if not gt_rest:
        return 0.0
    errors: List[float] = []
    unknown = snapshot.unknown_rank_counts
    total_unknown = max(1, sum(unknown.values()))
    prior_high = sum(unknown.get(rank, 0) for rank in ("A", "2", "B", "R")) / total_unknown
    for pos, cards in gt_rest.items():
        if pos == my_pos:
            continue
        actual_high = sum(1 for card in cards if card_rank(card) in {"A", "2", "B", "R"}) / max(1, len(cards))
        errors.append(abs(actual_high - prior_high))
    return sum(errors) / len(errors) if errors else 0.0

