# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .game import Candidate, GameContext
from .topology import Edge, edge_to_key


@dataclass
class EdgeRecord:
    edge: Edge
    raw_message: str
    parsed_distribution: List[float]
    before_belief: List[float]
    after_belief: List[float]
    weight: float
    info_gain: float
    clarity: float
    edge_error: float

    def to_json(self) -> Dict[str, Any]:
        return {
            "edge": edge_to_key(self.edge),
            "raw_message": self.raw_message,
            "parsed_distribution": self.parsed_distribution,
            "before_belief": self.before_belief,
            "after_belief": self.after_belief,
            "weight": self.weight,
            "info_gain": self.info_gain,
            "clarity": self.clarity,
            "edge_error": self.edge_error,
        }


@dataclass
class DecisionRecord:
    step: int
    context: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    chosen_index: int
    action_distribution: List[float]
    source: str
    edge_records: List[EdgeRecord]
    action_reward: float
    message_reward: float

    def to_json(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "context": self.context,
            "candidates": self.candidates,
            "chosen_index": self.chosen_index,
            "action_distribution": self.action_distribution,
            "source": self.source,
            "edge_records": [record.to_json() for record in self.edge_records],
            "action_reward": self.action_reward,
            "message_reward": self.message_reward,
        }


@dataclass
class EpisodeRecord:
    episode_id: int
    decisions: List[DecisionRecord] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    outcome: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "decisions": [decision.to_json() for decision in self.decisions],
            "observations": self.observations[-256:],
            "outcome": self.outcome,
        }


class TrajectoryLogger:
    def __init__(self, client_id: Optional[int] = None) -> None:
        self.client_id = client_id
        self.episode_id = 0
        self.current = EpisodeRecord(episode_id=self.episode_id)
        trace_path = os.environ.get("COTE_TRACE_PATH", "")
        self.trace_path = Path(trace_path) if trace_path else None

    def observe(self, msg: Mapping[str, Any]) -> None:
        stage = msg.get("stage")
        msg_type = msg.get("type")
        if stage == "beginning" and msg_type == "notify":
            if self.current.decisions or self.current.observations or self.current.outcome:
                self.flush_current()
            self.episode_id += 1
            self.current = EpisodeRecord(episode_id=self.episode_id)
        compact = {
            "stage": stage,
            "type": msg_type,
            "curPos": msg.get("curPos"),
            "curAction": msg.get("curAction"),
            "greaterPos": msg.get("greaterPos"),
            "greaterAction": msg.get("greaterAction"),
            "myPos": msg.get("myPos"),
        }
        self.current.observations.append(compact)

    def record_decision(
        self,
        context: GameContext,
        candidates: List[Candidate],
        chosen_index: int,
        action_distribution: List[float],
        source: str,
        edge_records: List[EdgeRecord],
    ) -> None:
        chosen = next((candidate for candidate in candidates if candidate.index == chosen_index), candidates[0])
        top_score = max(candidate.base_score for candidate in candidates) if candidates else 0.0
        action_reward = (chosen.base_score - top_score) / 100.0
        if chosen.index == candidates[0].index:
            action_reward += 0.05
        if context.partner_winning and chosen.action and chosen.action[0] == "PASS":
            action_reward += 0.05
        if context.opponent_winning and chosen.action and chosen.action[0] != "PASS":
            action_reward += 0.04
        msg_reward = sum(record.weight * (record.info_gain + record.clarity - record.edge_error) for record in edge_records)
        msg_reward /= max(1, len(edge_records))
        record = DecisionRecord(
            step=len(self.current.decisions),
            context={
                "myPos": context.my_pos,
                "partnerPos": context.partner_pos,
                "opponents": list(context.opponents),
                "stage": context.stage,
                "handSize": context.hand_size,
                "greaterPos": context.greater_pos,
                "greaterAction": context.greater_action,
                "rests": [item.get("rest") if isinstance(item, dict) else None for item in context.public_info],
                "curRank": context.cur_rank,
            },
            candidates=[
                {
                    "actIndex": candidate.index,
                    "action": candidate.action,
                    "score": candidate.base_score,
                    "probability": candidate.probability,
                    "nodes": candidate.node_scores,
                    "reason": candidate.reason,
                }
                for candidate in candidates[:12]
            ],
            chosen_index=chosen_index,
            action_distribution=action_distribution,
            source=source,
            edge_records=edge_records,
            action_reward=action_reward,
            message_reward=msg_reward,
        )
        self.current.decisions.append(record)

    def mark_episode_over(self, msg: Mapping[str, Any], my_pos: int) -> Optional[EpisodeRecord]:
        if msg.get("stage") != "episodeOver" or msg.get("type") != "notify":
            return None
        order = list(msg.get("order") or [])
        winner_team = int(order[0]) % 2 if order else None
        my_team = my_pos % 2
        self.current.outcome = {
            "order": order,
            "winnerTeam": winner_team,
            "myTeam": my_team,
            "win": bool(winner_team == my_team) if winner_team is not None else None,
            "curRank": msg.get("curRank"),
            "restCards": msg.get("restCards"),
        }
        completed = self.current
        self.flush_current()
        self.episode_id += 1
        self.current = EpisodeRecord(episode_id=self.episode_id)
        return completed

    def flush_current(self) -> None:
        if not self.trace_path:
            return
        try:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(self.current.to_json(), ensure_ascii=False, separators=(",", ":")) + "\n")
        except OSError:
            pass

