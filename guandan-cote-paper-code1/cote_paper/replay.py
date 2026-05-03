# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ReplayStep:
    step: int
    context: Dict[str, Any]
    action: int
    edge_messages: Dict[str, str] = field(default_factory=dict)
    node_outputs: Dict[str, Any] = field(default_factory=dict)
    action_distribution: List[float] = field(default_factory=list)
    log_prob: float = 0.0

    def to_json(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "context": self.context,
            "action": self.action,
            "edge_messages": self.edge_messages,
            "node_outputs": self.node_outputs,
            "action_distribution": self.action_distribution,
            "log_prob": self.log_prob,
        }


class TrajectoryReplay:
    """Replay buffer used by strict COTE candidate evaluation."""

    def __init__(self) -> None:
        self.steps: List[ReplayStep] = []

    def add_step(self, step: ReplayStep) -> None:
        self.steps.append(step)

    def extend(self, steps: List[ReplayStep]) -> None:
        self.steps.extend(steps)

    def to_json(self) -> Dict[str, Any]:
        return {"steps": [step.to_json() for step in self.steps]}

    @classmethod
    def from_episode_record(cls, episode: Any) -> "TrajectoryReplay":
        replay = cls()
        for decision in getattr(episode, "decisions", []):
            edge_messages = {
                f"{record.edge[0]}->{record.edge[1]}": record.raw_message
                for record in getattr(decision, "edge_records", [])
            }
            replay.add_step(
                ReplayStep(
                    step=int(getattr(decision, "step", len(replay.steps))),
                    context=dict(getattr(decision, "context", {}) or {}),
                    action=int(getattr(decision, "chosen_index", 0)),
                    edge_messages=edge_messages,
                    action_distribution=list(getattr(decision, "action_distribution", []) or []),
                )
            )
        return replay
