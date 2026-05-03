# -*- coding: utf-8 -*-
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

from .belief import BeliefSnapshot, belief_calibration_proxy
from .config import HyperParams
from .fitness import CoteFitnessEvaluator, FitnessBreakdown, edge_record_from_message
from .message_semantics import parse_message_distribution
from .prompts import prompt_distribution
from .replay import TrajectoryReplay
from .topology import CoteTopologyState, Edge
from .trajectory import EdgeRecord, EpisodeRecord


@dataclass
class StrictFitnessArtifacts:
    replay: TrajectoryReplay
    gt_calibration_loss: float
    prompt_gradients: Dict[Edge, Sequence[float]]


class StrictCoteFitnessEvaluator(CoteFitnessEvaluator):
    """Equation-level wrapper for strict COTE reproduction mode.

    It re-parses messages through `q_phi`, evaluates on a replay object, and
    exposes prompt/topology artifacts needed by Algorithm 2 Phase D.
    """

    def __init__(self, hp: HyperParams, rng: random.Random) -> None:
        super().__init__(hp, rng)
        self.last_artifacts: Optional[StrictFitnessArtifacts] = None

    def strict_edge_record(self, record: EdgeRecord) -> EdgeRecord:
        parsed = parse_message_distribution(record.raw_message)
        return edge_record_from_message(
            record.edge,
            record.raw_message,
            parsed,
            record.before_belief,
            record.after_belief,
            record.weight,
            strict_parse=True,
        )

    def evaluate_episode(
        self,
        episode: EpisodeRecord,
        topology: CoteTopologyState,
        prompt_vectors: Mapping[Edge, Sequence[float]],
        final_snapshot: Optional[BeliefSnapshot] = None,
    ) -> FitnessBreakdown:
        strict_episode = self._episode_with_q_phi_records(episode)
        breakdown = super().evaluate_episode(strict_episode, topology, prompt_vectors, final_snapshot)
        replay = TrajectoryReplay.from_episode_record(strict_episode)
        gt_loss = 0.0
        if self.hp.belief_channel and final_snapshot is not None and strict_episode.outcome is not None:
            gt_loss = belief_calibration_proxy(final_snapshot, strict_episode.outcome.get("restCards"), int(strict_episode.outcome.get("myTeam", 0)))
        self.last_artifacts = StrictFitnessArtifacts(
            replay=replay,
            gt_calibration_loss=gt_loss,
            prompt_gradients=self.prompt_gradients(prompt_vectors, breakdown),
        )
        return breakdown

    def _episode_with_q_phi_records(self, episode: EpisodeRecord) -> EpisodeRecord:
        for decision in episode.decisions:
            decision.edge_records = [self.strict_edge_record(record) for record in decision.edge_records]
        return episode

    def prompt_gradients(
        self,
        prompt_vectors: Mapping[Edge, Sequence[float]],
        breakdown: FitnessBreakdown,
    ) -> Dict[Edge, Sequence[float]]:
        gradients: Dict[Edge, Sequence[float]] = {}
        for edge, vector in prompt_vectors.items():
            metrics = breakdown.edge_metrics.get(edge)
            if metrics is None:
                continue
            dist = prompt_distribution(vector)
            # Proxy for d(-KL(target || prompt))/dp_head. The soft prompt
            # trainer can still supply LM gradients; this keeps strict Phase D
            # explicit and deterministic for vector-space reproduction tests.
            gradients[edge] = [
                (metrics.info_gain - metrics.edge_error) * (1.0 - value)
                for value in dist
            ] + [0.0] * max(0, len(vector) - len(dist))
        return gradients
