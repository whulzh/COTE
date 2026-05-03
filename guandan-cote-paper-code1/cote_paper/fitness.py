# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .belief import BeliefSnapshot, belief_calibration_proxy
from .config import HyperParams
from .game import entropy, kl_divergence, normalize_distribution
from .message_semantics import parse_message_distribution
from .prompts import edge_target_distribution, prompt_distribution
from .topology import CoteTopologyState, Edge
from .trajectory import EdgeRecord, EpisodeRecord


@dataclass
class EdgeMetrics:
    edge_error: float = 0.0
    info_gain: float = 0.0
    clarity: float = 0.0
    downstream: float = 0.0
    samples: int = 0


@dataclass
class FitnessBreakdown:
    total: float
    topo_fitness: float
    prompt_fitness_sum: float
    action_reward: float
    message_reward: float
    decision_reward: float
    gt_calibration_loss: float
    edge_metrics: Dict[Edge, EdgeMetrics] = field(default_factory=dict)


class CoteFitnessEvaluator:
    """Computes COTE fitness channels from platform trajectories."""

    def __init__(self, hp: HyperParams, rng: random.Random) -> None:
        self.hp = hp
        self.rng = rng
        self.last_breakdown: Optional[FitnessBreakdown] = None

    def evaluate_episode(
        self,
        episode: EpisodeRecord,
        topology: CoteTopologyState,
        prompt_vectors: Mapping[Edge, Sequence[float]],
        final_snapshot: Optional[BeliefSnapshot] = None,
    ) -> FitnessBreakdown:
        edge_metrics = self.edge_metrics_from_episode(episode, topology)
        action_reward = self.discounted_action_reward(episode) if self.hp.reward_channel else 0.0
        message_reward = self.discounted_message_reward(episode)
        decision_reward = self.decision_reward(episode) if self.hp.reward_channel else 0.0
        gt_loss = 0.0
        if self.hp.belief_channel and final_snapshot is not None and episode.outcome is not None:
            gt_loss = belief_calibration_proxy(final_snapshot, episode.outcome.get("restCards"), int(episode.outcome.get("myTeam", 0)))

        expected_return = action_reward + message_reward + decision_reward
        topo_quality = 0.0
        info_quality = 0.0
        for edge in topology.all_edges():
            weight = topology.weight(edge)
            metrics = edge_metrics.get(edge, EdgeMetrics())
            if self.hp.error_channel:
                topo_quality += weight * math.exp(-self.hp.gamma_q * metrics.edge_error)
            if self.hp.belief_channel:
                info_quality += weight * metrics.info_gain
        topo_fitness = expected_return + self.hp.lambda_q * topo_quality + self.hp.lambda_i * info_quality - self.hp.eta_gt * gt_loss

        prompt_sum = 0.0
        for edge, vector in prompt_vectors.items():
            metrics = edge_metrics.get(edge, EdgeMetrics())
            prompt_sum += self.edge_prompt_fitness(edge, vector, topology.weight(edge), metrics)

        topo_regularizer = sum(abs(topology.weight(edge)) for edge in topology.all_edges()) / 64.0
        prompt_regularizer = sum(sum(value * value for value in vector) for vector in prompt_vectors.values())
        prompt_regularizer /= max(1, sum(len(vector) for vector in prompt_vectors.values()))
        total = (
            self.hp.alpha_t * topo_fitness
            + self.hp.alpha_p * prompt_sum
            - self.hp.lambda_b * topo_regularizer
            - self.hp.lambda_p * prompt_regularizer
        )
        breakdown = FitnessBreakdown(
            total=total,
            topo_fitness=topo_fitness,
            prompt_fitness_sum=prompt_sum,
            action_reward=action_reward,
            message_reward=message_reward,
            decision_reward=decision_reward,
            gt_calibration_loss=gt_loss,
            edge_metrics=edge_metrics,
        )
        self.last_breakdown = breakdown
        return breakdown

    def edge_metrics_from_episode(self, episode: EpisodeRecord, topology: CoteTopologyState) -> Dict[Edge, EdgeMetrics]:
        buckets: Dict[Edge, List[EdgeRecord]] = {edge: [] for edge in topology.all_edges()}
        for decision in episode.decisions:
            for record in decision.edge_records:
                buckets.setdefault(record.edge, []).append(record)
        metrics: Dict[Edge, EdgeMetrics] = {}
        for edge, records in buckets.items():
            if not records:
                metrics[edge] = EdgeMetrics(edge_error=0.25, info_gain=0.0, clarity=0.0, downstream=0.0, samples=0)
                continue
            edge_error = sum(record.edge_error for record in records) / len(records)
            info_gain = sum(record.info_gain for record in records) / len(records)
            clarity = sum(record.clarity for record in records) / len(records)
            downstream = self.downstream_contribution(edge, topology)
            metrics[edge] = EdgeMetrics(edge_error, info_gain, clarity, downstream, len(records))
        return metrics

    def edge_prompt_fitness(self, edge: Edge, vector: Sequence[float], weight: float, metrics: EdgeMetrics) -> float:
        # f_e^COTE = w_ij[-L_e + lambda_I^(e) I_e] + lambda_d J_downstream
        #            + lambda_c Clarity - lambda_l ||p_e||^2
        prompt_dist = prompt_distribution(vector)
        target_dist = edge_target_distribution(edge)
        prompt_error = kl_divergence(target_dist, prompt_dist)
        norm = sum(value * value for value in vector) / max(1, len(vector))
        fidelity = -(0.7 * metrics.edge_error + 0.3 * prompt_error) if self.hp.error_channel else 0.0
        info_gain = metrics.info_gain if self.hp.belief_channel else 0.0
        return (
            weight * (fidelity + self.hp.lambda_edge_i * info_gain)
            + self.hp.lambda_d * metrics.downstream
            + self.hp.lambda_c * metrics.clarity
            - self.hp.lambda_l * norm
        )

    def evaluate_candidate_prompt(
        self,
        edge: Edge,
        vector: Sequence[float],
        topology: CoteTopologyState,
        metrics: EdgeMetrics,
        current_total: float,
        samples: Optional[int] = None,
    ) -> tuple[float, float]:
        sample_count = max(1, samples or self.hp.repeated_samples)
        values: List[float] = []
        base = self.edge_prompt_fitness(edge, vector, topology.weight(edge), metrics)
        target = edge_target_distribution(edge)
        dist = prompt_distribution(vector)
        semantic_error = kl_divergence(target, dist)
        for _ in range(sample_count):
            noise = self.rng.gauss(0.0, 0.02 + 0.03 * semantic_error)
            values.append(current_total + base + noise)
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        return mean, math.sqrt(max(variance, 1e-9))

    def topology_gradients(self, topology: CoteTopologyState, breakdown: FitnessBreakdown) -> Dict[Edge, float]:
        gradients: Dict[Edge, float] = {}
        reward_channel = breakdown.action_reward + breakdown.decision_reward if self.hp.reward_channel else 0.0
        for edge in topology.all_edges():
            metrics = breakdown.edge_metrics.get(edge, EdgeMetrics())
            error_channel = -metrics.edge_error if self.hp.error_channel else 0.0
            belief_channel = metrics.info_gain - breakdown.gt_calibration_loss if self.hp.belief_channel else 0.0
            sparsity = -0.03 if topology.weight(edge) < 0.05 else 0.0
            gradients[edge] = reward_channel * metrics.downstream + error_channel + belief_channel + sparsity
        return gradients

    def downstream_contribution(self, edge: Edge, topology: CoteTopologyState) -> float:
        src, dst = edge
        if dst == "T8_action_decider":
            return topology.weight(edge)
        direct_to_output = topology.weight((dst, "T8_action_decider"))
        two_hop = 0.0
        for _, middle_dst in topology.outbound_edges(dst):
            if middle_dst == "T8_action_decider":
                continue
            two_hop += topology.weight((dst, middle_dst)) * topology.weight((middle_dst, "T8_action_decider"))
        return max(direct_to_output, two_hop)

    def discounted_action_reward(self, episode: EpisodeRecord) -> float:
        total = 0.0
        discount = 1.0
        for decision in episode.decisions:
            total += discount * decision.action_reward
            discount *= self.hp.gamma_reward
        return total / max(1, len(episode.decisions))

    def discounted_message_reward(self, episode: EpisodeRecord) -> float:
        total = 0.0
        discount = 1.0
        for decision in episode.decisions:
            total += discount * decision.message_reward
            discount *= self.hp.gamma_reward
        return total / max(1, len(episode.decisions))

    def decision_reward(self, episode: EpisodeRecord) -> float:
        if not episode.outcome:
            return 0.0
        win = episode.outcome.get("win")
        order = episode.outcome.get("order") or []
        if win is True:
            reward = 1.0
        elif win is False:
            reward = -1.0
        else:
            reward = 0.0
        if isinstance(order, list) and len(order) == 4:
            my_team = int(episode.outcome.get("myTeam", 0))
            team_positions = [pos for pos in order if int(pos) % 2 == my_team]
            if team_positions:
                avg_finish = sum(order.index(pos) for pos in team_positions) / len(team_positions)
                reward += (1.5 - avg_finish) / 3.0
        return reward


def edge_record_from_message(
    edge: Edge,
    raw_message: str,
    parsed_distribution: Sequence[float],
    before_belief: Sequence[float],
    after_belief: Sequence[float],
    weight: float,
    strict_parse: bool = False,
) -> EdgeRecord:
    if strict_parse:
        parsed_distribution = parse_message_distribution(raw_message)
    info_gain = kl_divergence(after_belief, before_belief)
    clarity = clarity_score(parsed_distribution)
    target = edge_target_distribution(edge)
    edge_error = kl_divergence(target, parsed_distribution)
    if "{" not in raw_message and "[" not in raw_message:
        edge_error += 0.08
    return EdgeRecord(
        edge=edge,
        raw_message=raw_message,
        parsed_distribution=list(normalize_distribution(parsed_distribution)),
        before_belief=list(normalize_distribution(before_belief)),
        after_belief=list(normalize_distribution(after_belief)),
        weight=weight,
        info_gain=info_gain,
        clarity=clarity,
        edge_error=edge_error,
    )


def clarity_score(distribution: Sequence[float]) -> float:
    dist = normalize_distribution(distribution)
    if not dist:
        return 0.0
    max_entropy = math.log(len(dist))
    if max_entropy <= 0:
        return 1.0
    return 1.0 - entropy(dist) / max_entropy
