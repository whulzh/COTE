# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .config import HyperParams
from .game import normalize_distribution, softmax
from .nodes import OUTPUT_NODE
from .topology import Edge, edge_to_key, key_to_edge


SEMANTIC_AXES = [
    "finish",
    "block_opponent",
    "help_partner",
    "preserve_bomb",
    "shed_cards",
    "low_ambiguity",
]


@dataclass
class FitnessStats:
    mean: float = 0.0
    std: float = 1.0


@dataclass
class PromptGenome:
    edge: Edge
    vector: List[float]
    reachable_belief: List[float] = field(default_factory=list)
    noise: List[float] = field(default_factory=list)
    sigma: float = 0.08
    fitness: FitnessStats = field(default_factory=FitnessStats)
    generation: int = 0

    def clone(self) -> "PromptGenome":
        return PromptGenome(
            edge=self.edge,
            vector=list(self.vector),
            reachable_belief=list(self.reachable_belief),
            noise=list(self.noise),
            sigma=self.sigma,
            fitness=FitnessStats(self.fitness.mean, self.fitness.std),
            generation=self.generation,
        )

    def to_json(self) -> Dict[str, object]:
        return {
            "edge": edge_to_key(self.edge),
            "vector": [round(value, 8) for value in self.vector],
            "reachable_belief": [round(value, 8) for value in self.reachable_belief],
            "noise": [round(value, 8) for value in self.noise],
            "sigma": self.sigma,
            "fitness": {"mean": self.fitness.mean, "std": self.fitness.std},
            "generation": self.generation,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, object], default_dim: int, default_sigma: float) -> "PromptGenome":
        edge_raw = str(payload.get("edge", "T1_board_parser->T2_history_tracker"))
        vector_raw = payload.get("vector", [])
        vector = [float(value) for value in vector_raw] if isinstance(vector_raw, list) else []
        if not vector:
            vector = [0.0] * default_dim
        fitness_raw = payload.get("fitness", {})
        fitness = FitnessStats()
        if isinstance(fitness_raw, dict):
            fitness.mean = float(fitness_raw.get("mean", 0.0) or 0.0)
            fitness.std = float(fitness_raw.get("std", 1.0) or 1.0)
        return cls(
            edge=key_to_edge(edge_raw),
            vector=vector,
            reachable_belief=[float(value) for value in payload.get("reachable_belief", [])]
            if isinstance(payload.get("reachable_belief", []), list)
            else [],
            noise=[float(value) for value in payload.get("noise", [])]
            if isinstance(payload.get("noise", []), list)
            else [],
            sigma=float(payload.get("sigma", default_sigma) or default_sigma),
            fitness=fitness,
            generation=int(payload.get("generation", 0) or 0),
        )


class PromptPopulation:
    def __init__(self, populations: Optional[Dict[Edge, List[PromptGenome]]] = None) -> None:
        self.populations: Dict[Edge, List[PromptGenome]] = populations or {}
        self.history_elites: Dict[Edge, List[PromptGenome]] = {}

    @classmethod
    def bootstrap(cls, edges: Iterable[Edge], rng: random.Random, hp: HyperParams) -> "PromptPopulation":
        populations: Dict[Edge, List[PromptGenome]] = {}
        for edge in edges:
            items: List[PromptGenome] = []
            base = edge_prior_vector(edge, hp.prompt_dim)
            for _ in range(hp.population_size):
                vector = [clip_prompt(value + rng.gauss(0.0, hp.initial_sigma), hp.prompt_bound) for value in base]
                items.append(PromptGenome(edge=edge, vector=vector, sigma=hp.initial_sigma))
            populations[edge] = items
        return cls(populations)

    @classmethod
    def from_json(cls, payload: Mapping[str, object], hp: HyperParams) -> "PromptPopulation":
        populations: Dict[Edge, List[PromptGenome]] = {}
        raw_pop = payload.get("populations", {})
        if isinstance(raw_pop, dict):
            for edge_key, raw_items in raw_pop.items():
                if not isinstance(raw_items, list):
                    continue
                edge = key_to_edge(str(edge_key))
                populations[edge] = [
                    PromptGenome.from_json(item, hp.prompt_dim, hp.initial_sigma)
                    for item in raw_items
                    if isinstance(item, dict)
                ]
        instance = cls(populations)
        raw_elites = payload.get("history_elites", {})
        if isinstance(raw_elites, dict):
            for edge_key, raw_items in raw_elites.items():
                if not isinstance(raw_items, list):
                    continue
                edge = key_to_edge(str(edge_key))
                instance.history_elites[edge] = [
                    PromptGenome.from_json(item, hp.prompt_dim, hp.initial_sigma)
                    for item in raw_items
                    if isinstance(item, dict)
                ]
        return instance

    def to_json(self) -> Dict[str, object]:
        return {
            "populations": {
                edge_to_key(edge): [item.to_json() for item in items]
                for edge, items in sorted(self.populations.items())
            },
            "history_elites": {
                edge_to_key(edge): [item.to_json() for item in items[-20:]]
                for edge, items in sorted(self.history_elites.items())
            },
        }

    def ensure_edges(self, edges: Iterable[Edge], rng: random.Random, hp: HyperParams) -> None:
        for edge in edges:
            if edge in self.populations and len(self.populations[edge]) >= hp.population_size:
                continue
            base = edge_prior_vector(edge, hp.prompt_dim)
            items = self.populations.setdefault(edge, [])
            while len(items) < hp.population_size:
                vector = [clip_prompt(value + rng.gauss(0.0, hp.initial_sigma), hp.prompt_bound) for value in base]
                items.append(PromptGenome(edge=edge, vector=vector, sigma=hp.initial_sigma))

    def best(self, edge: Edge) -> PromptGenome:
        items = self.populations.get(edge, [])
        if not items:
            raise KeyError(edge)
        return max(items, key=lambda item: item.fitness.mean)

    def best_by_edge(self) -> Dict[Edge, PromptGenome]:
        return {edge: self.best(edge) for edge in self.populations if self.populations[edge]}

    def active_prompt_vectors(self) -> Dict[Edge, List[float]]:
        return {edge: self.best(edge).vector for edge in self.populations if self.populations[edge]}

    def inject_gradient_individual(self, genome: PromptGenome, hp: HyperParams) -> None:
        items = self.populations.setdefault(genome.edge, [])
        if not items:
            items.append(genome.clone())
            return
        worst_index = min(range(len(items)), key=lambda idx: items[idx].fitness.mean)
        if genome.fitness.mean >= items[worst_index].fitness.mean:
            bounded = genome.clone()
            bounded.vector = [clip_prompt(value, hp.prompt_bound) for value in bounded.vector]
            items[worst_index] = bounded

    def remember_elites(self, edge: Edge, elites: Sequence[PromptGenome], max_history: int = 100) -> None:
        history = self.history_elites.setdefault(edge, [])
        history.extend(item.clone() for item in elites)
        history.sort(key=lambda item: item.fitness.mean, reverse=True)
        del history[max_history:]


def edge_prior_vector(edge: Edge, dim: int) -> List[float]:
    src, dst = edge
    values = [0.0] * dim
    axis = edge_target_distribution(edge)
    for idx, value in enumerate(axis):
        if idx < dim:
            values[idx] = 0.8 * (value - 1.0 / len(axis))
    if dst == OUTPUT_NODE:
        for idx in range(len(axis), min(dim, len(axis) + 8)):
            values[idx] = 0.15
    if "card_counter" in src:
        values[3] += 0.35
    if "opponent" in src:
        values[1] += 0.35
    if "teammate" in src:
        values[2] += 0.35
    return values


def edge_target_distribution(edge: Edge) -> List[float]:
    src, dst = edge
    scores = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    if "card_counter" in src:
        scores[1] += 0.8
        scores[3] += 0.7
    if "opponent" in src:
        scores[1] += 1.0
    if "teammate" in src:
        scores[2] += 1.0
    if "macro" in src or "macro" in dst:
        scores[0] += 0.6
        scores[2] += 0.3
    if "hand_value" in src:
        scores[4] += 0.9
        scores[3] += 0.3
    if dst == OUTPUT_NODE:
        scores[0] += 0.8
        scores[5] += 0.5
    return normalize_distribution(scores)


def prompt_distribution(vector: Sequence[float]) -> List[float]:
    head = list(vector[: len(SEMANTIC_AXES)])
    if len(head) < len(SEMANTIC_AXES):
        head.extend([0.0] * (len(SEMANTIC_AXES) - len(head)))
    return softmax(head, temperature=0.8)


def decode_soft_prompt(vector: Sequence[float], edge: Edge, compact: bool = True) -> str:
    """Map a continuous edge prompt to a compact textual fallback."""

    dist = prompt_distribution(vector)
    ranked = sorted(zip(SEMANTIC_AXES, dist), key=lambda item: item[1], reverse=True)
    src, dst = edge
    keys = ",".join(axis for axis, _ in ranked[:3])
    values = ";".join(f"{axis}:{prob:.2f}" for axis, prob in ranked[:4])
    if compact:
        return f"[EDGE {src}->{dst}; keys={keys}; {values}; format=JSON-microcode]"
    return (
        f"Edge {src}->{dst}. Use a compact, low-ambiguity Guandan belief message. "
        f"Prioritize {keys}. Continuous prompt projection: {values}."
    )


def slerp(vec_a: Sequence[float], vec_b: Sequence[float], weight_b: float) -> List[float]:
    if len(vec_a) != len(vec_b):
        size = min(len(vec_a), len(vec_b))
        vec_a = vec_a[:size]
        vec_b = vec_b[:size]
    norm_a = l2_norm(vec_a)
    norm_b = l2_norm(vec_b)
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return [(1.0 - weight_b) * a + weight_b * b for a, b in zip(vec_a, vec_b)]
    unit_a = [value / norm_a for value in vec_a]
    unit_b = [value / norm_b for value in vec_b]
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(unit_a, unit_b))))
    omega = math.acos(dot)
    if abs(omega) < 1e-6:
        return [(1.0 - weight_b) * a + weight_b * b for a, b in zip(vec_a, vec_b)]
    sin_omega = math.sin(omega)
    scale_a = math.sin((1.0 - weight_b) * omega) / sin_omega
    scale_b = math.sin(weight_b * omega) / sin_omega
    radius = (1.0 - weight_b) * norm_a + weight_b * norm_b
    unit = [scale_a * a + scale_b * b for a, b in zip(unit_a, unit_b)]
    return [radius * value for value in unit]


def mutate_vector(
    vector: Sequence[float],
    sigma: float,
    reachable_belief: Sequence[float],
    rng: random.Random,
    hp: HyperParams,
) -> Tuple[List[float], List[float]]:
    entropy_gain = belief_entropy_gate(reachable_belief)
    noise = [rng.gauss(0.0, 1.0) for _ in vector]
    mutated = [
        clip_prompt(value + sigma * entropy_gain * noise_value, hp.prompt_bound)
        for value, noise_value in zip(vector, noise)
    ]
    return mutated, noise


def belief_entropy_gate(reachable_belief: Sequence[float]) -> float:
    if not reachable_belief:
        return 1.0
    dist = normalize_distribution([abs(value) for value in reachable_belief])
    ent = -sum(value * math.log(max(value, 1e-12)) for value in dist)
    max_ent = math.log(max(len(dist), 2))
    normalized = ent / max_ent if max_ent > 0 else 0.0
    return 0.5 + normalized


def update_sigma(sigma: float, accepted: bool, hp: HyperParams) -> float:
    factor = hp.sigma_success_up if accepted else hp.sigma_success_down
    return max(hp.sigma_min, min(hp.sigma_max, sigma * factor))


def clip_prompt(value: float, bound: float) -> float:
    return max(-bound, min(bound, float(value)))


def l2_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def cosine_similarity(a_values: Sequence[float], b_values: Sequence[float]) -> float:
    size = min(len(a_values), len(b_values))
    if size <= 0:
        return 0.0
    a = a_values[:size]
    b = b_values[:size]
    denom = l2_norm(a) * l2_norm(b)
    if denom <= 1e-12:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / denom


def vector_mean(vectors: Iterable[Sequence[float]], dim: int) -> List[float]:
    total = [0.0] * dim
    count = 0
    for vector in vectors:
        count += 1
        for idx, value in enumerate(vector[:dim]):
            total[idx] += value
    if count <= 0:
        return total
    return [value / count for value in total]
