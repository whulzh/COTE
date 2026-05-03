# -*- coding: utf-8 -*-
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from .config import HyperParams
from .fitness import CoteFitnessEvaluator, EdgeMetrics, FitnessBreakdown
from .prompts import (
    FitnessStats,
    PromptGenome,
    PromptPopulation,
    cosine_similarity,
    mutate_vector,
    slerp,
    update_sigma,
)
from .topology import CoteTopologyState, Edge


class ELBDPEA:
    """Edge-Level Bidirectional Prompt Evolution Algorithm.

    This follows Algorithm 1: per-edge population evaluation with the joint
    fitness, elite selection/history, fitness-aware SLERP crossover, reachable
    belief guided Gaussian mutation, M repeated samples, 1.5 sigma acceptance,
    success-based step-size adaptation, and Lamarckian gradient injection.
    """

    def __init__(self, hp: HyperParams, rng: random.Random) -> None:
        self.hp = hp
        self.rng = rng

    def evolve(
        self,
        population: PromptPopulation,
        topology: CoteTopologyState,
        evaluator: CoteFitnessEvaluator,
        breakdown: FitnessBreakdown,
        reachable_beliefs: Dict[str, List[float]],
    ) -> Dict[Edge, Tuple[PromptGenome, float]]:
        best_by_edge: Dict[Edge, Tuple[PromptGenome, float]] = {}
        current_total = breakdown.total
        for edge in topology.all_edges():
            items = population.populations.get(edge)
            if not items:
                continue
            metrics = breakdown.edge_metrics.get(edge, EdgeMetrics())
            self._evaluate_population(edge, items, topology, evaluator, metrics, current_total)

            elite_count = max(1, int(round(self.hp.elite_ratio * len(items))))
            items.sort(key=lambda item: item.fitness.mean, reverse=True)
            elites = [item.clone() for item in items[:elite_count]]
            population.remember_elites(edge, elites)

            child_items: List[PromptGenome] = [item.clone() for item in elites]
            while len(child_items) < self.hp.population_size:
                parent_a = self._tournament(items)
                parent_b = self._tournament(items)
                child = self._make_child(parent_a, parent_b, reachable_beliefs.get(edge[0], []))
                mean, std = evaluator.evaluate_candidate_prompt(
                    edge,
                    child.vector,
                    topology,
                    metrics,
                    current_total,
                    samples=self.hp.repeated_samples,
                )
                child.fitness = FitnessStats(mean, std)
                parent = parent_a if parent_a.fitness.mean >= parent_b.fitness.mean else parent_b
                accepted = mean > parent.fitness.mean + self.hp.suspicion_delta * parent.fitness.std
                child.sigma = update_sigma(parent.sigma, accepted, self.hp)
                if accepted:
                    child_items.append(child)
                else:
                    fallback = parent.clone()
                    fallback.sigma = update_sigma(parent.sigma, False, self.hp)
                    child_items.append(fallback)

            population.populations[edge] = child_items[: self.hp.population_size]
            best = max(population.populations[edge], key=lambda item: item.fitness.mean)
            best_by_edge[edge] = (best.clone(), best.fitness.std)
        return best_by_edge

    def inject_trained_elites(
        self,
        population: PromptPopulation,
        topology: CoteTopologyState,
        breakdown: FitnessBreakdown,
        trainer: object,
    ) -> None:
        train_population = getattr(trainer, "train_population", None)
        if callable(train_population):
            train_population(population, topology, breakdown)

    def _evaluate_population(
        self,
        edge: Edge,
        items: Sequence[PromptGenome],
        topology: CoteTopologyState,
        evaluator: CoteFitnessEvaluator,
        metrics: EdgeMetrics,
        current_total: float,
    ) -> None:
        for item in items:
            mean, std = evaluator.evaluate_candidate_prompt(
                edge,
                item.vector,
                topology,
                metrics,
                current_total,
                samples=self.hp.repeated_samples,
            )
            item.fitness = FitnessStats(mean - self._crowding_penalty(item, items), std)

    def _tournament(self, items: Sequence[PromptGenome], k: int = 3) -> PromptGenome:
        sample_size = min(k, len(items))
        competitors = self.rng.sample(list(items), sample_size)
        return max(competitors, key=lambda item: item.fitness.mean)

    def _make_child(self, parent_a: PromptGenome, parent_b: PromptGenome, reachable_belief: List[float]) -> PromptGenome:
        child = parent_a.clone()
        if self.rng.random() < self.hp.crossover_rate:
            total = parent_a.fitness.mean + parent_b.fitness.mean
            if abs(total) <= 1e-9:
                weight_b = 0.5
            else:
                weight_b = max(0.05, min(0.95, parent_b.fitness.mean / total))
            child.vector = slerp(parent_a.vector, parent_b.vector, weight_b)
        if self.rng.random() < self.hp.mutation_rate:
            child.vector, child.noise = mutate_vector(child.vector, parent_a.sigma, reachable_belief, self.rng, self.hp)
        child.reachable_belief = list(reachable_belief)
        child.generation = max(parent_a.generation, parent_b.generation) + 1
        return child

    def _crowding_penalty(self, item: PromptGenome, items: Sequence[PromptGenome]) -> float:
        if len(items) <= 1:
            return 0.0
        similarities = [cosine_similarity(item.vector, other.vector) for other in items if other is not item]
        if not similarities:
            return 0.0
        return 0.01 * max(0.0, sum(similarities) / len(similarities))
