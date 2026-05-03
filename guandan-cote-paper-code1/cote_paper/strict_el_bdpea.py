# -*- coding: utf-8 -*-
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .config import HyperParams
from .el_bdpea import ELBDPEA
from .fitness import CoteFitnessEvaluator, EdgeMetrics, FitnessBreakdown
from .prompts import PromptGenome, PromptPopulation
from .topology import CoteTopologyState, Edge


class StrictELBDPEA(ELBDPEA):
    """Algorithm 1 variant that uses strict candidate replay sampling counts."""

    def __init__(self, hp: HyperParams, rng: random.Random) -> None:
        super().__init__(hp, rng)

    def candidate_replay_count(self) -> int:
        return max(1, int(self.hp.strict_candidate_replays))

    def _evaluate_population(
        self,
        edge: Edge,
        items: List[PromptGenome],
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
                samples=self.candidate_replay_count(),
            )
            item.fitness.mean = mean - self._crowding_penalty(item, items)
            item.fitness.std = std

    def evolve(
        self,
        population: PromptPopulation,
        topology: CoteTopologyState,
        evaluator: CoteFitnessEvaluator,
        breakdown: FitnessBreakdown,
        reachable_beliefs: Dict[str, List[float]],
    ) -> Dict[Edge, Tuple[PromptGenome, float]]:
        return super().evolve(population, topology, evaluator, breakdown, reachable_beliefs)
